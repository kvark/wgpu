use std::{
    env,
    mem::size_of,
    num::NonZeroU64,
    process,
    time::{Duration, Instant},
};
use wgpu::util::DeviceExt as _;

const COMPUTE_WORKGROUP_SIZE: u32 = 64;
const MAX_PASS_COUNT: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Workload {
    ComputeIndependent,
    ComputeChain,
    GraphicsIndependent,
    GraphicsChain,
}

impl Workload {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "compute-independent" => Ok(Self::ComputeIndependent),
            "compute-chain" => Ok(Self::ComputeChain),
            "graphics-independent" => Ok(Self::GraphicsIndependent),
            "graphics-chain" => Ok(Self::GraphicsChain),
            _ => Err(format!("unknown workload: {value}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ComputeIndependent => "compute-independent",
            Self::ComputeChain => "compute-chain",
            Self::GraphicsIndependent => "graphics-independent",
            Self::GraphicsChain => "graphics-chain",
        }
    }

    fn is_compute(self) -> bool {
        matches!(self, Self::ComputeIndependent | Self::ComputeChain)
    }
}

#[derive(Clone, Debug)]
struct Config {
    workload: Workload,
    passes: usize,
    elements: u32,
    rounds: u32,
    width: u32,
    height: u32,
    warmups: usize,
    samples: usize,
    validation: bool,
    allow_software: bool,
    gpu_timing: bool,
    list_adapters: bool,
    capture: bool,
    shader_checks: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            workload: Workload::ComputeIndependent,
            passes: 16,
            elements: 1 << 20,
            rounds: 8,
            width: 1024,
            height: 1024,
            warmups: 10,
            samples: 30,
            validation: false,
            allow_software: false,
            gpu_timing: true,
            list_adapters: false,
            capture: false,
            shader_checks: false,
        }
    }
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--workload" => {
                    config.workload = Workload::parse(&next_value(&mut args, "--workload")?)?
                }
                "--policy" => {
                    let value = next_value(&mut args, "--policy")?;
                    if value != "tracked" {
                        return Err(format!("wgpu only supports --policy tracked, got {value}"));
                    }
                }
                "--passes" => {
                    config.passes = parse_value(&next_value(&mut args, "--passes")?, "--passes")?
                }
                "--elements" => {
                    config.elements =
                        parse_value(&next_value(&mut args, "--elements")?, "--elements")?
                }
                "--rounds" => {
                    config.rounds = parse_value(&next_value(&mut args, "--rounds")?, "--rounds")?
                }
                "--width" => {
                    config.width = parse_value(&next_value(&mut args, "--width")?, "--width")?
                }
                "--height" => {
                    config.height = parse_value(&next_value(&mut args, "--height")?, "--height")?
                }
                "--warmups" => {
                    config.warmups = parse_value(&next_value(&mut args, "--warmups")?, "--warmups")?
                }
                "--samples" => {
                    config.samples = parse_value(&next_value(&mut args, "--samples")?, "--samples")?
                }
                "--validation" => config.validation = true,
                "--allow-software" => config.allow_software = true,
                "--no-gpu-timing" => config.gpu_timing = false,
                "--list-adapters" => config.list_adapters = true,
                "--capture" => config.capture = true,
                "--shader-checks" => config.shader_checks = true,
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }

        if config.passes == 0 || config.passes >= MAX_PASS_COUNT {
            return Err(format!("--passes must be in 1..{MAX_PASS_COUNT}"));
        }
        if config.elements == 0 {
            return Err("--elements must be nonzero".into());
        }
        if config.width == 0 || config.height == 0 {
            return Err("--width and --height must be nonzero".into());
        }
        if config.samples == 0 {
            return Err("--samples must be nonzero".into());
        }
        Ok(config)
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, argument: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {argument}"))
}

fn parse_value<T: std::str::FromStr>(value: &str, argument: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {argument}: {value}"))
}

fn print_usage() {
    println!(
        "\
wgpu synchronization benchmark

Usage:
  cargo run --release -p wgpu-sync-bench -- [options]

Options:
  --workload <name>   compute-independent, compute-chain,
                      graphics-independent, or graphics-chain
  --policy tracked    accepted for matrix-runner compatibility
  --passes <count>    passes per measured command buffer (default: 16)
  --elements <count>  u32 elements in compute buffers (default: 1048576)
  --rounds <count>    shader mixing rounds per invocation (default: 8)
  --width <pixels>    render-target width (default: 1024)
  --height <pixels>   render-target height (default: 1024)
  --warmups <count>   unreported warm-up iterations (default: 10)
  --samples <count>   reported iterations (default: 30)
  --validation        enable backend validation
  --allow-software    permit a software adapter (correctness only)
  --no-gpu-timing     disable timestamp queries for CPU-only collection
  --list-adapters     list selectable adapters and exit
  --shader-checks     keep wgpu's injected bounds, division and loop checks
                      (off by default, to match Blade's shader compilation)
  --capture           wrap one measured iteration in a RenderDoc capture
                      (requires librenderdoc.so to be loaded, e.g. via
                      LD_PRELOAD)
  -h, --help          show this help

Adapter selection:
  WGPU_BACKEND=vulkan|metal
  WGPU_ADAPTER_NAME=<case-insensitive substring>
"
    );
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputeParams {
    element_count: u32,
    rounds: u32,
    seed: u32,
    _padding: u32,
}

struct ComputeBench {
    pipeline: wgpu::ComputePipeline,
    buffers: Vec<wgpu::Buffer>,
    bind_groups: Vec<wgpu::BindGroup>,
    element_count: u32,
    rounds: u32,
    independent: bool,
}

struct ValidationReadback {
    buffer: wgpu::Buffer,
    total_size: u64,
    stride: usize,
    valid_size: usize,
    count: usize,
}

impl ValidationReadback {
    fn hash(&self, bytes: &[u8]) -> u64 {
        assert_eq!(bytes.len(), self.total_size as usize);
        let mut logical = Vec::with_capacity(self.valid_size * self.count);
        for (index, item) in bytes.chunks_exact(self.stride).take(self.count).enumerate() {
            let output = &item[..self.valid_size];
            assert!(
                output.iter().any(|&byte| byte != 0),
                "validation output {index} contains only zero bytes"
            );
            logical.extend_from_slice(output);
        }
        fnv1a64(&logical)
    }
}

/// Compiles a shader with the same runtime checks Blade's pipelines use.
///
/// Blade hands naga `BoundsCheckPolicies::default()` -- which is `Unchecked`
/// throughout -- and `emit_int_div_checks: false`, so its SPIR-V carries no
/// injected checks at all. `create_shader_module` would give the wgpu side
/// clamped indices, division guards, and a bounded-loop counter that is loaded,
/// compared and stored on every iteration of the mixing loop both shaders run.
/// That is a shader-code difference, not a synchronization difference, and it
/// would land in the device-time comparison as if it were one.
///
/// `--shader-checks` restores wgpu's defaults so the cost can be measured.
fn create_shader(
    device: &wgpu::Device,
    config: &Config,
    descriptor: wgpu::ShaderModuleDescriptor<'_>,
) -> wgpu::ShaderModule {
    if config.shader_checks {
        device.create_shader_module(descriptor)
    } else {
        // SAFETY: both shaders index only in-bounds, divide by nothing, and
        // loop a bounded number of times set by a uniform the harness writes.
        unsafe {
            device.create_shader_module_trusted(descriptor, wgpu::ShaderRuntimeChecks::unchecked())
        }
    }
}

impl ComputeBench {
    fn new(device: &wgpu::Device, config: &Config) -> Self {
        let shader = create_shader(device, config, wgpu::include_wgsl!("compute.wgsl"));
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sync-bench-compute-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sync-bench-compute-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: size_of::<ComputeParams>() as u32,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sync-bench-compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let independent = config.workload == Workload::ComputeIndependent;
        let buffer_count = if independent { config.passes + 1 } else { 2 };
        let buffer_size = u64::from(config.elements) * 4;
        let buffers = (0..buffer_count)
            .map(|index| {
                let contents = vec![(index + 1) as u8; buffer_size as usize];
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sync-bench-compute-buffer"),
                    contents: &contents,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                })
            })
            .collect::<Vec<_>>();
        let bind_groups = (0..config.passes)
            .map(|pass_index| {
                let (input_index, output_index) = if independent {
                    (0, pass_index + 1)
                } else {
                    (pass_index % 2, (pass_index + 1) % 2)
                };
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sync-bench-compute-bind-group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffers[input_index].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffers[output_index].as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        Self {
            pipeline,
            buffers,
            bind_groups,
            element_count: config.elements,
            rounds: config.rounds,
            independent,
        }
    }

    fn record_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass_index: usize,
        pass_count: usize,
        iteration: usize,
        query_set: Option<&wgpu::QuerySet>,
    ) {
        let timestamp_writes = query_set.and_then(|query_set| {
            let beginning_of_pass_write_index = (pass_index == 0).then_some(0);
            let end_of_pass_write_index = (pass_index + 1 == pass_count).then_some(1);
            (beginning_of_pass_write_index.is_some() || end_of_pass_write_index.is_some())
                .then_some(wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index,
                    end_of_pass_write_index,
                })
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sync-bench-compute-pass"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_groups[pass_index], &[]);
        let params = ComputeParams {
            element_count: self.element_count,
            rounds: self.rounds,
            seed: (iteration as u32)
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(pass_index as u32),
            _padding: 0,
        };
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(self.element_count.div_ceil(COMPUTE_WORKGROUP_SIZE), 1, 1);
    }

    fn validation_buffer(&self, device: &wgpu::Device, passes: usize) -> ValidationReadback {
        let stride = u64::from(self.element_count.min(1024)) * 4;
        let count = if self.independent { passes } else { 1 };
        let total_size = stride * count as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sync-bench-compute-readback"),
            size: total_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        ValidationReadback {
            buffer,
            total_size,
            stride: stride as usize,
            valid_size: stride as usize,
            count,
        }
    }

    fn encode_validation_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        readback: &ValidationReadback,
        passes: usize,
    ) {
        if self.independent {
            for output_index in 1..=passes {
                encoder.copy_buffer_to_buffer(
                    &self.buffers[output_index],
                    0,
                    &readback.buffer,
                    (output_index - 1) as u64 * readback.stride as u64,
                    readback.valid_size as u64,
                );
            }
        } else {
            encoder.copy_buffer_to_buffer(
                &self.buffers[passes % 2],
                0,
                &readback.buffer,
                0,
                readback.valid_size as u64,
            );
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GraphicsParams {
    rounds: u32,
    seed: u32,
    _padding_a: u32,
    _padding_b: u32,
}

struct GraphicsBench {
    pipeline: wgpu::RenderPipeline,
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
    width: u32,
    rounds: u32,
    independent: bool,
}

impl GraphicsBench {
    fn new(device: &wgpu::Device, config: &Config) -> Self {
        let shader = create_shader(device, config, wgpu::include_wgsl!("graphics.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sync-bench-graphics-pipeline-layout"),
            bind_group_layouts: &[],
            immediate_size: size_of::<GraphicsParams>() as u32,
        });
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sync-bench-graphics"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState {
                        color: additive,
                        alpha: additive,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let independent = config.workload == Workload::GraphicsIndependent;
        let target_count = if independent { config.passes } else { 1 };
        let textures = (0..target_count)
            .map(|_| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("sync-bench-target"),
                    size: wgpu::Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
            })
            .collect::<Vec<_>>();
        let views = textures
            .iter()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();

        Self {
            pipeline,
            textures,
            views,
            width: config.width,
            rounds: config.rounds,
            independent,
        }
    }

    fn record_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass_index: usize,
        pass_count: usize,
        iteration: usize,
        query_set: Option<&wgpu::QuerySet>,
    ) {
        let view_index = if self.independent { pass_index } else { 0 };
        let load = if self.independent || pass_index == 0 {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        } else {
            wgpu::LoadOp::Load
        };
        let timestamp_writes = query_set.and_then(|query_set| {
            let beginning_of_pass_write_index = (pass_index == 0).then_some(0);
            let end_of_pass_write_index = (pass_index + 1 == pass_count).then_some(1);
            (beginning_of_pass_write_index.is_some() || end_of_pass_write_index.is_some())
                .then_some(wgpu::RenderPassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index,
                    end_of_pass_write_index,
                })
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sync-bench-graphics-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.views[view_index],
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        let params = GraphicsParams {
            rounds: self.rounds,
            seed: (iteration as u32)
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(pass_index as u32),
            _padding_a: 0,
            _padding_b: 0,
        };
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.draw(0..3, 0..1);
    }

    fn validation_buffer(&self, device: &wgpu::Device) -> ValidationReadback {
        let unpadded = self.width * 4;
        let bytes_per_row = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let count = if self.independent {
            self.textures.len()
        } else {
            1
        };
        let total_size = u64::from(bytes_per_row) * count as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sync-bench-graphics-readback"),
            size: total_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        ValidationReadback {
            buffer,
            total_size,
            stride: bytes_per_row as usize,
            valid_size: unpadded as usize,
            count,
        }
    }

    fn encode_validation_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        readback: &ValidationReadback,
    ) {
        for (index, texture) in self.textures[..readback.count].iter().enumerate() {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: index as u64 * readback.stride as u64,
                        bytes_per_row: Some(readback.stride as u32),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

enum Bench {
    Compute(ComputeBench),
    Graphics(GraphicsBench),
}

impl Bench {
    fn new(device: &wgpu::Device, config: &Config) -> Self {
        if config.workload.is_compute() {
            Self::Compute(ComputeBench::new(device, config))
        } else {
            Self::Graphics(GraphicsBench::new(device, config))
        }
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        config: &Config,
        iteration: usize,
        query_set: Option<&wgpu::QuerySet>,
    ) {
        for pass_index in 0..config.passes {
            match *self {
                Self::Compute(ref bench) => {
                    bench.record_pass(encoder, pass_index, config.passes, iteration, query_set)
                }
                Self::Graphics(ref bench) => {
                    bench.record_pass(encoder, pass_index, config.passes, iteration, query_set)
                }
            }
        }
    }

    fn validate(&self, device: &wgpu::Device, queue: &wgpu::Queue, passes: usize) -> u64 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sync-bench-validation"),
        });
        let readback = match *self {
            Self::Compute(ref bench) => {
                let readback = bench.validation_buffer(device, passes);
                bench.encode_validation_copy(&mut encoder, &readback, passes);
                readback
            }
            Self::Graphics(ref bench) => {
                let readback = bench.validation_buffer(device);
                bench.encode_validation_copy(&mut encoder, &readback);
                readback
            }
        };
        queue.submit([encoder.finish()]);
        let bytes = read_buffer(device, &readback.buffer, readback.total_size);
        readback.hash(&bytes)
    }
}

struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    timestamp_period: f32,
}

impl GpuTimer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("sync-bench-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sync-bench-timestamp-resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sync-bench-timestamp-readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            timestamp_period: queue.get_timestamp_period(),
        }
    }

    fn finish(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(&self.resolve_buffer, 0, &self.readback_buffer, 0, 16);
    }

    fn read(&self) -> u64 {
        let view = self.readback_buffer.slice(..).get_mapped_range().unwrap();
        let timestamps = bytemuck::cast_slice::<u8, u64>(&view);
        let ticks = timestamps[1].saturating_sub(timestamps[0]);
        drop(view);
        self.readback_buffer.unmap();
        ((ticks as f64) * f64::from(self.timestamp_period)).min(u64::MAX as f64) as u64
    }
}

fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let slice = buffer.slice(..size);
    slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let view = slice.get_mapped_range().unwrap();
    let bytes = view.to_vec();
    drop(view);
    buffer.unmap();
    bytes
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Wraps one measured iteration in a RenderDoc capture, when the library is
/// present in the process.
///
/// RenderDoc normally delimits captures at swapchain presents, and this
/// benchmark is headless, so the capture has to be requested explicitly. The
/// library has to be loaded already, which the study's `capture-streams.py`
/// arranges with `LD_PRELOAD`; without it `RenderDoc::new` fails and the run
/// continues uncaptured rather than aborting a measurement.
#[cfg(any(target_os = "linux", target_os = "windows"))]
struct Capture(Option<renderdoc::RenderDoc<renderdoc::V141>>);

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl Capture {
    fn new(enabled: bool, template: &str) -> Self {
        if !enabled {
            return Self(None);
        }
        match renderdoc::RenderDoc::<renderdoc::V141>::new() {
            Ok(mut api) => {
                api.set_capture_file_path_template(template);
                Self(Some(api))
            }
            Err(error) => {
                eprintln!(
                    "warning: --capture requested but RenderDoc is not loaded ({error}); \
                     preload librenderdoc.so to capture"
                );
                Self(None)
            }
        }
    }

    fn begin(&mut self) {
        if let Some(ref mut api) = self.0 {
            api.start_frame_capture(std::ptr::null(), std::ptr::null());
        }
    }

    fn end(&mut self) {
        if let Some(ref mut api) = self.0 {
            api.end_frame_capture(std::ptr::null(), std::ptr::null());
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
struct Capture;

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
impl Capture {
    fn new(enabled: bool, _template: &str) -> Self {
        if enabled {
            eprintln!("warning: --capture is not supported on this platform");
        }
        Self
    }
    fn begin(&mut self) {}
    fn end(&mut self) {}
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn csv_string(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn default_backends() -> wgpu::Backends {
    if cfg!(any(target_os = "macos", target_os = "ios")) {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    }
}

fn backend_name(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Noop => "noop",
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gles",
        wgpu::Backend::BrowserWebGpu => "webgpu",
    }
}

fn select_adapter(instance: &wgpu::Instance, backends: wgpu::Backends) -> wgpu::Adapter {
    if env::var_os("WGPU_ADAPTER_NAME").is_some() {
        return pollster::block_on(wgpu::util::initialize_adapter_from_env(instance, None))
            .expect("WGPU_ADAPTER_NAME did not match an adapter");
    }
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .unwrap_or_else(|error| {
        eprintln!("error: failed to select adapter for {backends:?}: {error}");
        process::exit(2);
    })
}

fn main() {
    #[cfg(feature = "tracy")]
    tracy_client::Client::start();

    let config = Config::parse().unwrap_or_else(|error| {
        eprintln!("error: {error}\n");
        print_usage();
        process::exit(2);
    });

    let backends = wgpu::Backends::from_env().unwrap_or_else(default_backends);
    let flags = if config.validation {
        wgpu::InstanceFlags::debugging()
    } else {
        wgpu::InstanceFlags::empty()
    };
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    if config.list_adapters {
        for adapter in pollster::block_on(instance.enumerate_adapters(backends)) {
            let info = adapter.get_info();
            println!(
                "0x{:04x}:0x{:04x}\t{}\t{:?}\t{}\t{}",
                info.vendor,
                info.device,
                info.name,
                info.device_type,
                info.driver,
                info.driver_info,
            );
        }
        return;
    }
    let adapter = select_adapter(&instance, backends);
    let adapter_info = adapter.get_info();
    let software_emulated = adapter_info.device_type == wgpu::DeviceType::Cpu;
    if software_emulated && !config.allow_software {
        eprintln!(
            "error: {} is a software adapter; pass --allow-software for correctness-only runs",
            adapter_info.name
        );
        process::exit(2);
    }

    let mut required_features = wgpu::Features::IMMEDIATES;
    if config.gpu_timing {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    if !adapter.features().contains(required_features) {
        eprintln!(
            "error: adapter is missing required features: {:?}",
            required_features - adapter.features()
        );
        process::exit(2);
    }
    let required_limits = wgpu::Limits {
        max_immediate_size: 16,
        ..wgpu::Limits::default()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sync-bench-device"),
        required_features,
        required_limits,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .unwrap_or_else(|error| {
        eprintln!("error: failed to create device: {error}");
        process::exit(2);
    });

    let bench = Bench::new(&device, &config);
    let timer = config.gpu_timing.then(|| GpuTimer::new(&device, &queue));

    println!("# schema,blade-sync-bench-v1");
    println!("# implementation,wgpu");
    println!("# wgpu_version,{}", env!("CARGO_PKG_VERSION"));
    println!("# backend,{}", backend_name(adapter_info.backend));
    println!("# device_name,{}", csv_string(&adapter_info.name));
    println!("# vendor_id,0x{:04x}", adapter_info.vendor);
    println!("# device_id,0x{:04x}", adapter_info.device);
    println!("# device_type,{:?}", adapter_info.device_type);
    println!(
        "# device_pci_bus_id,{}",
        csv_string(&adapter_info.device_pci_bus_id)
    );
    println!("# driver_name,{}", csv_string(&adapter_info.driver));
    println!("# driver_info,{}", csv_string(&adapter_info.driver_info));
    println!("# software_emulated,{software_emulated}");
    println!("# validation,{}", config.validation);
    // Recorded so an analysis can tell a collection taken with wgpu's injected
    // bounds, division and loop checks from one taken without them. They are
    // worth tens of percent of GPU span on a fragment-bound workload, so the
    // two are not comparable and must not be pooled.
    println!("# shader_checks,{}", config.shader_checks);
    println!("# gpu_timing,{}", config.gpu_timing);
    println!(
        "# timestamp_period_ns,{}",
        timer.as_ref().map_or(0.0, |timer| timer.timestamp_period)
    );
    println!(
        "sample,workload,policy,passes,elements,rounds,width,height,start_ns,record_ns,submit_ns,wait_ns,gpu_ns,gpu_pass_count"
    );

    // Capture a single warmed iteration, matching Blade's benchmark: the first
    // one after the warmups, so pipelines and bind groups are established and
    // the command stream is the steady-state one the timings describe.
    let capture_iteration = config.warmups;
    let mut capture = Capture::new(
        config.capture,
        &format!("wgpu-sync-bench__{}__tracked", config.workload.as_str()),
    );

    for iteration in 0..config.warmups + config.samples {
        profiling::scope!("sync-bench iteration");

        if config.capture && iteration == capture_iteration {
            capture.begin();
        }

        let start_begin = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sync-bench"),
        });
        let start_time = start_begin.elapsed();

        let record_begin = Instant::now();
        bench.record(
            &mut encoder,
            &config,
            iteration,
            timer.as_ref().map(|timer| &timer.query_set),
        );
        let record_time = record_begin.elapsed();

        let submit_begin = Instant::now();
        if let Some(ref timer) = timer {
            timer.finish(&mut encoder);
        }
        let submission = queue.submit([encoder.finish()]);
        let submit_time = submit_begin.elapsed();

        let wait_begin = Instant::now();
        if let Some(ref timer) = timer {
            timer
                .readback_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |result| result.unwrap());
        }
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        let wait_time = wait_begin.elapsed();
        let gpu_ns = timer.as_ref().map_or(0, GpuTimer::read);

        if iteration >= config.warmups {
            println!(
                "{},{},tracked,{},{},{},{},{},{},{},{},{},{},{}",
                iteration - config.warmups,
                config.workload.as_str(),
                config.passes,
                config.elements,
                config.rounds,
                config.width,
                config.height,
                duration_ns(start_time),
                duration_ns(record_time),
                duration_ns(submit_time),
                duration_ns(wait_time),
                gpu_ns,
                if config.gpu_timing { config.passes } else { 0 },
            );
        }

        if config.capture && iteration == capture_iteration {
            capture.end();
        }
    }

    let validation_hash = bench.validate(&device, &queue, config.passes);
    println!("# validation_hash,fnv1a64-standard:{validation_hash:016x}");
}

#[cfg(test)]
mod tests {
    use super::{fnv1a64, Config, Workload};

    #[test]
    fn workload_names_match_blade() {
        let names = [
            "compute-independent",
            "compute-chain",
            "graphics-independent",
            "graphics-chain",
        ];
        for name in names {
            assert_eq!(Workload::parse(name).unwrap().as_str(), name);
        }
        assert_eq!(Config::default().passes, 16);
    }

    #[test]
    fn validation_hash_is_fnv1a64() {
        // Published FNV-1a-64 test vector for the ASCII string "hello".
        assert_eq!(fnv1a64(b"hello"), 0xa430_d846_80aa_bd0b);
    }
}
