use wgpu::util::DeviceExt;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

const BUFFER_ELEMENTS: usize = 128;
const ROW_STRIDE: usize = 12;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(COOPERATIVE_MATRIX_VECTOR_MEMORY);
}

#[apply(gpu_test!)]
static COOPERATIVE_MATRIX_VECTOR_MEMORY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX),
    )
    .run_async(|ctx| async move {
        if !ctx.adapter.cooperative_matrix_properties().iter().any(|p| {
            p.m_size == 8
                && p.n_size == 8
                && p.k_size == 8
                && p.ab_type == wgpu::CooperativeScalarType::F32
                && p.cr_type == wgpu::CooperativeScalarType::F32
        }) {
            return;
        }

        for width in [2, 3, 4] {
            let source =
                include_str!("cooperative_matrix.wgsl").replace("$WIDTH", &width.to_string());
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("cooperative vector memory"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let pipeline = ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &shader,
                    entry_point: None,
                    compilation_options: Default::default(),
                    cache: None,
                });
            let input: Vec<f32> = (0..BUFFER_ELEMENTS).map(|i| i as f32 + 1.0).collect();
            let mut expected = vec![-1.0f32; BUFFER_ELEMENTS];
            let element_stride = if width == 3 { 4 } else { width };
            for row in 0..8 {
                for col in 0..8 {
                    expected[3 * element_stride + row * ROW_STRIDE + col] =
                        input[element_stride + row * ROW_STRIDE + col];
                }
            }
            let src = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&input),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let dst = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[-1.0f32; BUFFER_ELEMENTS]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
            let params = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[(ROW_STRIDE / width - 1) as u32, 1, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit([encoder.finish()]);
            let (send, receive) = std::sync::mpsc::channel();
            wgpu::util::DownloadBuffer::read_buffer(
                &ctx.device,
                &ctx.queue,
                &dst.slice(..),
                move |result| {
                    let data = result.unwrap();
                    send.send(bytemuck::cast_slice::<_, f32>(&data).to_vec())
                        .unwrap();
                },
            );
            ctx.async_poll(wgpu::PollType::wait_indefinitely())
                .await
                .unwrap();
            assert_eq!(receive.try_recv().unwrap(), expected, "vec{width}");
        }
    });
