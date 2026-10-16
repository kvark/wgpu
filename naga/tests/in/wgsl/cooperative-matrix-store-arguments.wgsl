enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read_write> c: array<f32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;
@compute @workgroup_size(32)
fn main() {
    let result = coop_mat8x8<f32,C>();
    coopStoreT(result, &c[params.x + params.y], params.z + params.w);
}
