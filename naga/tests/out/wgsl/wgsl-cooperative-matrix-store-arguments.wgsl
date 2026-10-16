enable wgpu_cooperative_matrix;

@group(0) @binding(0)
var<storage, read_write> c: array<f32>;
@group(0) @binding(1)
var<uniform> params: vec4<u32>;

@compute @workgroup_size(32, 1, 1)
fn main() {
    let _e4 = params.x;
    let _e7 = params.y;
    let _e12 = params.z;
    let _e15 = params.w;
    coopStoreT(coop_mat8x8<f32,C>(), (&c[(_e4 + _e7)]), (_e12 + _e15));
    return;
}
