enable wgpu_cooperative_matrix;

alias Element = vec$WIDTH<f32>;
@group(0) @binding(0) var<storage> src: array<Element>;
@group(0) @binding(1) var<storage, read_write> dst: array<Element>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
var<workgroup> tile: array<Element, 64>;

@compute @workgroup_size(32)
fn main() {
    let a = coopLoadT<coop_mat8x8<f32, C>>(&src[1], params.x + params.y);
    coopStore(a, &tile[2], params.x + params.y);
    workgroupBarrier();
    let b = coopLoad<coop_mat8x8<f32, C>>(&tile[2], params.x + params.y);
    coopStoreT(b, &dst[3], params.x + params.y);
}
