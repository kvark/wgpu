enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage> src: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
var<workgroup> tile: array<vec4<f16>, 64>;
var<workgroup> pairs: array<vec2<f32>, 64>;
var<workgroup> triples: array<vec3<f32>, 64>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) lane: u32) {
    tile[lane] = src[lane];
    workgroupBarrier();

    let a = coopLoadT<coop_mat8x8<f16, A>>(&tile[2], params.x + 1u);
    let b = coopLoad<coop_mat8x8<f16, B>>(&src[params.y], params.x);
    let c = coopLoadT<coop_mat8x8<f32, C>>(&dst[4], 3u);
    let d = coopMultiplyAdd(a, b, c);
    coopStoreT(d, &dst[params.z], params.x + 1u);
    coopStore(d, &pairs[2], 6u);
    workgroupBarrier();
    let e = coopLoad<coop_mat8x8<f32, C>>(&pairs[2], 6u);
    coopStoreT(e, &triples[1], 4u);
    workgroupBarrier();
    let f = coopLoadT<coop_mat8x8<f32, C>>(&triples[1], 4u);
    coopStore(f, &dst[4], 3u);
}
