enable f16;

@group(0) @binding(0)
var<storage> src: array<vec4<f16>>;
@group(0) @binding(1)
var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2)
var<uniform> params: vec4<u32>;
var<workgroup> tile: array<vec4<f16>, 64>;
var<workgroup> pairs: array<vec2<f32>, 64>;
var<workgroup> triples: array<vec3<f32>, 64>;

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(local_invocation_index) lane: u32) {
    let _e5 = src[lane];
    tile[lane] = _e5;
    workgroupBarrier();
    let _e10 = params.x;
    let a = coopLoadT<coop_mat8x8<f16,A>>((&tile[2]), (_e10 + 1u));
    let _e17 = params.y;
    let _e21 = params.x;
    let b = coopLoad<coop_mat8x8<f16,B>>((&src[_e17]), _e21);
    let c = coopLoadT<coop_mat8x8<f32,C>>((&dst[4]), 3u);
    let d = coopMultiplyAdd(a, b, c);
    let _e31 = params.z;
    let _e35 = params.x;
    coopStoreT(d, (&dst[_e31]), (_e35 + 1u));
    coopStore(d, (&pairs[2]), 6u);
    workgroupBarrier();
    let e = coopLoad<coop_mat8x8<f32,C>>((&pairs[2]), 6u);
    coopStoreT(e, (&triples[1]), 4u);
    workgroupBarrier();
    let f = coopLoadT<coop_mat8x8<f32,C>>((&triples[1]), 4u);
    coopStore(f, (&dst[4]), 3u);
    return;
}
