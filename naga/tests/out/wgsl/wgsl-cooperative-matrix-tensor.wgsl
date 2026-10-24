enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0)
var<storage> a: array<f32>;
@group(0) @binding(1)
var<storage> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> c: array<f32>;
@group(0) @binding(3)
var<uniform> dimensions: vec4<u32>;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>) {
    let m = dimensions.x;
    let n = dimensions.y;
    let k = dimensions.z;
    let left = coopLoadTensorT<coop_mat32x64<f32,C,workgroup>>((&a), vec2<u32>(m, k), vec2<u32>(k, 1u), vec2<i32>(i32((group.y * 64u)), -1i));
    let right = coopLoadTensor<coop_mat64x32<f32,C,workgroup>>((&b), vec2<u32>(n, k), vec2<u32>(k, 1u), vec2<i32>(i32((group.x * 64u)), -1i));
    let left_half = coopConvert<coop_mat32x64<f16,A,workgroup>>(left);
    let right_half = coopConvert<coop_mat64x32<f16,B,workgroup>>(right);
    let product = coopMultiplyAdd(left_half, right_half, coop_mat64x64<f32,C,workgroup>());
    coopStoreTensorT(product, (&c), vec2<u32>(m, n), vec2<u32>(n, 1u), vec2<i32>((group.yx * 64u)));
    return;
}
