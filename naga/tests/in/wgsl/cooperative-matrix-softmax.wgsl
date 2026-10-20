enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage> a: array<f16>;
@group(0) @binding(1) var<storage> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
struct Params { m: u32, n: u32, k: u32 }
var<immediate> params: Params;

fn combine_max(left: f32, right: f32) -> f32 {
    return max(left, right);
}
fn combine_sum(left: f32, right: f32) -> f32 {
    return left + right;
}
fn mask(row: u32, col: u32, value: f32, width: u32) -> f32 {
    return select(-1e30, value, col < width);
}
fn exponent(row: u32, col: u32, value: f32, maximum: f32) -> f32 {
    return exp(value - maximum);
}
fn normalize(row: u32, col: u32, value: f32, sum: f32) -> f32 {
    return value / sum;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group: vec3u) {
    let row = group.x * 64u;
    let col = group.y * 64u;
    var sum = coop_mat64x64<f32,C,workgroup>();
    for (var t = 0u; t < params.k; t += 32u) {
        let left = coopLoadTensorT<coop_mat32x64<f16,A,workgroup>>(
            &a, vec2u(params.m, params.k), vec2u(params.k, 1u), vec2i(i32(row), i32(t)));
        let right = coopLoadTensorT<coop_mat64x32<f16,B,workgroup>>(
            &b, vec2u(params.k, params.n), vec2u(params.n, 1u), vec2i(i32(t), i32(col)));
        sum = coopMultiplyAdd(left, right, sum);
    }
    let masked = coopMap(sum, mask, min(params.n - col, 64u));
    let maximum = coopReduceRows<coop_mat64x64<f32,C,workgroup>>(masked, combine_max);
    let numerator = coopMap(masked, exponent, maximum);
    let denominator = coopReduceRows<coop_mat64x64<f32,C,workgroup>>(numerator, combine_sum);
    let result = coopMap(numerator, normalize, denominator);
    coopStoreTensorT(result, &c, vec2u(params.m, params.n), vec2u(params.n, 1u), vec2i(i32(row), i32(col)));
}
