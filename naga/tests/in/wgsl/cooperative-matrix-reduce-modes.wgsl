enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read_write> data: array<f32>;

fn combine(left: f32, right: f32) -> f32 {
    return left + right;
}
fn scale(row: u32, col: u32, value: f32, multiplier: f32, offset: vec2f) -> f32 {
    return value * multiplier + f32(row) * offset.x + f32(col) * offset.y;
}
fn half_sum(left: f16, right: f16) -> f16 {
    return left + right;
}
fn half_scale(row: u32, col: u32, value: f16, factor: f16) -> f16 {
    return value * factor;
}

@compute @workgroup_size(32)
fn main() {
    let input = coopLoadTensorT<coop_mat32x32<f32,C,workgroup>>(&data, vec2u(32), vec2u(32, 1), vec2i(0));
    let rows = coopReduceRows<coop_mat16x32<f32,C,workgroup>>(input, combine);
    let columns = coopReduceColumns<coop_mat32x16<f32,C,workgroup>>(input, combine);
    let all = coopReduceAll<coop_mat16x16<f32,C,workgroup>>(input, combine);
    let quad = coopReduceQuad<coop_mat16x16<f32,C,workgroup>>(input, combine);
    let mapped = coopMap(quad, scale, 2, vec2f(1, -1));
    coopStoreTensorT(rows, &data, vec2u(32), vec2u(32, 1), vec2i(0));
    coopStoreTensorT(columns, &data, vec2u(32), vec2u(32, 1), vec2i(0));
    coopStoreTensorT(all, &data, vec2u(32), vec2u(32, 1), vec2i(0));
    coopStoreTensorT(mapped, &data, vec2u(32), vec2u(32, 1), vec2i(0));
    let half = coop_mat16x16<f16,C>();
    let half_reduced = coopReduceAll<coop_mat16x16<f16,C>>(half, half_sum);
    let half_mapped = coopMap(half_reduced, half_scale, 2);
    _ = half_mapped;
}
