enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0)
var<storage, read_write> data: array<f32>;

fn combine(left: f32, right: f32) -> f32 {
    return (left + right);
}

fn scale(row: u32, col: u32, value: f32, multiplier: f32, offset: vec2<f32>) -> f32 {
    return (((value * multiplier) + (f32(row) * offset.x)) + (f32(col) * offset.y));
}

fn half_sum(left_1: f16, right_1: f16) -> f16 {
    return (left_1 + right_1);
}

fn half_scale(row_1: u32, col_1: u32, value_1: f16, factor: f16) -> f16 {
    return (value_1 * factor);
}

@compute @workgroup_size(32, 1, 1)
fn main() {
    let input = coopLoadTensorT<coop_mat32x32<f32,C,workgroup>>((&data), vec2(32u), vec2<u32>(32u, 1u), vec2(0i));
    let rows = coopReduceRows<coop_mat16x32<f32,C,workgroup>>(input, combine);
    let columns = coopReduceColumns<coop_mat32x16<f32,C,workgroup>>(input, combine);
    let all_ = coopReduceAll<coop_mat16x16<f32,C,workgroup>>(input, combine);
    let quad = coopReduceQuad<coop_mat16x16<f32,C,workgroup>>(input, combine);
    let mapped = coopMap(quad, scale, 2f, vec2<f32>(1f, -1f));
    coopStoreTensorT(rows, (&data), vec2(32u), vec2<u32>(32u, 1u), vec2(0i));
    coopStoreTensorT(columns, (&data), vec2(32u), vec2<u32>(32u, 1u), vec2(0i));
    coopStoreTensorT(all_, (&data), vec2(32u), vec2<u32>(32u, 1u), vec2(0i));
    coopStoreTensorT(mapped, (&data), vec2(32u), vec2<u32>(32u, 1u), vec2(0i));
    let half_reduced = coopReduceAll<coop_mat16x16<f16,C>>(coop_mat16x16<f16,C>(), half_sum);
    let phony = coopMap(half_reduced, half_scale, 2h);
    return;
}
