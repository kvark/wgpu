enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0)
var<storage, read_write> ab: array<f16>;
@group(0) @binding(1)
var<storage, read_write> accum: array<f32>;

@compute @workgroup_size(32, 1, 1)
fn main() {
    let a = coopLoad<coop_mat16x16<f16,A>>((&ab[0]), 16u);
    let b = coopLoad<coop_mat16x16<f16,B>>((&ab[256]), 16u);
    let partial = coopMultiplyAdd(a, b, coop_mat16x16<f16,C>());
    let wide = coop_mat16x16<f32,C>(partial);
    coopStore(wide, (&accum[0]), 16u);
    let narrow = coop_mat16x16<f16,C>(wide);
    coopStore(narrow, (&ab[512]), 16u);
    return;
}
