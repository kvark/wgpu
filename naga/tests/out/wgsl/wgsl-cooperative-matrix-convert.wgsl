enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0)
var<storage> input: array<f32>;
@group(0) @binding(1)
var<storage, read_write> output: array<f16>;

@compute @workgroup_size(128, 1, 1)
fn main() {
    let value = coopLoadT<coop_mat32x64<f32,C,workgroup>>((&input[0]), 32u);
    let narrowed = coopConvert<coop_mat32x64<f16,C,workgroup>>(value);
    let same_type = coopConvert<coop_mat32x64<f16,C,workgroup>>(narrowed);
    let operand = coopConvert<coop_mat32x64<f16,A,workgroup>>(same_type);
    let transposed = coopTranspose<coop_mat64x32<f16,B,workgroup>>(narrowed);
    coopStoreT(operand, (&output[0]), 32u);
    coopStore(transposed, (&output[2048]), 32u);
    return;
}
