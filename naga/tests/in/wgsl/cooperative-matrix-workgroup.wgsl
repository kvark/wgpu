enable f16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage> a: array<f16>;
@group(0) @binding(1) var<storage> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32) {
    let left = coopLoadT<coop_mat32x64<f16,A,workgroup>>(&a[0], 32u);
    let right = coopLoadT<coop_mat64x32<f16,B,workgroup>>(&b[0], 64u);
    let zero = coop_mat64x64<f32,C,workgroup>();
    let result = coopMultiplyAdd(left, right, zero);
    coopStoreT(result, &c[params.x + params.y], params.z + params.w);
}
