enable wgpu_bfloat16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage> a: array<bf16>;
@group(0) @binding(1) var<storage> b: array<bf16>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
var<private> result: coop_mat256x128<f32, C, workgroup>;

fn load_a(offset: u32) -> coop_mat64x128<bf16, A, workgroup> {
    return coopLoadT<coop_mat64x128<bf16, A, workgroup>>(&a[offset], 64u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>) {
    let left = load_a(group.x * 8192u);
    let right = coopLoadT<coop_mat256x64<bf16, B, workgroup>>(&b[0], 256u);
    result = coopMultiplyAdd(left, right, coop_mat256x128<f32, C, workgroup>());
    coopStoreT(result, &c[0], 256u);
}
