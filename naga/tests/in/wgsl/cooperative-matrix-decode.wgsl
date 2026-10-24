enable f16;
enable wgpu_cooperative_matrix;

struct BlockQ8 {
    scale: u32,
    quant: array<u32, 8>,
}

@group(0) @binding(0) var<storage> a: array<f16>;
@group(0) @binding(1) var<storage> b: array<BlockQ8>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
struct Params { m: u32, n: u32, k: u32 }
var<immediate> params: Params;

fn scale_value(value: i32, scale: u32) -> f16 {
    return f16(f32(value) * unpack2x16float(scale).x);
}

fn decode_q8(block: BlockQ8, block_coord: vec2u, element: vec2u) -> f16 {
    let packed = bitcast<i32>(block.quant[element.y / 4u]);
    return scale_value(extractBits(packed, (element.y % 4u) * 8u, 8u), block.scale);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group: vec3u) {
    let row = i32(group.x * 64u);
    let col = i32(group.y * 64u);
    var sum = coop_mat64x64<f32,C,workgroup>();
    for (var t = 0u; t < params.k; t += 32u) {
        let left = coopLoadTensorT<coop_mat32x64<f16,A,workgroup>>(
            &a, vec2u(params.m, params.k), vec2u(params.k, 1u), vec2i(row, i32(t)));
        let right = coopLoadTensorBlock<coop_mat64x32<f16,B,workgroup>, 1u, 32u>(
            &b, vec2u(params.n, params.k), vec2u((params.k + 31u) / 32u, 1u),
            vec2i(col, i32(t)), decode_q8);
        sum = coopMultiplyAdd(left, right, sum);
    }
    coopStoreTensorT(sum, &c, vec2u(params.m, params.n), vec2u(params.n, 1u), vec2i(row, col));
}
