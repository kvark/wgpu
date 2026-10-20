enable f16;
enable wgpu_cooperative_matrix;

struct Packed4 { values: u32 }
@group(0) @binding(0) var<storage> input: array<Packed4>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;

fn decode(packed: Packed4, block: vec2u, element: vec2u) -> f16 {
    return f16(extractBits(bitcast<i32>(packed.values), element.y * 8u, 8u));
}
fn decode4(packed: Packed4, block: vec2u, element: vec2u) -> vec4h {
    let q = bitcast<i32>(packed.values);
    return vec4h(f16(extractBits(q, 0u, 8u)), f16(extractBits(q, 8u, 8u)),
        f16(extractBits(q, 16u, 8u)), f16(extractBits(q, 24u, 8u)));
}
@compute @workgroup_size(32)
fn main() {
    let tile = coopLoadTensorBlockT<coop_mat16x16<f16,C>, 1u, 4u>(
        &input, vec2u(16u), vec2u(4u, 1u), vec2i(0), decode, decode4);
    coopStoreTensorT(tile, &output, vec2u(16u), vec2u(16u, 1u), vec2i(0));
}
