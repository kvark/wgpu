enable f16;
enable wgpu_cooperative_matrix;

struct Packed4_ {
    values: u32,
}

@group(0) @binding(0)
var<storage> input: array<Packed4_>;
@group(0) @binding(1)
var<storage, read_write> output: array<f16>;

fn decode(packed: Packed4_, block: vec2<u32>, element: vec2<u32>) -> f16 {
    return f16(extractBits(bitcast<i32>(packed.values), (element.y * 8u), 8u));
}

fn decode4_(packed_1: Packed4_, block_1: vec2<u32>, element_1: vec2<u32>) -> vec4<f16> {
    let q = bitcast<i32>(packed_1.values);
    return vec4<f16>(f16(extractBits(q, 0u, 8u)), f16(extractBits(q, 8u, 8u)), f16(extractBits(q, 16u, 8u)), f16(extractBits(q, 24u, 8u)));
}

@compute @workgroup_size(32, 1, 1)
fn main() {
    let tile = coopLoadTensorBlockT<coop_mat16x16<f16,C>, 1u, 4u>((&input), vec2(16u), vec2<u32>(4u, 1u), vec2(0i), decode, decode4_);
    coopStoreTensorT(tile, (&output), vec2(16u), vec2<u32>(16u, 1u), vec2(0i));
    return;
}
