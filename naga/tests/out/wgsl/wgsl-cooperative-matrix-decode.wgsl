enable f16;
enable wgpu_cooperative_matrix;

struct BlockQ8_ {
    scale: u32,
    quant: array<u32, 8>,
}

struct Params {
    m: u32,
    n: u32,
    k: u32,
}

@group(0) @binding(0)
var<storage> a: array<f16>;
@group(0) @binding(1)
var<storage> b: array<BlockQ8_>;
@group(0) @binding(2)
var<storage, read_write> c: array<f32>;
var<immediate> params: Params;

fn scale_value(value: i32, scale: u32) -> f16 {
    return f16((f32(value) * unpack2x16float(scale).x));
}

fn decode_q8_(block: BlockQ8_, block_coord: vec2<u32>, element: vec2<u32>) -> f16 {
    let packed = bitcast<i32>(block.quant[(element.y / 4u)]);
    let _e17 = scale_value(extractBits(packed, ((element.y % 4u) * 8u), 8u), block.scale);
    return _e17;
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>) {
    var sum: coop_mat64x64<f32,C,workgroup> = coop_mat64x64<f32,C,workgroup>();
    var t: u32 = 0u;

    let row = i32((group.x * 64u));
    let col = i32((group.y * 64u));
    loop {
        let _e13 = t;
        let _e16 = params.k;
        if (_e13 < _e16) {
        } else {
            break;
        }
        {
            let _e21 = params.m;
            let _e24 = params.k;
            let _e28 = params.k;
            let _e31 = t;
            let left = coopLoadTensorT<coop_mat32x64<f16,A,workgroup>>((&a), vec2<u32>(_e21, _e24), vec2<u32>(_e28, 1u), vec2<i32>(row, i32(_e31)));
            let _e38 = params.n;
            let _e41 = params.k;
            let _e45 = params.k;
            let _e52 = t;
            let right = coopLoadTensorBlock<coop_mat64x32<f16,B,workgroup>, 1u, 32u>((&b), vec2<u32>(_e38, _e41), vec2<u32>(((_e45 + 31u) / 32u), 1u), vec2<i32>(col, i32(_e52)), decode_q8_);
            let _e56 = sum;
            sum = coopMultiplyAdd(left, right, _e56);
        }
        continuing {
            let _e58 = t;
            t = (_e58 + 32u);
        }
    }
    let _e61 = sum;
    let _e65 = params.m;
    let _e68 = params.n;
    let _e72 = params.n;
    coopStoreTensorT(_e61, (&c), vec2<u32>(_e65, _e68), vec2<u32>(_e72, 1u), vec2<i32>(row, col));
    return;
}
