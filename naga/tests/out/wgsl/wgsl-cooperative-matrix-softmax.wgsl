enable f16;
enable wgpu_cooperative_matrix;

struct Params {
    m: u32,
    n: u32,
    k: u32,
}

@group(0) @binding(0)
var<storage> a: array<f16>;
@group(0) @binding(1)
var<storage> b: array<f16>;
@group(0) @binding(2)
var<storage, read_write> c: array<f32>;
var<immediate> params: Params;

fn combine_max(left: f32, right: f32) -> f32 {
    return max(left, right);
}

fn combine_sum(left_1: f32, right_1: f32) -> f32 {
    return (left_1 + right_1);
}

fn mask(row: u32, col: u32, value: f32, width: u32) -> f32 {
    return select(-1000000000000000000000000000000f, value, (col < width));
}

fn exponent(row_1: u32, col_1: u32, value_1: f32, maximum: f32) -> f32 {
    return exp((value_1 - maximum));
}

fn normalize_(row_2: u32, col_2: u32, value_2: f32, sum_1: f32) -> f32 {
    return (value_2 / sum_1);
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>) {
    var sum: coop_mat64x64<f32,C,workgroup> = coop_mat64x64<f32,C,workgroup>();
    var t: u32 = 0u;

    let row_3 = (group.x * 64u);
    let col_3 = (group.y * 64u);
    loop {
        let _e11 = t;
        let _e14 = params.k;
        if (_e11 < _e14) {
        } else {
            break;
        }
        {
            let _e19 = params.m;
            let _e22 = params.k;
            let _e26 = params.k;
            let _e30 = t;
            let left_2 = coopLoadTensorT<coop_mat32x64<f16,A,workgroup>>((&a), vec2<u32>(_e19, _e22), vec2<u32>(_e26, 1u), vec2<i32>(i32(row_3), i32(_e30)));
            let _e37 = params.k;
            let _e40 = params.n;
            let _e44 = params.n;
            let _e47 = t;
            let right_2 = coopLoadTensorT<coop_mat64x32<f16,B,workgroup>>((&b), vec2<u32>(_e37, _e40), vec2<u32>(_e44, 1u), vec2<i32>(i32(_e47), i32(col_3)));
            let _e52 = sum;
            sum = coopMultiplyAdd(left_2, right_2, _e52);
        }
        continuing {
            let _e54 = t;
            t = (_e54 + 32u);
        }
    }
    let _e57 = sum;
    let _e60 = params.n;
    let masked = coopMap(_e57, mask, min((_e60 - col_3), 64u));
    let maximum_1 = coopReduceRows<coop_mat64x64<f32,C,workgroup>>(masked, combine_max);
    let numerator = coopMap(masked, exponent, maximum_1);
    let denominator = coopReduceRows<coop_mat64x64<f32,C,workgroup>>(numerator, combine_sum);
    let result = coopMap(numerator, normalize_, denominator);
    let _e72 = params.m;
    let _e75 = params.n;
    let _e79 = params.n;
    coopStoreTensorT(result, (&c), vec2<u32>(_e72, _e75), vec2<u32>(_e79, 1u), vec2<i32>(i32(row_3), i32(col_3)));
    return;
}
