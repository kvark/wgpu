// language: metal2.3
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
    uint size1;
};

typedef metal::half4 type_1[1];
typedef metal::float4 type_3[1];
struct type_6 {
    metal::half4 inner[64];
};
struct type_8 {
    metal::float2 inner[64];
};
struct type_10 {
    metal::float3 inner[64];
};
metal::simdgroup_half8x8 NagaCooperativeLoad(const threadgroup half* ptr, int stride, bool is_row_major) {
    metal::simdgroup_half8x8 m;
    simdgroup_load(m, ptr, stride, 0, is_row_major);
    return m;
}

metal::simdgroup_half8x8 NagaCooperativeLoad(const device half* ptr, int stride, bool is_row_major) {
    metal::simdgroup_half8x8 m;
    simdgroup_load(m, ptr, stride, 0, is_row_major);
    return m;
}

metal::simdgroup_float8x8 NagaCooperativeLoad(const device float* ptr, int stride, bool is_row_major) {
    metal::simdgroup_float8x8 m;
    simdgroup_load(m, ptr, stride, 0, is_row_major);
    return m;
}

metal::simdgroup_float8x8 NagaCooperativeMultiplyAdd(const thread metal::simdgroup_half8x8& a, const thread metal::simdgroup_half8x8& b, const thread metal::simdgroup_float8x8& c) {
    metal::simdgroup_float8x8 d;
    simdgroup_multiply_accumulate(d,a,b,c);
    return d;
}

metal::simdgroup_float8x8 NagaCooperativeLoad(const threadgroup float* ptr, int stride, bool is_row_major) {
    metal::simdgroup_float8x8 m;
    simdgroup_load(m, ptr, stride, 0, is_row_major);
    return m;
}


struct main_Input {
};
[[max_total_threads_per_threadgroup(32)]] kernel void main_(
  uint lane [[thread_index_in_threadgroup]]
, device type_1 const& src [[user(fake0)]]
, device type_3& dst [[user(fake0)]]
, constant metal::uint4& params [[user(fake0)]]
, threadgroup type_6& tile
, threadgroup type_8& pairs
, threadgroup type_10& triples
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    if (lane == 0u) {
        tile = {};
        pairs = {};
        triples = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::half4 _e5 = src[lane];
    tile.inner[lane] = _e5;
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint _e10 = params.x;
    metal::simdgroup_half8x8 a = NagaCooperativeLoad(reinterpret_cast<threadgroup half*>(&tile.inner[2]), (_e10 + 1u) * 4, false);
    uint _e17 = params.y;
    uint _e21 = params.x;
    metal::simdgroup_half8x8 b = NagaCooperativeLoad(reinterpret_cast<const device half*>(&src[_e17]), _e21 * 4, true);
    metal::simdgroup_float8x8 c = NagaCooperativeLoad(reinterpret_cast<device float*>(&dst[4]), 3u * 4, false);
    metal::simdgroup_float8x8 d = NagaCooperativeMultiplyAdd(a, b, c);
    uint _e31 = params.z;
    uint _e35 = params.x;
    simdgroup_store(d, reinterpret_cast<device float*>(&dst[_e31]), (_e35 + 1u) * 4);
    simdgroup_store(d, reinterpret_cast<threadgroup float*>(&pairs.inner[2]), 6u * 2, 0, true);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::simdgroup_float8x8 e = NagaCooperativeLoad(reinterpret_cast<threadgroup float*>(&pairs.inner[2]), 6u * 2, true);
    simdgroup_store(e, reinterpret_cast<threadgroup float*>(&triples.inner[1]), 4u * 3);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::simdgroup_float8x8 f = NagaCooperativeLoad(reinterpret_cast<threadgroup float*>(&triples.inner[1]), 4u * 3, false);
    simdgroup_store(f, reinterpret_cast<device float*>(&dst[4]), 3u * 4, 0, true);
    return;
}
