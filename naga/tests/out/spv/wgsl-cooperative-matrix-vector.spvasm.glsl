#version 460
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer _18_17
{
    f16vec4 _m0[];
} _17;

layout(set = 0, binding = 1, std430) buffer _21_20
{
    vec4 _m0[];
} _20;

layout(set = 0, binding = 2, std140) uniform _24_23
{
    uvec4 _m0;
} _23;

shared f16vec4 _26[64];
shared vec2 _28[64];
shared vec3 _30[64];

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _26 = f16vec4[](f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)), f16vec4(float16_t(0.0)));
        _28 = vec2[](vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0));
        _30 = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    }
    barrier();
    _26[gl_LocalInvocationIndex] = _17._m0[gl_LocalInvocationIndex];
    barrier();
    coopmat<float16_t, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseA> _74;
    coopMatLoad(_74, _26, 2u, _23._m0.x + 1u, gl_CooperativeMatrixLayoutRowMajor);
    coopmat<float16_t, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseB> _81;
    coopMatLoad(_81, _17._m0, _23._m0.y, _23._m0.x, gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator> _85;
    coopMatLoad(_85, _20._m0, 4u, 3u, gl_CooperativeMatrixLayoutRowMajor);
    coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator> _86 = coopMatMulAdd(_74, _81, _85, 0);
    coopMatStore(_86, _20._m0, _23._m0.z, _23._m0.x + 1u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatStore(_86, _28, 2u, 6u, gl_CooperativeMatrixLayoutColumnMajor);
    barrier();
    coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator> _96;
    coopMatLoad(_96, _28, 2u, 6u, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatStore(_96, _30, 1u, 4u, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator> _100;
    coopMatLoad(_100, _30, 1u, 4u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatStore(_100, _20._m0, 4u, 3u, gl_CooperativeMatrixLayoutColumnMajor);
}

