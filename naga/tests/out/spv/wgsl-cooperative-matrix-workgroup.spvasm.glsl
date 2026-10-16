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
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer _13_12
{
    float16_t _m0[];
} _12;

layout(set = 0, binding = 1, std430) readonly buffer _16_15
{
    float16_t _m0[];
} _15;

layout(set = 0, binding = 2, std430) buffer _19_18
{
    float _m0[];
} _18;

layout(set = 0, binding = 3, std140) uniform _22_21
{
    uvec4 _m0;
} _21;

void main()
{
    coopmat<float16_t, gl_ScopeWorkgroup, 64u, 32u, gl_MatrixUseA> _44;
    coopMatLoad(_44, _12._m0, 0u, 32u, gl_CooperativeMatrixLayoutRowMajor);
    coopmat<float16_t, gl_ScopeWorkgroup, 32u, 64u, gl_MatrixUseB> _48;
    coopMatLoad(_48, _15._m0, 0u, 64u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatStore(coopMatMulAdd(_44, _48, coopmat<float, gl_ScopeWorkgroup, 64u, 64u, gl_MatrixUseAccumulator>(0.0), 0), _18._m0, _21._m0.x + _21._m0.y, _21._m0.z + _21._m0.w, gl_CooperativeMatrixLayoutRowMajor);
}

