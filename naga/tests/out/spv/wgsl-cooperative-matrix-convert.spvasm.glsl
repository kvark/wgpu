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
#extension GL_NV_cooperative_matrix2 : require
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer _17_16
{
    float _m0[];
} _16;

layout(set = 0, binding = 1, std430) buffer _20_19
{
    float16_t _m0[];
} _19;

void main()
{
    coopmat<float, gl_ScopeWorkgroup, 64u, 32u, gl_MatrixUseAccumulator> _33;
    coopMatLoad(_33, _16._m0, 0u, 32u, gl_CooperativeMatrixLayoutRowMajor);
    // unimplemented op 5390
    coopMatStore(coopmat<float16_t, gl_ScopeWorkgroup, 64u, 32u, gl_MatrixUseA>(coopmat<float16_t, gl_ScopeWorkgroup, 64u, 32u, gl_MatrixUseAccumulator>(_33)), _19._m0, 0u, 32u, gl_CooperativeMatrixLayoutRowMajor);
    coopMatStore(_36, _19._m0, 2048u, 32u, gl_CooperativeMatrixLayoutColumnMajor);
}

