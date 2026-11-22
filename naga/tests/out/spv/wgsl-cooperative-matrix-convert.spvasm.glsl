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

layout(set = 0, binding = 0, std430) buffer _13_12
{
    float16_t _m0[];
} _12;

layout(set = 0, binding = 1, std430) buffer _16_15
{
    float _m0[];
} _15;

void main()
{
    coopmat<float16_t, gl_ScopeSubgroup, 16u, 16u, gl_MatrixUseA> _32;
    coopMatLoad(_32, _12._m0, 0u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float16_t, gl_ScopeSubgroup, 16u, 16u, gl_MatrixUseB> _36;
    coopMatLoad(_36, _12._m0, 256u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float, gl_ScopeSubgroup, 16u, 16u, gl_MatrixUseAccumulator> _39 = coopmat<float, gl_ScopeSubgroup, 16u, 16u, gl_MatrixUseAccumulator>(coopMatMulAdd(_32, _36, coopmat<float16_t, gl_ScopeSubgroup, 16u, 16u, gl_MatrixUseAccumulator>(float16_t(0.0)), 0));
    coopMatStore(_39, _15._m0, 0u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
    coopMatStore(coopmat<float16_t, gl_ScopeSubgroup, 16u, 16u, gl_MatrixUseAccumulator>(_39), _12._m0, 512u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
}

