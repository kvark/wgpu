#version 460
#extension GL_EXT_bfloat16 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer _17_16
{
    bfloat16_t _m0[];
} _16;

layout(set = 0, binding = 1, std430) readonly buffer _20_19
{
    bfloat16_t _m0[];
} _19;

layout(set = 0, binding = 2, std430) buffer _23_22
{
    float _m0[];
} _22;

coopmat<float, gl_ScopeWorkgroup, 128u, 256u, gl_MatrixUseAccumulator> _25 = coopmat<float, gl_ScopeWorkgroup, 128u, 256u, gl_MatrixUseAccumulator>(0.0);

coopmat<bfloat16_t, gl_ScopeWorkgroup, 128u, 64u, gl_MatrixUseA> _30(uint _29)
{
    coopmat<bfloat16_t, gl_ScopeWorkgroup, 128u, 64u, gl_MatrixUseA> _37;
    coopMatLoad(_37, _16._m0, _29, 64u, gl_CooperativeMatrixLayoutRowMajor);
    return _37;
}

void main()
{
    coopmat<bfloat16_t, gl_ScopeWorkgroup, 64u, 256u, gl_MatrixUseB> _56;
    coopMatLoad(_56, _19._m0, 0u, 256u, gl_CooperativeMatrixLayoutRowMajor);
    _25 = coopMatMulAdd(_30(gl_WorkGroupID.x * 8192u), _56, coopmat<float, gl_ScopeWorkgroup, 128u, 256u, gl_MatrixUseAccumulator>(0.0), 0);
    coopMatStore(_25, _22._m0, 0u, 256u, gl_CooperativeMatrixLayoutRowMajor);
}

