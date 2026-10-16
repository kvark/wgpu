#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _12_11
{
    float _m0[];
} _11;

layout(set = 0, binding = 1, std140) uniform _15_14
{
    uvec4 _m0;
} _14;

void main()
{
    coopMatStore(coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator>(0.0), _11._m0, _14._m0.x + _14._m0.y, _14._m0.z + _14._m0.w, gl_CooperativeMatrixLayoutRowMajor);
}

