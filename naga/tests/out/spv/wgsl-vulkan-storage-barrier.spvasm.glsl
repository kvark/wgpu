#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _8_7
{
    uint _m0[];
} _7;

layout(set = 0, binding = 1, std430) readonly buffer _11_10
{
    uint _m0[];
} _10;

shared uint _13[64];

uint _18(uint _16, uint _17)
{
    _7._m0[_16] = _17;
    groupMemoryBarrier();
    barrier();
    return _7._m0[_16 ^ 1u];
}

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _13 = uint[](0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    barrier();
    _13[gl_LocalInvocationIndex] = _10._m0[gl_LocalInvocationIndex];
    barrier();
    uint _56 = _13[gl_LocalInvocationIndex ^ 1u];
    uint _57 = _18(gl_LocalInvocationIndex, _56);
    groupMemoryBarrier();
    barrier();
    _7._m0[gl_LocalInvocationIndex] = _57;
    memoryBarrierBuffer();
}

