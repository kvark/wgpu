#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _7_6
{
    uint _m0[];
} _6;

layout(set = 0, binding = 1, std430) readonly buffer _10_9
{
    uint _m0[];
} _9;

layout(set = 0, binding = 2, std430) buffer _13_12
{
    uint _m0;
} _12;

layout(set = 0, binding = 3, std430) buffer _16_15
{
    uint _m0[];
} _15;

shared uint _18;

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _18 = 0u;
    }
    barrier();
    if (gl_WorkGroupID.x == 0u)
    {
        _6._m0[gl_LocalInvocationIndex] = gl_LocalInvocationIndex + 1u;
        memoryBarrierBuffer();
        barrier();
        if (gl_LocalInvocationIndex == 0u)
        {
            memoryBarrierBuffer();
            uint _62 = atomicOr(_12._m0, 1u);
            return;
        }
        else
        {
            return;
        }
    }
    else
    {
        if (gl_LocalInvocationIndex == 0u)
        {
            uint _67 = atomicAnd(_12._m0, 4294967294u);
            _18 = _67 & 1u;
            memoryBarrierBuffer();
        }
        barrier();
        uint _69 = _18;
        barrier();
        if (_69 != 0u)
        {
            memoryBarrierBuffer();
            _15._m0[gl_LocalInvocationIndex] = _9._m0[gl_LocalInvocationIndex];
            return;
        }
        else
        {
            return;
        }
    }
}

