enable wgpu_memory_fence;

@group(0) @binding(0) var<storage, read_write> payload: array<u32>;
@group(0) @binding(1) @coherent var<storage, read> incoming: array<u32>;
@group(0) @binding(2) var<storage, read_write> ready: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> result: array<u32>;
var<workgroup> claimed: u32;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if group.x == 0u {
        payload[lane] = lane + 1u;
        storageFence();
        workgroupBarrier();
        if lane == 0u {
            storageFence();
            atomicOr(&ready, 1u);
        }
    } else {
        if lane == 0u {
            claimed = atomicAnd(&ready, ~1u) & 1u;
            storageFence();
        }
        if workgroupUniformLoad(&claimed) != 0u {
            storageFence();
            result[lane] = incoming[lane];
        }
    }
}
