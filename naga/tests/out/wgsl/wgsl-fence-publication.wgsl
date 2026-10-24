enable wgpu_memory_fence;

@group(0) @binding(0)
var<storage, read_write> payload: array<u32>;
@group(0) @binding(1)
@coherent var<storage> incoming: array<u32>;
@group(0) @binding(2)
var<storage, read_write> ready: atomic<u32>;
@group(0) @binding(3)
var<storage, read_write> result: array<u32>;
var<workgroup> claimed: u32;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if (group.x == 0u) {
        payload[lane] = (lane + 1u);
        storageFence();
        workgroupBarrier();
        if (lane == 0u) {
            storageFence();
            let _e13 = atomicOr((&ready), 1u);
            return;
        } else {
            return;
        }
    } else {
        if (lane == 0u) {
            let _e19 = atomicAnd((&ready), 4294967294u);
            claimed = (_e19 & 1u);
            storageFence();
        }
        let _e23 = workgroupUniformLoad((&claimed));
        if (_e23 != 0u) {
            storageFence();
            let _e30 = incoming[lane];
            result[lane] = _e30;
            return;
        } else {
            return;
        }
    }
}
