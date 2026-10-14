enable wgpu_memory_fence;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<u32>;
var<workgroup> scratch: array<u32, 64>;

fn exchange(index: u32, value: u32) -> u32 {
    data[index] = value;
    storageBarrier();
    return data[index ^ 1u];
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) index: u32) {
    scratch[index] = input[index];
    workgroupBarrier();
    let first = exchange(index, scratch[index ^ 1u]);
    storageBarrier();
    data[index] = first;
    storageFence();
}
