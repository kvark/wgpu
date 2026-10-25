enable wgpu_memory_fence;
enable wgpu_binding_array;

struct Scalars { data: array<f32> }
struct Vectors { data: array<vec4<f32>> }

@group(0) @binding(0) var<storage, read_write> writer: binding_array<Scalars, 2>;
@group(0) @binding(1) var<storage, read> reader: binding_array<Vectors, 2>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(1)
fn publish() {
    writer[1].data[0] = 7.0;
    storageFence();
    result[0] = reader[1].data[0].x;
}

@compute @workgroup_size(1)
fn read_only() {
    result[0] = reader[1].data[0].x;
}
