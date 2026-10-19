enable f16;

@group(0) @binding(0) var<storage, read_write> value: f32;

@compute @workgroup_size(1)
fn before() {
    value += 1.0;
}

@compute @workgroup_size(1)
fn half() {
    value = f32(f16(value));
}

@compute @workgroup_size(1)
fn after() {
    value += 2.0;
}
