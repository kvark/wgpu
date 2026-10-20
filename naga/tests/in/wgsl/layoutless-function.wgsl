struct Item {
    values: array<vec2u, 4>,
}
struct Block {
    items: array<Item, 2>,
}
@group(0) @binding(0) var<storage, read_write> data: array<Block>;

fn read_item(block: Block, i: u32, j: u32) -> vec2u {
    let item = block.items[i];
    return item.values[j];
}

fn change_item(block: Block, i: u32, j: u32) -> Block {
    var local = Block(array<Item, 2>(
        Item(array<vec2u, 4>(vec2u(1), vec2u(2), vec2u(3), vec2u(4))),
        Item()));
    local = block;
    local.items[i].values[j] += vec2u(1);
    return local;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let block = data[0];
    data[1] = change_item(block, id.x, id.y);
    data[0].items[0].values[0] = read_item(block, id.x, id.y);
}
