# Blade synchronization study benchmark

This headless benchmark is the tracked wgpu counterpart to Blade's
`examples/sync-bench`. It emits the same CSV schema and implements the same four
workload graphs:

- `compute-independent`
- `compute-chain`
- `graphics-independent`
- `graphics-chain`

The WGSL differs only where the APIs require it: wgpu has explicit binding
annotations and uses native immediate data where Blade uses inline uniform
data. Blade's `paper/check-workload-equivalence.py` removes those declarations
and requires the remaining shader source to match byte-for-byte.

Build and list adapters:

```sh
cargo build --release -p wgpu-sync-bench
WGPU_BACKEND=vulkan target/release/wgpu-sync-bench --list-adapters
```

Run one configuration:

```sh
WGPU_BACKEND=vulkan \
WGPU_ADAPTER_NAME="RTX 5070" \
target/release/wgpu-sync-bench \
  --workload compute-independent \
  --policy tracked
```

Use `--no-gpu-timing` for CPU-only collection. The matched matrix, metadata,
correctness, and analysis workflow is owned by the Blade study artifact.
