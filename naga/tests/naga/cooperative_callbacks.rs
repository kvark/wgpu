#![cfg(all(feature = "wgsl-in", feature = "wgsl-out", spv_out))]

use naga::{
    back::spv,
    compact::KeepUnused,
    valid::{Capabilities, ValidationFlags, Validator},
};
use rspirv::binary::Disassemble;

const SOFTMAX: &str = include_str!("../in/wgsl/cooperative-matrix-softmax.wgsl");
const MODES: &str = include_str!("../in/wgsl/cooperative-matrix-reduce-modes.wgsl");

fn validate(source: &str, caps: Capabilities) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    Validator::new(ValidationFlags::all(), caps)
        .validate(&module)
        .map(|_| ())
        .map_err(|e| e.emit_to_string(source))
}

#[test]
fn callbacks_survive_compaction_roundtrip_and_spirv() {
    for source in [SOFTMAX, MODES] {
        let mut module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        for _ in 0..2 {
            let info = Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .unwrap();
            let options = spv::Options {
                lang_version: (1, 6),
                ..Default::default()
            };
            let words = spv::write_vec(&module, &info, &options, None).unwrap();
            let assembly = rspirv::dr::load_words(&words).unwrap().disassemble();
            for op in [
                "OpCooperativeMatrixReduceNV",
                "OpCooperativeMatrixPerElementOpNV",
                "OpCapability CooperativeMatrixReductionsNV",
                "OpCapability CooperativeMatrixPerElementOperationsNV",
            ] {
                assert!(assembly.contains(op), "{op}");
            }
            let text = naga::back::wgsl::write_string(
                &module,
                &info,
                naga::back::wgsl::WriterFlags::empty(),
            )
            .unwrap();
            validate(&text, Capabilities::all()).unwrap();
            naga::compact::compact(&mut module, KeepUnused::No);
        }
    }
}

#[test]
fn reduction_shapes_components_and_scope() {
    for source in [
        MODES.replace(
            "coop_mat16x32<f32,C,workgroup>",
            "coop_mat16x16<f32,C,workgroup>",
        ),
        MODES.replace(
            "coop_mat32x16<f32,C,workgroup>",
            "coop_mat16x16<f32,C,workgroup>",
        ),
        MODES.replace(
            "coopReduceQuad<coop_mat16x16",
            "coopReduceQuad<coop_mat16x32",
        ),
        MODES.replace(
            "coop_mat32x32<f32,C,workgroup>",
            "coop_mat31x32<f32,C,workgroup>",
        ),
        MODES.replace(
            "coopReduceAll<coop_mat16x16<f32,C,workgroup>>",
            "coopReduceAll<coop_mat16x16<f16,C,workgroup>>",
        ),
        MODES.replace(
            "coopReduceAll<coop_mat16x16<f32,C,workgroup>>",
            "coopReduceAll<coop_mat16x16<f32,A,workgroup>>",
        ),
        MODES.replace(
            "coopReduceAll<coop_mat16x16<f32,C,workgroup>>",
            "coopReduceAll<coop_mat16x16<f32,C>>",
        ),
    ] {
        let error = validate(&source, Capabilities::all()).unwrap_err();
        assert!(error.contains("cooperative"), "{error}");
    }
}

#[test]
fn callback_signatures_and_purity() {
    for source in [
        SOFTMAX.replace("right: f32", "right: f32, extra: f32"),
        SOFTMAX.replace("row: u32", "row: i32"),
        SOFTMAX
            .replace("maximum: f32", "maximum: f16")
            .replace("value - maximum", "value - f32(maximum)"),
        SOFTMAX.replace(
            "exponent, maximum)",
            "exponent, coopConvert<coop_mat64x64<f32,A,workgroup>>(maximum))",
        ),
        SOFTMAX.replace("mask, min(params.n - col, 64u)", "mask"),
        SOFTMAX.replace(
            "return max(left, right);",
            "workgroupBarrier(); return max(left, right);",
        ),
        SOFTMAX.replace(
            "return max(left, right);",
            "c[0] = left; return max(left, right);",
        ),
        SOFTMAX.replace(
            "return exp(value - maximum);",
            "let v = coop_mat16x16<f32,C>(); return exp(value - maximum);",
        ),
        SOFTMAX
            .replace(
                "return max(left, right);",
                "return combine_sum(left, right);",
            )
            .replace("return left + right;", "c[0] = left; return left + right;"),
    ] {
        let error = validate(&source, Capabilities::all()).unwrap_err();
        assert!(error.contains("Invalid cooperative callback"), "{error}");
    }
}

#[test]
fn scope_uniformity_includes_extra_operands_and_control_flow() {
    let source = "enable wgpu_cooperative_matrix;
        fn add(a: f32, b: f32) -> f32 { return a + b; }
        fn scale(r: u32, c: u32, x: f32, factor: f32) -> f32 { return x * factor; }
        @compute @workgroup_size(64)
        fn main(@builtin(subgroup_id) group: u32) {
            let matrix = coop_mat16x16<f32,C>();
            let mapped = coopMap(matrix, scale, f32(group));
            if group == 0u {
                let reduced = coopReduceAll<coop_mat16x16<f32,C>>(mapped, add);
                _ = reduced;
            }
        }";
    validate(source, Capabilities::all()).unwrap();
    for invalid in [
        source.replace("subgroup_id", "local_invocation_index"),
        source.replace("f32,C>", "f32,C,workgroup>"),
    ] {
        let error = validate(&invalid, Capabilities::all()).unwrap_err();
        assert!(error.contains("not uniform"), "{error}");
    }
}

#[test]
fn capabilities_and_spirv_version() {
    for cap in [
        Capabilities::COOPERATIVE_MATRIX_REDUCTIONS,
        Capabilities::COOPERATIVE_MATRIX_PER_ELEMENT_OPERATIONS,
    ] {
        assert!(validate(SOFTMAX, Capabilities::all() - cap).is_err());
    }
    let module = naga::front::wgsl::parse_str(SOFTMAX).unwrap();
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    assert!(matches!(
        spv::write_vec(&module, &info, &spv::Options::default(), None),
        Err(spv::Error::SpirvVersionTooLow(1, 6))
    ));
    for missing in [
        spirv::Capability::CooperativeMatrixReductionsNV,
        spirv::Capability::CooperativeMatrixPerElementOperationsNV,
    ] {
        let mut options = spv::Options {
            lang_version: (1, 6),
            ..Default::default()
        };
        options.capabilities = Some(
            [
                spirv::Capability::Shader,
                spirv::Capability::Float16,
                spirv::Capability::StorageBuffer16BitAccess,
                spirv::Capability::UniformAndStorageBuffer16BitAccess,
                spirv::Capability::VulkanMemoryModel,
                spirv::Capability::VulkanMemoryModelDeviceScope,
                spirv::Capability::CooperativeMatrixKHR,
                spirv::Capability::CooperativeMatrixTensorAddressingNV,
                spirv::Capability::TensorAddressingNV,
                spirv::Capability::CooperativeMatrixReductionsNV,
                spirv::Capability::CooperativeMatrixPerElementOperationsNV,
            ]
            .into_iter()
            .filter(|&cap| cap != missing)
            .collect(),
        );
        assert!(matches!(
            spv::write_vec(&module, &info, &options, None),
            Err(spv::Error::MissingCapabilities(_, capabilities)) if capabilities.contains(&missing)
        ));
    }
}

#[test]
fn callback_uniformity_propagates_through_functions() {
    let source = "enable wgpu_cooperative_matrix;
        fn scale(r: u32, c: u32, x: f32, factor: f32) -> f32 { return x * factor; }
        fn apply(factor: f32) {
            let matrix = coop_mat16x16<f32,C>();
            _ = coopMap(matrix, scale, factor);
        }
        @compute @workgroup_size(64)
        fn main(@builtin(subgroup_id) group: u32) {
            if group == 0u { apply(f32(group)); }
        }";
    validate(source, Capabilities::all()).unwrap();
    for invalid in [
        source.replace("subgroup_id", "local_invocation_index"),
        source.replace("f32,C>", "f32,C,workgroup>"),
    ] {
        let error = validate(&invalid, Capabilities::all()).unwrap_err();
        assert!(error.to_lowercase().contains("cooperative"), "{error}");
    }
}
