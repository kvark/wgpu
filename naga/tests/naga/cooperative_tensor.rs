#![cfg(all(feature = "wgsl-in", feature = "wgsl-out", spv_out))]

use naga::{
    back::spv,
    compact::KeepUnused,
    valid::{Capabilities, ValidationFlags, Validator},
};
use rspirv::binary::Disassemble;

const TENSOR: &str = include_str!("../in/wgsl/cooperative-matrix-tensor.wgsl");
const CONVERT: &str = include_str!("../in/wgsl/cooperative-matrix-convert.wgsl");

fn validate(source: &str, capabilities: Capabilities) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    Validator::new(ValidationFlags::all(), capabilities)
        .validate(&module)
        .map(|_| ())
        .map_err(|e| e.emit_to_string(source))
}

#[test]
fn roundtrip_compaction_and_spirv() {
    for source in [TENSOR, CONVERT] {
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
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
            assert!(assembly.contains("OpCapability CooperativeMatrixConversionsNV"));
            assert!(assembly.contains("OpFConvert"));
            if source == TENSOR {
                for instruction in [
                    "OpCapability TensorAddressingNV",
                    "OpCapability CooperativeMatrixTensorAddressingNV",
                    "OpTypeTensorLayoutNV",
                    "OpTensorLayoutSliceNV",
                    "OpCreateTensorViewNV",
                    "OpCooperativeMatrixLoadTensorNV",
                    "OpCooperativeMatrixStoreTensorNV",
                ] {
                    assert!(assembly.contains(instruction), "{instruction}");
                }
            } else {
                assert!(assembly.contains("OpCooperativeMatrixConvertNV"));
                assert!(assembly.contains("OpCooperativeMatrixTransposeNV"));
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
fn capabilities_and_version_are_required() {
    for (source, capability) in [
        (TENSOR, Capabilities::COOPERATIVE_MATRIX_TENSOR_ADDRESSING),
        (CONVERT, Capabilities::COOPERATIVE_MATRIX_CONVERSIONS),
    ] {
        assert!(validate(source, Capabilities::all() - capability).is_err());
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap();
        assert!(matches!(
            spv::write_vec(&module, &info, &spv::Options::default(), None),
            Err(spv::Error::SpirvVersionTooLow(1, 6))
        ));
        let options = spv::Options {
            lang_version: (1, 6),
            capabilities: Some(
                [
                    spirv::Capability::Shader,
                    spirv::Capability::VulkanMemoryModel,
                    spirv::Capability::CooperativeMatrixKHR,
                    spirv::Capability::Float16,
                    spirv::Capability::StorageBuffer16BitAccess,
                ]
                .into_iter()
                .collect(),
            ),
            ..Default::default()
        };
        assert!(matches!(
            spv::write_vec(&module, &info, &options, None),
            Err(spv::Error::MissingCapabilities(..))
        ));
    }
}

#[test]
fn tensor_operand_types_and_access_are_checked() {
    for source in [
        TENSOR.replace("&a,", "&a[0],"),
        TENSOR.replace("a: array<f32>", "a: array<f16>"),
        TENSOR.replace("vec2u(m, k)", "vec2i(m, k)"),
        TENSOR.replace("vec2u(k, 1u)", "vec3u(k, 1u, 1u)"),
        TENSOR.replace("vec2i(i32(group.y * 64u), -1)", "vec2u(0u)"),
        TENSOR.replace("var<storage, read_write> c", "var<storage> c"),
        TENSOR.replace("coopStoreTensorT(product,", "coopStoreTensorT(0u,"),
        TENSOR
            .replace("var<storage> a", "var<workgroup> a")
            .replace("@group(0) @binding(0) ", "")
            .replace("a: array<f32>", "a: array<f32, 4096>"),
    ] {
        assert!(validate(&source, Capabilities::all()).is_err(), "{source}");
    }
}

#[test]
fn illegal_conversions_are_rejected() {
    for source in [
        CONVERT.replace(
            "coop_mat32x64<f16,A,workgroup>>(same_type)",
            "coop_mat64x32<f16,A,workgroup>>(same_type)",
        ),
        CONVERT.replace(
            "coop_mat64x32<f16,B,workgroup>>(narrowed)",
            "coop_mat32x64<f16,B,workgroup>>(narrowed)",
        ),
        CONVERT.replace(
            "coop_mat64x32<f16,B,workgroup>>(narrowed)",
            "coop_mat64x32<f16,A,workgroup>>(narrowed)",
        ),
        CONVERT.replace(">>(same_type)", ">>(0.0f)"),
        CONVERT.replace(">>(same_type)", ">>(operand)"),
    ] {
        assert!(validate(&source, Capabilities::all()).is_err(), "{source}");
    }
    let source = "enable wgpu_cooperative_matrix;
        @compute @workgroup_size(32) fn main() {
            let a = coop_mat16x16<f32,A>();
            let b = coopConvert<coop_mat16x16<f32,B>>(a);
        }";
    assert!(validate(source, Capabilities::all()).is_err());
}

#[test]
fn tensor_operands_must_be_uniform_in_matrix_scope() {
    for operand in [
        "vec2u(m, k)",
        "vec2u(k, 1u)",
        "vec2i(i32(group.y * 64u), -1)",
    ] {
        let source = TENSOR
            .replace(
                "group: vec3<u32>)",
                "group: vec3<u32>, @builtin(local_invocation_index) lane: u32)",
            )
            .replace(
                operand,
                if operand.starts_with("vec2i") {
                    "vec2i(i32(lane))"
                } else {
                    "vec2u(lane)"
                },
            );
        let error = validate(&source, Capabilities::all()).unwrap_err();
        assert!(error.contains("not uniform"), "{error}");
    }
    let subgroup = "enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage> input: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(subgroup_id) group: u32) {
            let tile = coopLoadTensorT<coop_mat16x16<f32,C>>(
                &input, vec2u(64u, 16u), vec2u(16u, 1u), vec2i(i32(group * 16u), 0));
            let operand = coopConvert<coop_mat16x16<f32,A>>(tile);
        }";
    validate(subgroup, Capabilities::all()).unwrap();
    let error = validate(
        &subgroup.replace("subgroup_id", "local_invocation_index"),
        Capabilities::all(),
    )
    .unwrap_err();
    assert!(error.contains("not uniform"), "{error}");
}

#[test]
fn tensor_store_accepts_a_structurally_resolved_matrix() {
    let source = "enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(32) fn main() {
            let tile = coopLoadT<coop_mat16x16<f32,C>>(&data[0], 16u);
            coopStoreTensorT(tile, &data, vec2u(16u), vec2u(16u, 1u), vec2i(0));
        }";
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    let options = spv::Options {
        lang_version: (1, 6),
        ..Default::default()
    };
    spv::write_vec(&module, &info, &options, None).unwrap();
}
