#![cfg(all(feature = "wgsl-in", feature = "wgsl-out", spv_out))]

use naga::{
    back::spv,
    compact::KeepUnused,
    valid::{Capabilities, ValidationFlags, Validator},
};
use rspirv::binary::Disassemble;

const SCALAR: &str = include_str!("../in/wgsl/cooperative-matrix-decode.wgsl");
const VECTOR: &str = include_str!("../in/wgsl/cooperative-matrix-decode-vector.wgsl");

fn compile(source: &str, policy: naga::proc::BoundsCheckPolicy) -> rspirv::dr::Module {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    let options = spv::Options {
        lang_version: (1, 6),
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: policy,
            ..Default::default()
        },
        ..Default::default()
    };
    let words = spv::write_vec(&module, &info, &options, None).unwrap();
    rspirv::dr::load_words(&words).unwrap()
}

fn decode_body(module: &rspirv::dr::Module) -> &rspirv::dr::Function {
    use rspirv::dr::Operand;
    let load = module
        .all_inst_iter()
        .find(|inst| {
            inst.class.opcode == spirv::Op::CooperativeMatrixLoadTensorNV
                && inst.operands.iter().any(|operand| {
                    matches!(operand, Operand::TensorAddressingOperands(mask)
                        if mask.contains(spirv::TensorAddressingOperands::DECODE_FUNC))
                })
        })
        .unwrap();
    let Operand::IdRef(wrapper_id) = *load.operands.last().unwrap() else {
        panic!("missing decoder");
    };
    let wrapper = module
        .functions
        .iter()
        .find(|function| function.def.as_ref().unwrap().result_id == Some(wrapper_id))
        .unwrap();
    let call = wrapper
        .all_inst_iter()
        .find(|inst| inst.class.opcode == spirv::Op::FunctionCall)
        .unwrap();
    let Operand::IdRef(body_id) = call.operands[0] else {
        panic!("missing decoder body");
    };
    assert!(!wrapper
        .all_inst_iter()
        .any(|inst| inst.class.opcode == spirv::Op::Load));
    module
        .functions
        .iter()
        .find(|function| function.def.as_ref().unwrap().result_id == Some(body_id))
        .unwrap()
}

#[test]
fn decoder_loads_fields_without_materializing_the_block() {
    let module = compile(SCALAR, naga::proc::BoundsCheckPolicy::Unchecked);
    let body = decode_body(&module);
    let loads: Vec<_> = body
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == spirv::Op::Load)
        .collect();
    assert_eq!(loads.len(), 2);
    for load in loads {
        let ty = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == load.result_type)
            .unwrap();
        assert_eq!(ty.class.opcode, spirv::Op::TypeInt);
        assert!(load.disassemble().contains("Aligned 4"));
    }
    assert!(!body
        .all_inst_iter()
        .any(|inst| inst.class.opcode == spirv::Op::Variable));
}

#[test]
fn decoder_preserves_index_checks() {
    use naga::proc::BoundsCheckPolicy;
    for policy in [
        BoundsCheckPolicy::Restrict,
        BoundsCheckPolicy::ReadZeroSkipWrite,
    ] {
        let module = compile(
            &SCALAR.replace("element.y / 4u", "element.y / 4u + 8u"),
            policy,
        );
        let function = decode_body(&module);
        let body = function.disassemble();
        match policy {
            BoundsCheckPolicy::Restrict => {
                let maximum = module
                    .types_global_values
                    .iter()
                    .find(|inst| {
                        inst.class.opcode == spirv::Op::Constant
                            && inst.operands == [rspirv::dr::Operand::LiteralBit32(7)]
                    })
                    .unwrap()
                    .result_id
                    .unwrap();
                assert!(
                    function.all_inst_iter().any(|inst| {
                        inst.class.opcode == spirv::Op::ExtInst
                            && inst.operands[1]
                                == rspirv::dr::Operand::LiteralExtInstInteger(
                                    spirv::GlslStd450Op::UMin as u32,
                                )
                            && inst.operands.last() == Some(&rspirv::dr::Operand::IdRef(maximum))
                    }),
                    "{body}"
                );
            }
            BoundsCheckPolicy::ReadZeroSkipWrite => {
                assert!(body.contains("OpULessThan"), "{body}");
                assert!(body.contains("OpBranchConditional"), "{body}");
                assert!(body.contains("OpPhi"), "{body}");
            }
            BoundsCheckPolicy::Unchecked => unreachable!(),
        }
    }
}

#[test]
fn decoder_can_still_pass_a_whole_block_by_value() {
    let source = SCALAR.replace(
        "fn decode_q8(",
        "fn decode_q8(block: BlockQ8, block_coord: vec2u, element: vec2u) -> f16 {\n\
            return helper(block, block_coord, element);\n}\nfn helper(",
    );
    let module = compile(&source, naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite);
    let body = decode_body(&module);
    let load = body
        .all_inst_iter()
        .find(|inst| inst.class.opcode == spirv::Op::Load)
        .unwrap();
    assert!(load.disassemble().contains("Aligned 4"));
    assert!(body
        .all_inst_iter()
        .any(|inst| inst.class.opcode == spirv::Op::FunctionCall));
}

#[test]
fn decoder_defers_nested_accesses_and_keeps_local_copies() {
    for (source, expected_loads) in [
        (
            SCALAR
                .replace("quant: array<u32, 8>", "quant: array<array<u32, 4>, 2>")
                .replace(
                    "block.quant[element.y / 4u]",
                    "block.quant[element.y / 16u][(element.y / 4u) % 4u]",
                ),
            2,
        ),
        (
            SCALAR.replace(
                "let packed = bitcast<i32>(block.quant[element.y / 4u]);",
                "var copy = block;\ncopy.quant[0] = 0u;\n\
                 let packed = bitcast<i32>(copy.quant[element.y / 4u]);",
            ),
            3,
        ),
    ] {
        let module = compile(&source, naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite);
        let body = decode_body(&module);
        assert_eq!(
            body.all_inst_iter()
                .filter(|inst| inst.class.opcode == spirv::Op::Load)
                .count(),
            expected_loads
        );
    }
}

#[test]
fn ordinary_calls_do_not_use_physical_decoder_parameters() {
    let source = format!(
        "{SCALAR}\n@compute @workgroup_size(1) fn ordinary() {{\n\
         c[0] = f32(decode_q8(b[0], vec2u(0u), vec2u(0u)));\n}}"
    );
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    let options = spv::Options {
        lang_version: (1, 6),
        ..Default::default()
    };
    let pipeline = spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "ordinary".into(),
    };
    let words = spv::write_vec(&module, &info, &options, Some(&pipeline)).unwrap();
    let assembly = rspirv::dr::load_words(&words).unwrap().disassemble();
    assert!(!assembly.contains("PhysicalStorageBuffer"), "{assembly}");
    assert!(assembly.contains("OpFunctionCall"));
}

fn error(source: &str, capabilities: Capabilities) -> String {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    Validator::new(ValidationFlags::all(), capabilities)
        .validate(&module)
        .unwrap_err()
        .emit_to_string(source)
}

#[test]
fn callbacks_survive_compaction_and_roundtrip() {
    for source in [SCALAR, VECTOR] {
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        for _ in 0..2 {
            let info = Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .unwrap();
            let text = naga::back::wgsl::write_string(
                &module,
                &info,
                naga::back::wgsl::WriterFlags::empty(),
            )
            .unwrap();
            let roundtrip = naga::front::wgsl::parse_str(&text).unwrap();
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&roundtrip)
                .unwrap();
            assert_eq!(module.functions.len(), roundtrip.functions.len());
            let options = spv::Options {
                lang_version: (1, 6),
                flags: spv::WriterFlags::COOPERATIVE_MATRIX_DECODE_VECTOR,
                ..Default::default()
            };
            let words = spv::write_vec(&module, &info, &options, None).unwrap();
            if source == SCALAR {
                let assembly = rspirv::dr::load_words(&words).unwrap().disassemble();
                for expected in [
                    "CooperativeMatrixBlockLoadsNV",
                    "PhysicalStorageBuffer64",
                    "OpTensorLayoutSetBlockSizeNV",
                    "DECODE_FUNC",
                    "Aligned 4",
                    "Aliased",
                    "OpCopyLogical",
                ] {
                    assert!(assembly.contains(expected), "{expected}");
                }
            } else {
                let mut offset = 5;
                let mut capability = false;
                let mut decode_vector = false;
                while offset < words.len() {
                    let count = (words[offset] >> 16) as usize;
                    let op = words[offset] & 0xffff;
                    if op == spirv::Op::Capability as u32 && words[offset + 1] == 5447 {
                        capability = true;
                    }
                    if op == spirv::Op::CooperativeMatrixLoadTensorNV as u32 {
                        assert_eq!(count, 10);
                        assert_eq!(words[offset + 6], 0);
                        assert_eq!(words[offset + 7], 6);
                        assert_ne!(words[offset + 8], words[offset + 9]);
                        decode_vector = true;
                    }
                    offset += count;
                }
                assert!(capability && decode_vector);
            }
            naga::compact::compact(&mut module, KeepUnused::No);
        }
    }
}

#[test]
fn block_dimensions_and_callback_signatures() {
    for source in [
        SCALAR.replace(", 1u, 32u>", ", 0u, 32u>"),
        SCALAR.replace(", 1u, 32u>", ", 1u, 0u>"),
        SCALAR.replace("block_coord: vec2u", "block_coord: vec2i"),
        SCALAR.replace("block_coord: vec2u, ", ""),
        SCALAR
            .replace("fn decode_q8(block: BlockQ8", "fn decode_q8(block: u32")
            .replace("block.quant[element.y / 4u]", "block")
            .replace("block.scale", "block"),
        SCALAR
            .replace(
                "fn decode_q8(block: BlockQ8, block_coord: vec2u, element: vec2u) -> f16",
                "fn decode_q8(block: BlockQ8, block_coord: vec2u, element: vec2u) -> f32",
            )
            .replace("return scale_value(", "return f32(scale_value(")
            .replace("8u), block.scale);", "8u), block.scale));"),
        VECTOR.replace(", 1u, 4u>", ", 1u, 3u>"),
        VECTOR.replace("decode, decode4);", "decode, decode);"),
        VECTOR.replace("decode, decode4);", "decode4, decode);"),
    ] {
        let error = error(&source, Capabilities::all());
        assert!(error.contains("Invalid cooperative callback"), "{error}");
    }
}

#[test]
fn padded_decoder_layouts_are_rejected() {
    for source in [
        SCALAR.replace("quant: array<u32, 8>", "@size(48) quant: array<u32, 8>"),
        SCALAR.replace("scale: u32,\n", "@align(16) scale: u32,\n"),
        SCALAR.replace("scale: u32,\n", "scale: u32,\n unused: array<vec3u, 2>,\n"),
        SCALAR.replace("scale: u32,\n", "scale: u32,\n unused: mat2x3f,\n"),
    ] {
        let error = error(&source, Capabilities::all());
        assert!(error.contains("Invalid cooperative callback"), "{error}");
    }
    let source = SCALAR.replace(
        "quant: array<u32, 8>,",
        "quant: array<u32, 8>, padding: array<u32, 3>,",
    );
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
}

#[test]
fn callbacks_are_pure_including_callees() {
    for body in [
        "c[0] = 1.0;",
        "workgroupBarrier();",
        "let forbidden = coop_mat16x16<f32,C>();",
    ] {
        let source = SCALAR.replace(
            "return f16(f32(value)",
            &format!("{body} return f16(f32(value)"),
        );
        let error = error(&source, Capabilities::all());
        assert!(error.contains("Invalid cooperative callback"), "{error}");
    }
}

#[test]
fn block_and_vector_capabilities_are_separate() {
    assert!(error(
        SCALAR,
        Capabilities::all() - Capabilities::COOPERATIVE_MATRIX_BLOCK_LOADS
    )
    .contains("COOPERATIVE_MATRIX_BLOCK_LOADS"));
    assert!(error(
        VECTOR,
        Capabilities::all() - Capabilities::COOPERATIVE_MATRIX_DECODE_VECTOR
    )
    .contains("COOPERATIVE_MATRIX_DECODE_VECTOR"));
    let module = naga::front::wgsl::parse_str(VECTOR).unwrap();
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    let options = spv::Options {
        lang_version: (1, 6),
        ..Default::default()
    };
    assert!(matches!(
        spv::write_vec(&module, &info, &options, None),
        Err(spv::Error::FeatureNotImplemented(_))
    ));
}
