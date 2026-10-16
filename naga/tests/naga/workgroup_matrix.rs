#![cfg(all(feature = "wgsl-in", feature = "wgsl-out", spv_out))]

use naga::valid::{EntryPointError, ExpressionError, FunctionError, TypeError, ValidationError};
use naga::{
    back::spv,
    valid::{Capabilities, ValidationFlags, Validator},
};
use rspirv::binary::Disassemble;

const SOURCE: &str = include_str!("../in/wgsl/cooperative-matrix-workgroup.wgsl");

#[test]
fn scope_roundtrip_and_spirv() {
    let module = naga::front::wgsl::parse_str(SOURCE).unwrap();
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    let options = spv::Options {
        lang_version: (1, 6),
        ..Default::default()
    };
    let words = spv::write_vec(&module, &info, &options, None).unwrap();
    let assembly = rspirv::dr::load_words(&words).unwrap().disassemble();
    assert!(assembly.contains("OpExtension \"SPV_NV_cooperative_matrix2\""));
    assert!(assembly.contains("OpCooperativeMatrixMulAddKHR"));
    let parsed = rspirv::dr::load_words(&words).unwrap();
    for instruction in &parsed.types_global_values {
        if instruction.class.opcode == rspirv::spirv::Op::TypeCooperativeMatrixKHR {
            let scope = match instruction.operands[1] {
                rspirv::dr::Operand::IdRef(scope) | rspirv::dr::Operand::IdScope(scope) => scope,
                ref other => panic!("{other:?}"),
            };
            let value = parsed
                .types_global_values
                .iter()
                .find(|i| i.result_id == Some(scope))
                .unwrap();
            assert_eq!(value.operands[0], rspirv::dr::Operand::LiteralBit32(2));
        }
    }
    let text =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .unwrap();
    let roundtrip = naga::front::wgsl::parse_str(&text).unwrap();
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&roundtrip)
        .unwrap();
    assert!(matches!(
        spv::write_vec(
            &module,
            &info,
            &spv::Options {
                lang_version: (1, 3),
                ..Default::default()
            },
            None
        ),
        Err(spv::Error::SpirvVersionTooLow(1, 6))
    ));
}

fn error(source: &str) -> ValidationError {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_err()
        .into_inner()
}

#[test]
fn missing_feature_and_subgroup_shapes() {
    assert!(spv::supported_capabilities()
        .contains(Capabilities::COOPERATIVE_MATRIX_WORKGROUP | Capabilities::SHADER_BFLOAT16));
    let module = naga::front::wgsl::parse_str(SOURCE).unwrap();
    let error = Validator::new(
        ValidationFlags::all(),
        Capabilities::all() - Capabilities::COOPERATIVE_MATRIX_WORKGROUP,
    )
    .validate(&module)
    .unwrap_err();
    assert!(matches!(
        error.as_inner(),
        ValidationError::Type {
            source: TypeError::MissingCapability(Capabilities::COOPERATIVE_MATRIX_WORKGROUP),
            ..
        }
    ));
    assert!(matches!(
        self::error(&SOURCE.replace(",workgroup", "")),
        ValidationError::Type {
            source: TypeError::UnsupportedCooperativeMatrixShape { .. },
            ..
        }
    ));
}

#[test]
fn scope_mismatch_identifies_b_or_c() {
    for operand in ["b", "c"] {
        let source = SOURCE
            .replace("32x64", "16x16")
            .replace("64x32", "16x16")
            .replace("64x64", "16x16")
            .replace(
                if operand == "b" {
                    "f16,B,workgroup"
                } else {
                    "f32,C,workgroup"
                },
                if operand == "b" {
                    "f16,B,subgroup"
                } else {
                    "f32,C,subgroup"
                },
            );
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        let (_, product) = module.entry_points[0]
            .function
            .expressions
            .iter()
            .find(|(_, e)| matches!(e, naga::Expression::CooperativeMultiplyAdd { .. }))
            .unwrap();
        let naga::Expression::CooperativeMultiplyAdd { b, c, .. } = *product else {
            unreachable!()
        };
        assert!(matches!(error(&source), ValidationError::EntryPoint {
            source: EntryPointError::Function(FunctionError::Expression {
                source: ExpressionError::InvalidCooperativeOperand(actual), ..
            }), ..
        } if actual == if operand == "b" { b } else { c }));
    }
}

#[test]
fn divergent_addresses_and_strides() {
    for source in [
        SOURCE.replace("&a[0]", "&a[lid * 4096u]"),
        SOURCE.replace("&a[0], 32u", "&a[0], 32u + lid"),
        SOURCE.replace("&c[params.x + params.y]", "&c[lid]"),
        SOURCE.replace("params.z + params.w", "params.z + lid"),
        SOURCE
            .replace(
                "    let left",
                "    var offset = 0u; if lid == 0u { offset = 32u; }\n    let left",
            )
            .replace("&a[0]", "&a[offset]"),
    ] {
        assert!(
            matches!(
                error(&source),
                ValidationError::EntryPoint {
                    source: EntryPointError::Function(FunctionError::NonUniformCooperativeOperand(
                        _
                    )),
                    ..
                }
            ),
            "{source}"
        );
    }
    let source = SOURCE
        .replace("    let left", "    if lid == 0u {\n    let left")
        .replace("params.z + params.w);", "params.z + params.w);\n    }");
    assert!(matches!(
        error(&source),
        ValidationError::EntryPoint {
            source: EntryPointError::Function(FunctionError::NonUniformControlFlow(..)),
            ..
        }
    ));
}

#[test]
fn uniform_locals_and_function_arguments() {
    let template = SOURCE
        .replace(
            "@compute",
            r#"
fn offset(value: u32) -> u32 { return value + 32u; }
fn load(index: u32) -> coop_mat32x64<f16,A,workgroup> {
    return coopLoadT<coop_mat32x64<f16,A,workgroup>>(&a[index], 32u);
}
@compute"#,
        )
        .replace(
            "coopLoadT<coop_mat32x64<f16,A,workgroup>>(&a[0], 32u)",
            "load(offset(INDEX))",
        );
    for index in ["0u", "params.x"] {
        let module = naga::front::wgsl::parse_str(&template.replace("INDEX", index)).unwrap();
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap();
    }
    assert!(matches!(
        error(&template.replace("INDEX", "lid")),
        ValidationError::EntryPoint {
            source: EntryPointError::Function(FunctionError::NonUniformCooperativeOperand(_)),
            ..
        }
    ));
    let source = SOURCE
        .replace(
            "    let left",
            "    for(var offset = 0u; offset < 128u; offset += 32u) {\n    let left",
        )
        .replace("&a[0]", "&a[offset]")
        .replace("params.z + params.w);", "params.z + params.w);\n    }");
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
}

#[test]
fn operand_dependencies_cross_calls_and_loop_iterations() {
    let template = SOURCE
        .replace(
            "@compute",
            r#"
fn offset(a: u32, b: u32) -> u32 { return a + b; }
fn load(index: u32, stride: u32) -> coop_mat32x64<f16,A,workgroup> {
    return coopLoadT<coop_mat32x64<f16,A,workgroup>>(&a[index], stride);
}
@compute"#,
        )
        .replace(
            "coopLoadT<coop_mat32x64<f16,A,workgroup>>(&a[0], 32u)",
            "load(offset(0u, INDEX), STRIDE)",
        );
    for (index, stride, valid) in [
        ("params.x", "params.z", true),
        ("lid", "32u", false),
        ("0u", "32u + lid", false),
    ] {
        let source = template.replace("INDEX", index).replace("STRIDE", stride);
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        let result = Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module);
        if valid {
            result.unwrap();
        } else {
            assert!(matches!(
                result.unwrap_err().as_inner(),
                ValidationError::EntryPoint {
                    source: EntryPointError::Function(FunctionError::NonUniformCooperativeOperand(
                        _
                    )),
                    ..
                }
            ));
        }
    }

    let source = SOURCE
        .replace("    let left", "    var offset = 0u; loop {\n    let left")
        .replace("&a[0]", "&a[offset]")
        .replace(
            "params.z + params.w);",
            "params.z + params.w);\n offset = lid; if params.x == 0u { break; }\n }",
        );
    assert!(matches!(
        error(&source),
        ValidationError::EntryPoint {
            source: EntryPointError::Function(FunctionError::NonUniformCooperativeOperand(_)),
            ..
        }
    ));
}

#[test]
fn entry_point_member_uniformity_and_validation_flags() {
    let template = SOURCE
        .replace(
            "@compute",
            "struct Input { @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) group: vec3<u32> }\n@compute",
        )
        .replace("@builtin(local_invocation_index) lid: u32) {", "input: Input) {")
        .replace("&a[0]", "&a[INDEX]");
    let module =
        naga::front::wgsl::parse_str(&template.replace("INDEX", "input.group.x * 2048u")).unwrap();
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();

    let source = template.replace("INDEX", "input.lid * 2048u");
    assert!(matches!(
        error(&source),
        ValidationError::EntryPoint {
            source: EntryPointError::Function(FunctionError::NonUniformCooperativeOperand(_)),
            ..
        }
    ));
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    Validator::new(
        ValidationFlags::all() - ValidationFlags::CONTROL_FLOW_UNIFORMITY,
        Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn generic_dimensions_and_scope_diagnostic() {
    let source = include_str!("../in/wgsl/cooperative-matrix-workgroup-large.wgsl");
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    let text =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .unwrap();
    assert!(text.contains("coop_mat256x128"));
    let error = naga::front::wgsl::parse_str(&SOURCE.replace(",workgroup", ",device")).unwrap_err();
    assert_eq!(error.message(), "unknown cooperative scope: `device`");
    for ty in [
        "coop_mat0x64",
        "coop_mat4294967296x64",
        "coop_mat4294967295x4294967295",
    ] {
        let source = SOURCE.replace("coop_mat64x64", ty);
        if let Ok(module) = naga::front::wgsl::parse_str(&source) {
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .unwrap_err();
        }
    }
}
