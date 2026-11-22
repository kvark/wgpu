#![cfg(all(feature = "wgsl-in", feature = "wgsl-out", spv_out))]

use naga::{
    back::spv,
    valid::{Capabilities, ValidationFlags, Validator},
};
use rspirv::binary::Disassemble;

const CONVERT: &str = r#"
enable f16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read_write> ab: array<f16>;
@group(0) @binding(1) var<storage, read_write> accum: array<f32>;
@compute @workgroup_size(32)
fn main() {
    let a = coopLoad<coop_mat16x16<f16, A>>(&ab[0]);
    let b = coopLoad<coop_mat16x16<f16, B>>(&ab[256]);
    let partial = coopMultiplyAdd(a, b, coop_mat16x16<f16, C>());
    let wide = coop_mat16x16<f32, C>(partial);
    coopStore(wide, &accum[0]);
}
"#;

fn parse(source: &str) -> naga::Module {
    naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)))
}

fn validate(module: &naga::Module) -> Result<naga::valid::ModuleInfo, String> {
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(module)
        .map_err(|e| format!("{:?}", e.into_inner()))
}

#[test]
fn accumulator_converts_between_float_widths() {
    let module = parse(CONVERT);
    let info = validate(&module).unwrap();

    let options = spv::Options {
        lang_version: (1, 4),
        ..Default::default()
    };
    let words = spv::write_vec(&module, &info, &options, None).unwrap();
    let assembly = rspirv::dr::load_words(&words).unwrap().disassemble();
    assert_eq!(assembly.matches("OpFConvert").count(), 1, "{assembly}");

    let wgsl =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .unwrap();
    assert!(wgsl.contains("coop_mat16x16<f32,C>(partial)"), "{wgsl}");
    validate(&parse(&wgsl)).unwrap();
}

#[test]
fn conversion_keeps_shape_role_and_scope() {
    for target in [
        "coop_mat16x16<f32, A>",
        "coop_mat16x8<f32, C>",
        "coop_mat16x16<f32, C, workgroup>",
    ] {
        let source = CONVERT.replace(
            "coop_mat16x16<f32, C>(partial)",
            &format!("{target}(partial)"),
        );
        if let Ok(module) = naga::front::wgsl::parse_str(&source) {
            assert!(validate(&module).is_err(), "accepted {target}");
        }
    }
}

#[test]
fn only_float_conversions_validate() {
    let module = parse(CONVERT);
    for (kind, convert) in [
        (naga::ScalarKind::Float, None),
        (naga::ScalarKind::Sint, Some(4)),
        (naga::ScalarKind::Uint, Some(4)),
    ] {
        let mut module = module.clone();
        let function = &mut module.entry_points[0].function;
        let mut casts = 0;
        for (_, expression) in function.expressions.iter_mut() {
            if let naga::Expression::As {
                kind: ref mut cast_kind,
                convert: ref mut cast_convert,
                ..
            } = *expression
            {
                *cast_kind = kind;
                *cast_convert = convert;
                casts += 1;
            }
        }
        assert_eq!(casts, 1);
        let error = validate(&module).expect_err("non-float cooperative cast validated");
        assert!(error.contains("InvalidCastArgument"), "{error}");
    }
}
