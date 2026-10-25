#![cfg(feature = "wgsl-in")]
use naga::valid::{
    Capabilities, EntryPointError, ExpressionError, FunctionError, ValidationError,
    ValidationFlags, Validator,
};
const SOURCE: &str = include_str!("../in/wgsl/cooperative-matrix-store-arguments.wgsl");

#[cfg(feature = "wgsl-out")]
#[test]
fn cooperative_matrix_roundtrip_preserves_required_enables() {
    for source in [
        r#"
enable wgpu_cooperative_matrix;
var<workgroup> tile: array<vec4<f32>, 64>;
@compute @workgroup_size(32)
fn main() {
    let matrix = coopLoadT<coop_mat8x8<f32, C>>(&tile[0], 8u);
    coopStoreT(matrix, &tile[0], 8u);
}
"#,
        r#"
enable f16;
enable wgpu_cooperative_matrix;
fn matrix() -> coop_mat8x8<f16, A> { return coop_mat8x8<f16, A>(); }
@compute @workgroup_size(32)
fn main() { let value = matrix(); }
"#,
    ] {
        let mut module = naga::front::wgsl::parse_str(source).unwrap();
        naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator.validate(&module).unwrap();
        let output = naga::back::wgsl::write_string(
            &module,
            &info,
            naga::back::wgsl::WriterFlags::empty(),
        )
        .unwrap();
        assert!(output.contains("enable wgpu_cooperative_matrix;"), "{output}");
        let roundtrip = naga::front::wgsl::parse_str(&output)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&output)));
        validator.validate(&roundtrip).unwrap();
    }
}

#[test]
fn computed_store_operands_must_be_emitted() {
    for pointer in [false, true] {
        let mut module = naga::front::wgsl::parse_str(SOURCE).unwrap();
        let function = &mut module.entry_points[0].function;
        let operand = function
            .body
            .iter()
            .find_map(|statement| match statement {
                naga::Statement::CooperativeStore { data, .. } => {
                    Some(if pointer { data.pointer } else { data.stride })
                }
                _ => None,
            })
            .unwrap();
        let mut body = naga::Block::new();
        for (statement, &span) in function.body.span_iter() {
            if let naga::Statement::Emit(range) = statement {
                for expression in range.clone().filter(|&h| h != operand) {
                    body.push(
                        naga::Statement::Emit(naga::Range::new_from_bounds(expression, expression)),
                        span,
                    );
                }
            } else {
                body.push(statement.clone(), span);
            }
        }
        function.body = body;
        let error = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_err();
        assert!(matches!(error.as_inner(), ValidationError::EntryPoint {
            source: EntryPointError::Function(FunctionError::Expression { handle, source: ExpressionError::NotInScope }), ..
        } if *handle == operand));
    }
}
