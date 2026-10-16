#![cfg(feature = "wgsl-in")]
use naga::valid::{
    Capabilities, EntryPointError, ExpressionError, FunctionError, ValidationError,
    ValidationFlags, Validator,
};
const SOURCE: &str = include_str!("../in/wgsl/cooperative-matrix-store-arguments.wgsl");

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
