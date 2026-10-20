use alloc::vec;

use super::{Capabilities, ExpressionError, ModuleInfo, TypeFlags, Validator};
use crate::{CooperativeTensorDecode, Function, Handle, Module, Scalar, Type, TypeInner};

impl Validator {
    pub(super) fn validate_tensor_decode(
        &self,
        decode: &CooperativeTensorDecode,
        element: Handle<Type>,
        scalar: Scalar,
        module: &Module,
        info: &ModuleInfo,
    ) -> Result<(), ExpressionError> {
        if !self
            .capabilities
            .contains(Capabilities::COOPERATIVE_MATRIX_BLOCK_LOADS)
        {
            return Err(ExpressionError::MissingCapabilities(
                Capabilities::COOPERATIVE_MATRIX_BLOCK_LOADS,
            ));
        }
        let invalid = |reason| ExpressionError::InvalidCooperativeCallback {
            function: decode.function,
            reason,
        };
        if decode.block_size.contains(&0) {
            return Err(invalid("block dimensions must be nonzero"));
        }
        if !info[element].contains(TypeFlags::HOST_SHAREABLE | TypeFlags::SIZED)
            || self.layouter[element].alignment > crate::proc::Alignment::SIXTEEN
        {
            return Err(invalid(
                "block must be a sized host-shareable value with alignment at most 16",
            ));
        }
        // Tensor decode addressing ignores the storage array's ArrayStride.
        if self.layouter[element].size != self.layouter[element].to_stride() {
            return Err(invalid("block stride must not include implicit padding"));
        }
        let mut types = vec![element];
        while let Some(ty) = types.pop() {
            match module.types[ty].inner {
                TypeInner::Scalar(_) | TypeInner::Vector { .. } => {}
                TypeInner::Matrix {
                    columns,
                    rows,
                    scalar,
                } => {
                    if self.layouter[ty].size != columns as u32 * rows as u32 * scalar.width as u32
                    {
                        return Err(invalid("decoder matrices must not have column padding"));
                    }
                }
                TypeInner::Array {
                    base,
                    size: crate::ArraySize::Constant(_),
                    stride,
                } => {
                    if stride != self.layouter[base].size {
                        return Err(invalid("decoder arrays must not have element padding"));
                    }
                    types.push(base);
                }
                TypeInner::Struct { ref members, span } => {
                    if members.last().is_none_or(|member| {
                        member.offset.checked_add(self.layouter[member.ty].size) != Some(span)
                    }) {
                        return Err(invalid(
                            "decoder structs must use explicit members for trailing padding",
                        ));
                    }
                    types.extend(members.iter().map(|m| m.ty));
                }
                _ => {
                    return Err(invalid(
                        "block may only contain numeric values, fixed arrays, and structs",
                    ))
                }
            }
        }
        for (function, vector) in core::iter::once((decode.function, false))
            .chain(decode.vector_function.map(|function| (function, true)))
        {
            let invalid = |reason| ExpressionError::InvalidCooperativeCallback { function, reason };
            let callback = &module.functions[function];
            if callback.arguments.len() != 3
                || callback.arguments[0].ty != element
                || callback.arguments[1..].iter().any(|arg| {
                    module.types[arg.ty].inner
                        != (TypeInner::Vector {
                            size: crate::VectorSize::Bi,
                            scalar: Scalar::U32,
                        })
                })
            {
                return Err(invalid(
                    "expected (block value, vec2u block coordinate, vec2u element coordinate)",
                ));
            }
            let result = callback
                .result
                .as_ref()
                .ok_or_else(|| invalid("decoder must return a value"))?;
            if vector {
                if !self
                    .capabilities
                    .contains(Capabilities::COOPERATIVE_MATRIX_DECODE_VECTOR)
                {
                    return Err(ExpressionError::MissingCapabilities(
                        Capabilities::COOPERATIVE_MATRIX_DECODE_VECTOR,
                    ));
                }
                let TypeInner::Vector {
                    size,
                    scalar: result_scalar,
                } = module.types[result.ty].inner
                else {
                    return Err(invalid(
                        "vector decoder must return vec2 or vec4 of the matrix component",
                    ));
                };
                if !matches!(size, crate::VectorSize::Bi | crate::VectorSize::Quad)
                    || result_scalar != scalar
                    || !decode.block_size[1].is_multiple_of(size as u32)
                {
                    return Err(invalid(
                        "vector width must be 2 or 4 and divide the innermost block dimension",
                    ));
                }
            } else if module.types[result.ty].inner != TypeInner::Scalar(scalar) {
                return Err(invalid(
                    "scalar decoder must return the matrix component type",
                ));
            }
            Self::validate_cooperative_callback(function, module, info)?;
        }
        Ok(())
    }

    pub(super) fn validate_cooperative_callback(
        callback: Handle<Function>,
        module: &Module,
        info: &ModuleInfo,
    ) -> Result<(), ExpressionError> {
        let invalid = |reason| ExpressionError::InvalidCooperativeCallback {
            function: callback,
            reason,
        };
        let mut functions = vec![callback];
        let mut seen = crate::FastHashSet::default();
        while let Some(handle) = functions.pop() {
            if !seen.insert(handle) {
                continue;
            }
            let function = &module.functions[handle];
            let function_info = &info[handle];
            if function_info
                .global_uses
                .iter()
                .any(|usage| !usage.is_empty())
            {
                return Err(invalid("callbacks may not access global memory"));
            }
            for (expression, value) in function.expressions.iter() {
                if matches!(
                    function_info[expression].ty.inner_with(&module.types),
                    TypeInner::CooperativeMatrix { .. }
                ) || matches!(value, crate::Expression::Derivative { .. })
                {
                    return Err(invalid(
                        "callbacks may not contain cooperative or derivative operations",
                    ));
                }
            }
            let mut blocks = vec![&function.body];
            while let Some(block) = blocks.pop() {
                for statement in block {
                    use crate::Statement as S;
                    match *statement {
                        S::Emit(_) | S::Break | S::Continue | S::Return { .. } | S::Store { .. } => {}
                        S::Block(ref block) => blocks.push(block),
                        S::If { ref accept, ref reject, .. } => { blocks.push(accept); blocks.push(reject); }
                        S::Switch { ref cases, .. } => blocks.extend(cases.iter().map(|case| &case.body)),
                        S::Loop { ref body, ref continuing, .. } => { blocks.push(body); blocks.push(continuing); }
                        S::Call { function, .. } => functions.push(function),
                        _ => return Err(invalid("callbacks may not synchronize, access resources, or have external side effects")),
                    }
                }
            }
        }
        Ok(())
    }
}
