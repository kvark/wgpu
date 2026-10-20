use alloc::{boxed::Box, collections::BTreeSet, vec, vec::Vec};

use super::FunctionInfo;
use crate::span::{AddSpan as _, WithSpan};
use crate::valid::FunctionError;
use crate::{Expression as E, Handle, Statement as S, TypeInner as Ti};

#[derive(Clone, Debug, Default, PartialEq)]
struct Dependencies {
    divergent: bool,
    subgroup_divergent: bool,
    arguments: BTreeSet<u32>,
}

impl Dependencies {
    fn merge(&mut self, other: &Self) {
        self.divergent |= other.divergent;
        self.subgroup_divergent |= other.subgroup_divergent;
        self.arguments.extend(&other.arguments);
    }

    fn substitute(&self, arguments: &[Handle<E>], expressions: &[Self]) -> Self {
        let mut result = Self {
            divergent: self.divergent,
            subgroup_divergent: self.subgroup_divergent,
            ..Self::default()
        };
        for &index in &self.arguments {
            result.merge(&expressions[arguments[index as usize].index()]);
        }
        result
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct Summary {
    result: Dependencies,
    required: Dependencies,
    cooperative: bool,
    required_subgroup: Dependencies,
    cooperative_subgroup: bool,
}

#[derive(Clone, Debug, Default)]
enum Source {
    #[default]
    Uniform,
    SubgroupUniform,
    NonUniform,
    Argument(u32),
    Local(Handle<crate::LocalVariable>),
    Call,
}

#[derive(Clone, Debug, Default)]
struct ExpressionDependencies {
    source: Source,
    references: Vec<Handle<E>>,
}

#[derive(Clone, Debug)]
/// Operand dependencies collected by the main expression analyzer.
pub(super) struct Graph {
    expressions: Vec<ExpressionDependencies>,
    current_expression: Option<Handle<E>>,
    entry_point: bool,
}

impl Graph {
    pub(super) fn new(expressions: usize, entry_point: bool) -> Self {
        Self {
            expressions: vec![ExpressionDependencies::default(); expressions],
            current_expression: None,
            entry_point,
        }
    }

    pub(super) const fn begin_expression(&mut self, handle: Handle<E>) {
        self.current_expression = Some(handle);
    }

    pub(super) fn add_ref(&mut self, input: Handle<E>) {
        if let Some(handle) = self.current_expression {
            self.expressions[handle.index()].references.push(input);
        }
    }

    pub(super) fn finish_expression(
        &mut self,
        arena: &crate::Arena<E>,
        context: &crate::proc::ResolveContext<'_>,
    ) {
        let handle = self.current_expression.take().unwrap();
        let dependencies = &mut self.expressions[handle.index()];
        dependencies.source = match arena[handle] {
            E::FunctionArgument(index) if !self.entry_point => Source::Argument(index),
            E::FunctionArgument(index) => {
                if uniform_builtin(&context.arguments[index as usize].binding) {
                    Source::Uniform
                } else if subgroup_uniform_builtin(&context.arguments[index as usize].binding) {
                    Source::SubgroupUniform
                } else {
                    Source::NonUniform
                }
            }
            E::AccessIndex { base, index } if self.entry_point => {
                if let E::FunctionArgument(argument) = arena[base] {
                    if let Ti::Struct { ref members, .. } =
                        context.types[context.arguments[argument as usize].ty].inner
                    {
                        if uniform_builtin(&members[index as usize].binding) {
                            dependencies.references.clear();
                        } else if subgroup_uniform_builtin(&members[index as usize].binding) {
                            dependencies.references.clear();
                            dependencies.source = Source::SubgroupUniform;
                            return;
                        }
                    }
                }
                Source::Uniform
            }
            E::Load { pointer } => match arena[root(arena, pointer)] {
                E::LocalVariable(local) => Source::Local(local),
                E::GlobalVariable(global) => match context.global_vars[global].space {
                    crate::AddressSpace::Uniform | crate::AddressSpace::Immediate => {
                        Source::Uniform
                    }
                    crate::AddressSpace::Storage { access }
                        if !access.contains(crate::StorageAccess::STORE) =>
                    {
                        Source::Uniform
                    }
                    _ => Source::NonUniform,
                },
                _ => Source::NonUniform,
            },
            E::CallResult(_) => Source::Call,
            E::Literal(_)
            | E::Constant(_)
            | E::Override(_)
            | E::ZeroValue(_)
            | E::GlobalVariable(_)
            | E::LocalVariable(_)
            | E::WorkGroupUniformLoadResult { .. }
            | E::Access { .. }
            | E::AccessIndex { .. }
            | E::Splat { .. }
            | E::Swizzle { .. }
            | E::Compose { .. }
            | E::Unary { .. }
            | E::Binary { .. }
            | E::Select { .. }
            | E::Math { .. }
            | E::Relational { .. }
            | E::As { .. }
            | E::ArrayLength(_)
            | E::CooperativeLoad { .. }
            | E::CooperativeConvert { .. }
            | E::CooperativeTensorLoad { .. }
            | E::CooperativeMultiplyAdd { .. } => Source::Uniform,
            _ => Source::NonUniform,
        };
    }
}

fn root(arena: &crate::Arena<E>, mut pointer: Handle<E>) -> Handle<E> {
    loop {
        match arena[pointer] {
            E::Access { base, .. } | E::AccessIndex { base, .. } => pointer = base,
            _ => return pointer,
        }
    }
}

struct Analysis<'a> {
    module: &'a crate::Module,
    function: &'a crate::Function,
    info: &'a FunctionInfo,
    callees: &'a [FunctionInfo],
    graph: Graph,
    expressions: Vec<Dependencies>,
    locals: Vec<Dependencies>,
    summary: Summary,
}

const fn uniform_builtin(binding: &Option<crate::Binding>) -> bool {
    matches!(
        *binding,
        Some(crate::Binding::BuiltIn(
            crate::BuiltIn::WorkGroupId
                | crate::BuiltIn::WorkGroupSize
                | crate::BuiltIn::NumWorkGroups
        ))
    )
}

const fn subgroup_uniform_builtin(binding: &Option<crate::Binding>) -> bool {
    matches!(
        *binding,
        Some(crate::Binding::BuiltIn(
            crate::BuiltIn::SubgroupId
                | crate::BuiltIn::SubgroupSize
                | crate::BuiltIn::NumSubgroups
        ))
    )
}

impl Analysis<'_> {
    fn root(&self, pointer: Handle<E>) -> Handle<E> {
        root(&self.function.expressions, pointer)
    }

    fn expression(&self, handle: Handle<E>) -> Dependencies {
        let expression = &self.graph.expressions[handle.index()];
        let mut result = match expression.source {
            Source::Uniform => Dependencies::default(),
            Source::SubgroupUniform => Dependencies {
                divergent: true,
                ..Dependencies::default()
            },
            Source::NonUniform => Dependencies {
                divergent: true,
                subgroup_divergent: true,
                ..Dependencies::default()
            },
            Source::Argument(index) => Dependencies {
                arguments: [index].into_iter().collect(),
                ..Dependencies::default()
            },
            Source::Local(local) => self.locals[local.index()].clone(),
            Source::Call => self.expressions[handle.index()].clone(),
        };
        for &input in &expression.references {
            result.merge(&self.expressions[input.index()]);
        }
        result
    }

    fn require(
        &mut self,
        operand: Handle<E>,
        control: &Dependencies,
    ) -> Result<(), WithSpan<FunctionError>> {
        self.require_at(operand, control, crate::CooperativeScope::Workgroup)
    }

    fn require_at(
        &mut self,
        operand: Handle<E>,
        control: &Dependencies,
        scope: crate::CooperativeScope,
    ) -> Result<(), WithSpan<FunctionError>> {
        let mut dependencies = self.expressions[operand.index()].clone();
        dependencies.merge(control);
        let subgroup = scope == crate::CooperativeScope::Subgroup;
        if if subgroup {
            dependencies.subgroup_divergent
        } else {
            dependencies.divergent
        } {
            return Err(FunctionError::NonUniformCooperativeOperand(operand)
                .with_span_handle(operand, &self.function.expressions));
        }
        if subgroup {
            self.summary.cooperative_subgroup = true;
            self.summary.required_subgroup.merge(&dependencies);
        } else {
            self.summary.cooperative = true;
            self.summary.required.merge(&dependencies);
        }
        Ok(())
    }

    fn matrix_scope(&self, expr: Handle<E>) -> crate::CooperativeScope {
        match *self.info[expr].ty.inner_with(&self.module.types) {
            Ti::CooperativeMatrix { scope, .. } => scope,
            _ => unreachable!(),
        }
    }

    fn workgroup_matrix(&self, expr: Handle<E>) -> bool {
        matches!(
            *self.info[expr].ty.inner_with(&self.module.types),
            Ti::CooperativeMatrix {
                scope: crate::CooperativeScope::Workgroup,
                ..
            }
        )
    }

    fn exits(&self, block: &crate::Block, control: &Dependencies, nested: bool) -> Dependencies {
        let mut result = Dependencies::default();
        for statement in block {
            match *statement {
                S::Break | S::Continue if !nested => result.merge(control),
                S::Return { .. } | S::Kill => result.merge(control),
                S::Block(ref block) => result.merge(&self.exits(block, control, nested)),
                S::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    let mut branch = control.clone();
                    branch.merge(&self.expressions[condition.index()]);
                    result.merge(&self.exits(accept, &branch, nested));
                    result.merge(&self.exits(reject, &branch, nested));
                }
                S::Switch {
                    selector,
                    ref cases,
                } => {
                    let mut branch = control.clone();
                    branch.merge(&self.expressions[selector.index()]);
                    for case in cases {
                        result.merge(&self.exits(&case.body, &branch, nested));
                    }
                }
                S::Loop {
                    ref body,
                    ref continuing,
                    ..
                } => {
                    result.merge(&self.exits(body, control, true));
                    result.merge(&self.exits(continuing, control, true));
                }
                _ => {}
            }
        }
        result
    }

    fn block(
        &mut self,
        block: &crate::Block,
        control: &Dependencies,
        check: bool,
    ) -> Result<(), WithSpan<FunctionError>> {
        let mut control = control.clone();
        for statement in block {
            match *statement {
                S::Emit(ref range) if check => {
                    for handle in range.clone() {
                        match self.function.expressions[handle] {
                            E::CooperativeLoad {
                                scope: crate::CooperativeScope::Workgroup,
                                ref data,
                                ..
                            } => {
                                self.require(data.pointer, &control)?;
                                self.require(data.stride, &control)?;
                            }
                            E::CooperativeTensorLoad { ref data, .. } => {
                                for operand in data.operands() {
                                    self.require_at(operand, &control, self.matrix_scope(handle))?;
                                }
                            }
                            E::CooperativeConvert { source, .. } => {
                                self.require_at(source, &control, self.matrix_scope(handle))?;
                            }
                            E::CooperativeMultiplyAdd { a, .. } if self.workgroup_matrix(a) => {
                                self.summary.cooperative = true;
                                if control.divergent {
                                    return Err(FunctionError::NonUniformCooperativeOperand(
                                        handle,
                                    )
                                    .with_span_handle(handle, &self.function.expressions));
                                }
                                self.summary.required.merge(&control);
                            }
                            _ => {}
                        }
                    }
                }
                S::Store { pointer, value } => {
                    if let E::LocalVariable(local) = self.function.expressions[self.root(pointer)] {
                        let dependencies = self.locals.get_mut(local.index()).unwrap();
                        dependencies.merge(&self.expressions[value.index()]);
                        dependencies.merge(&self.expressions[pointer.index()]);
                        dependencies.merge(&control);
                    }
                }
                S::Block(ref body) => self.block(body, &control, check)?,
                S::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    let mut branch = control.clone();
                    branch.merge(&self.expressions[condition.index()]);
                    self.block(accept, &branch, check)?;
                    self.block(reject, &branch, check)?;
                    control.merge(&self.exits(accept, &branch, true));
                    control.merge(&self.exits(reject, &branch, true));
                }
                S::Switch {
                    selector,
                    ref cases,
                } => {
                    let mut branch = control.clone();
                    branch.merge(&self.expressions[selector.index()]);
                    for case in cases {
                        self.block(&case.body, &branch, check)?;
                    }
                }
                S::Loop {
                    ref body,
                    ref continuing,
                    break_if,
                } => {
                    let mut loop_control = control.clone();
                    loop_control.merge(&self.exits(body, &control, false));
                    loop_control.merge(&self.exits(continuing, &control, false));
                    if let Some(condition) = break_if {
                        loop_control.merge(&self.expressions[condition.index()]);
                    }
                    self.block(body, &loop_control, check)?;
                    self.block(continuing, &loop_control, check)?;
                    control.merge(&self.exits(body, &control, true));
                }
                S::Call {
                    function,
                    ref arguments,
                    result,
                } => {
                    let summary = self.callees[function.index()]
                        .operand_uniformity
                        .as_deref()
                        .expect("callee operand uniformity has been analyzed");
                    if let Some(result) = result {
                        let mut value = summary.result.substitute(arguments, &self.expressions);
                        value.merge(&control);
                        self.expressions[result.index()].merge(&value);
                    }
                    if check {
                        if summary.cooperative {
                            self.summary.cooperative = true;
                            if control.divergent {
                                return Err(
                                    FunctionError::NonUniformCooperativeCall(function).with_span()
                                );
                            }
                            self.summary.required.merge(&control);
                        }
                        for &index in &summary.required.arguments {
                            self.require(arguments[index as usize], &control)?;
                        }
                        if summary.cooperative_subgroup {
                            self.summary.cooperative_subgroup = true;
                            if control.subgroup_divergent {
                                return Err(
                                    FunctionError::NonUniformCooperativeCall(function).with_span()
                                );
                            }
                            self.summary.required_subgroup.merge(&control);
                        }
                        for &index in &summary.required_subgroup.arguments {
                            self.require_at(
                                arguments[index as usize],
                                &control,
                                crate::CooperativeScope::Subgroup,
                            )?;
                        }
                    }
                    for &argument in arguments {
                        if let E::LocalVariable(local) =
                            self.function.expressions[self.root(argument)]
                        {
                            self.locals[local.index()].divergent = true;
                            self.locals[local.index()].subgroup_divergent = true;
                        }
                    }
                }
                S::Return { value: Some(value) } => {
                    self.summary.result.merge(&self.expressions[value.index()]);
                    self.summary.result.merge(&control);
                }
                S::CooperativeTensorStore { target, ref data } if check => {
                    for operand in data.operands() {
                        self.require_at(operand, &control, self.matrix_scope(target))?;
                    }
                }
                S::CooperativeStore { target, ref data }
                    if check && self.workgroup_matrix(target) =>
                {
                    for operand in [data.pointer, data.stride] {
                        self.require(operand, &control)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn run(mut self) -> Result<Summary, WithSpan<FunctionError>> {
        // Local assignments and loop-carried values form a monotone dependency graph.
        loop {
            let old_locals = self.locals.clone();
            let old_expressions = self.expressions.clone();
            for (handle, _) in self.function.expressions.iter() {
                let value = self.expression(handle);
                self.expressions[handle.index()].merge(&value);
            }
            for (handle, local) in self.function.local_variables.iter() {
                if let Some(init) = local.init {
                    self.locals[handle.index()].merge(&self.expressions[init.index()]);
                }
            }
            self.block(&self.function.body, &Dependencies::default(), false)?;
            if old_locals == self.locals && old_expressions == self.expressions {
                break;
            }
        }
        self.block(&self.function.body, &Dependencies::default(), true)?;
        Ok(self.summary)
    }
}

impl FunctionInfo {
    pub(in crate::valid) fn validate_operand_uniformity(
        &mut self,
        function: &crate::Function,
        module: &crate::Module,
        callees: &[FunctionInfo],
    ) -> Result<(), WithSpan<FunctionError>> {
        let Some(graph) = self.uniformity_graph.take() else {
            return Ok(());
        };
        let summary = Analysis {
            module,
            function,
            info: self,
            callees,
            graph: *graph,
            expressions: vec![Dependencies::default(); function.expressions.len()],
            locals: vec![Dependencies::default(); function.local_variables.len()],
            summary: Summary::default(),
        }
        .run()?;
        self.operand_uniformity = Some(Box::new(summary));
        Ok(())
    }
}
