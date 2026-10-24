#![cfg(feature = "wgsl-in")]

use naga::valid::{Capabilities, ValidationFlags, Validator};

const SOURCE: &str = include_str!("../in/wgsl/fence-publication.wgsl");

#[test]
fn memory_fences_require_ir_capability_and_reject_unsupported_metal() {
    let source = SOURCE.replace("@coherent ", "");
    assert!(
        naga::front::wgsl::parse_str(&source.replace("enable wgpu_memory_fence;", "")).is_err()
    );
    assert!(
        naga::front::wgsl::Frontend::new_with_options(naga::front::wgsl::Options {
            capabilities: Capabilities::all() - Capabilities::MEMORY_FENCE,
            ..naga::front::wgsl::Options::new()
        })
        .parse(&source)
        .is_err()
    );
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    let error = Validator::new(
        ValidationFlags::all(),
        Capabilities::all() - Capabilities::MEMORY_FENCE,
    )
    .validate(&module)
    .unwrap_err();
    assert!(matches!(
        error.as_inner(),
        naga::valid::ValidationError::EntryPoint {
            source: naga::valid::EntryPointError::Function(
                naga::valid::FunctionError::MissingCapability(Capabilities::MEMORY_FENCE)
            ),
            ..
        }
    ));
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    #[cfg(msl_out)]
    for version in [(2, 4), (3, 2)] {
        let result = naga::back::msl::write_string(
            &module,
            &info,
            &naga::back::msl::Options {
                lang_version: version,
                ..Default::default()
            },
            &Default::default(),
        );
        assert!(matches!(
            result,
            Err(naga::back::msl::Error::CapabilityNotSupported(
                Capabilities::MEMORY_FENCE
            ))
        ));
    }
    let _ = info;
}

#[cfg(all(spv_out, feature = "wgsl-out"))]
#[test]
fn publication_fences_preserve_scopes_and_relaxed_atomics() {
    use rspirv::binary::Disassemble;

    let mut module = naga::front::wgsl::parse_str(SOURCE).unwrap();
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    let roundtrip =
        naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
            .unwrap();
    let mut seen_atomics = naga::FastHashSet::default();
    for (source, publication) in [
        (SOURCE, true),
        (roundtrip.as_str(), true),
        (include_str!("../in/wgsl/atomicOps.wgsl"), false),
        (include_str!("../in/wgsl/atomicCompareExchange.wgsl"), false),
    ] {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap();
        let words = naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options {
                use_vulkan_memory_model: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let parsed = rspirv::dr::load_words(&words).unwrap();
        #[cfg(feature = "spv-in")]
        if publication {
            for strict_capabilities in [false, true] {
                let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                let error = naga::front::spv::parse_u8_slice(
                    &bytes,
                    &naga::front::spv::Options {
                        strict_capabilities,
                        ..Default::default()
                    },
                )
                .unwrap_err();
                assert!(matches!(
                    error,
                    naga::front::spv::Error::UnsupportedMemoryAccess(_)
                ));
            }
        }
        #[cfg(feature = "spv-in")]
        if publication {
            let mut core_model = parsed.clone();
            core_model.extensions.clear();
            core_model.header.as_mut().unwrap().version = 0x0001_0500;
            let bytes: Vec<_> = rspirv::binary::Assemble::assemble(&core_model)
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            assert!(matches!(
                naga::front::spv::parse_u8_slice(
                    &bytes,
                    &naga::front::spv::Options {
                        strict_capabilities: false,
                        ..Default::default()
                    }
                ),
                Err(naga::front::spv::Error::UnsupportedMemoryAccess(_))
            ));
        }
        let assembly = parsed.disassemble();
        assert!(
            assembly.contains("OpMemoryModel Logical Vulkan"),
            "{assembly}"
        );
        if publication {
            assert!(
                assembly.contains("MakePointerVisible|NonPrivatePointer"),
                "{assembly}"
            );
        }
        assert!(!assembly.contains("OpCapability VulkanMemoryModelDeviceScope"));
        let constants: naga::FastHashMap<_, _> = parsed
            .types_global_values
            .iter()
            .filter(|instruction| instruction.class.opcode == spirv::Op::Constant)
            .map(|instruction| {
                (
                    instruction.result_id.unwrap(),
                    instruction.operands[0].unwrap_literal_bit32(),
                )
            })
            .collect();
        let value = |operand: &rspirv::dr::Operand| constants[&operand.id_ref_any().unwrap()];
        let mut fences = 0;
        let mut atomics = 0;
        for instruction in parsed.all_inst_iter() {
            let operands = &instruction.operands;
            match instruction.class.opcode {
                spirv::Op::MemoryBarrier => {
                    fences += 1;
                    assert_eq!(value(&operands[0]), spirv::Scope::QueueFamily as u32);
                    assert_eq!(
                        value(&operands[1]),
                        (spirv::MemorySemantics::ACQUIRE_RELEASE
                            | spirv::MemorySemantics::UNIFORM_MEMORY
                            | spirv::MemorySemantics::MAKE_AVAILABLE
                            | spirv::MemorySemantics::MAKE_VISIBLE)
                            .bits()
                    );
                }
                spirv::Op::AtomicLoad
                | spirv::Op::AtomicStore
                | spirv::Op::AtomicExchange
                | spirv::Op::AtomicCompareExchange
                | spirv::Op::AtomicIAdd
                | spirv::Op::AtomicISub
                | spirv::Op::AtomicSMin
                | spirv::Op::AtomicUMin
                | spirv::Op::AtomicSMax
                | spirv::Op::AtomicUMax
                | spirv::Op::AtomicAnd
                | spirv::Op::AtomicOr
                | spirv::Op::AtomicXor => {
                    atomics += 1;
                    seen_atomics.insert(instruction.class.opcode as u32);
                    assert!([
                        spirv::Scope::QueueFamily as u32,
                        spirv::Scope::Workgroup as u32,
                    ]
                    .contains(&value(&operands[1])));
                    assert_eq!(value(&operands[2]), 0);
                    if instruction.class.opcode == spirv::Op::AtomicCompareExchange {
                        assert_eq!(value(&operands[3]), 0);
                    }
                }
                _ => {}
            }
        }
        assert_eq!(fences, if publication { 4 } else { 0 });
        assert!(atomics >= 2);
    }
    for opcode in [
        spirv::Op::AtomicLoad,
        spirv::Op::AtomicStore,
        spirv::Op::AtomicCompareExchange,
    ] {
        assert!(seen_atomics.contains(&(opcode as u32)));
    }
}
