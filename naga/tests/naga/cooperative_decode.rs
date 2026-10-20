#![cfg(all(feature = "wgsl-in", feature = "wgsl-out", spv_out))]

use naga::{
    back::spv,
    compact::KeepUnused,
    valid::{Capabilities, ValidationFlags, Validator},
};
use rspirv::binary::Disassemble;

const SCALAR: &str = include_str!("../in/wgsl/cooperative-matrix-decode.wgsl");
const VECTOR: &str = include_str!("../in/wgsl/cooperative-matrix-decode-vector.wgsl");

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
