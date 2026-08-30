use half::f16;
use mlx_rs::{
    io::{GgufFile, GgufMetadataKind, GgufMetadataValue},
    ops, Array,
};

#[test]
fn rust_save_then_rust_load_round_trip_self_oracle() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("round-trip.gguf");
    let source = Array::from_slice(&(0..15).collect::<Vec<i32>>(), &[3, 5]);
    let non_contiguous = ops::transpose(&source).unwrap();

    let mut file = GgufFile::new().unwrap();
    file.insert_array("matrix", &non_contiguous).unwrap();
    file.insert_metadata("count", Array::from_int(15)).unwrap();
    file.insert_metadata("model", "self-oracle").unwrap();
    file.insert_metadata("tags", vec!["gguf".to_owned(), "round-trip".to_owned()])
        .unwrap();
    file.save(&path).unwrap();

    let loaded = GgufFile::load(&path).unwrap();
    assert_eq!(loaded.array_keys().unwrap(), ["matrix"]);
    let matrix = loaded.get_array("matrix").unwrap().unwrap();
    assert_eq!(matrix.shape(), &[5, 3]);
    assert_eq!(
        matrix.as_slice::<i32>(),
        &[0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14]
    );
    assert_eq!(
        loaded.metadata_kind("count").unwrap(),
        Some(GgufMetadataKind::Array)
    );
    assert_eq!(
        loaded.get_metadata_string("model").unwrap().as_deref(),
        Some("self-oracle")
    );
    assert_eq!(
        loaded.get_metadata_strings("tags").unwrap().unwrap(),
        ["gguf", "round-trip"]
    );
    assert!(matches!(
        loaded.get_metadata("model").unwrap(),
        Some(GgufMetadataValue::String(value)) if value == "self-oracle"
    ));
}

#[test]
#[ignore]
fn write_save_qualification_artifact() {
    let path = std::env::var_os("GGUF_QUALIFY_OUT").expect("GGUF_QUALIFY_OUT must be set");
    let large_values = (0..129 * 257).map(|value| value as f32).collect::<Vec<_>>();
    let large = Array::from_slice(&large_values, &[129, 257]);
    let non_contiguous = ops::transpose(&large).unwrap();
    let values = [-3, -1, 0, 2, 7, 11];

    let mut file = GgufFile::new().unwrap();
    file.insert_array("shared", &non_contiguous).unwrap();
    file.insert_array(
        "tensor.f32",
        &Array::from_slice(&values.map(|value| value as f32), &[2, 3]),
    )
    .unwrap();
    file.insert_array(
        "tensor.f16",
        &Array::from_slice(&values.map(|value| f16::from_f32(value as f32)), &[2, 3]),
    )
    .unwrap();
    file.insert_array(
        "tensor.i8",
        &Array::from_slice(&values.map(|value| value as i8), &[2, 3]),
    )
    .unwrap();
    file.insert_array(
        "tensor.i16",
        &Array::from_slice(&values.map(|value| value as i16), &[2, 3]),
    )
    .unwrap();
    file.insert_array("tensor.i32", &Array::from_slice(&values, &[2, 3]))
        .unwrap();
    file.insert_metadata("shared", "metadata").unwrap();
    file.insert_metadata("metadata.array", Array::from_slice(&[17_i32, 19_i32], &[2]))
        .unwrap();
    file.insert_metadata("metadata.string", "qualification")
        .unwrap();
    file.insert_metadata(
        "metadata.strings",
        vec!["one".to_owned(), "two".to_owned(), "three".to_owned()],
    )
    .unwrap();
    file.save(path).unwrap();
}
