use super::adapters::{dispatch, ADAPTERS};
use half::{bf16, f16};
use mlx_rs::{ops, with_stream, Array, Device, Dtype, Stream};
use num_complex::Complex32;
use safetensors::{tensor::Dtype as SafeDtype, SafeTensors};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    schema_version: u32,
    corpus_seed: String,
    rng: Rng,
    canonical_device: String,
    generator_digest: String,
    fixture_shards: Option<BTreeMap<String, String>>,
    gguf_fixtures: Option<serde_json::Value>,
    environment: Environment,
    tolerance_policies: BTreeMap<String, Policy>,
    suites: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Rng {
    algorithm: String,
    case_seed_hash: String,
    seed_bytes: usize,
    byte_order: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Environment {
    python: String,
    architecture: String,
    mlx_package: String,
    mlx_metal_package: String,
    mlx_runtime: String,
    numpy: String,
}

#[derive(Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Policy {
    ExactNumeric,
    Float {
        atol: f64,
        rtol: f64,
        nan_equal: bool,
        infinity_sign: bool,
        signed_zero: bool,
        complex: ComplexRule,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ComplexRule {
    Componentwise,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Suite {
    schema_version: u32,
    name: String,
    fixture: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Case {
    id: String,
    semantic_op: String,
    recipe: String,
    pub(super) rust_call: String,
    pub(super) args: Vec<Arg>,
    expected: Expected,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Arg {
    Tensor {
        name: String,
        #[serde(rename = "ref")]
        tensor_ref: String,
        encoding: Option<String>,
        imag_ref: Option<String>,
    },
    Scalar {
        name: String,
        #[serde(rename = "type")]
        scalar_type: String,
        value: Option<serde_json::Value>,
        bits: Option<String>,
        real_bits: Option<String>,
        imag_bits: Option<String>,
    },
    Axes {
        name: String,
        values: Vec<i32>,
    },
    Axis {
        name: String,
        value: i32,
    },
    OptionalAxis {
        name: String,
        value: Option<i32>,
    },
    Shape {
        name: String,
        values: Vec<i32>,
    },
    OptionalBool {
        name: String,
        value: Option<bool>,
    },
    Dtype {
        name: String,
        value: String,
    },
    Execution {
        name: String,
        target: ExecutionTarget,
    },
}

impl Arg {
    pub(super) fn name(&self) -> &str {
        match self {
            Self::Tensor { name, .. }
            | Self::Scalar { name, .. }
            | Self::Axes { name, .. }
            | Self::Axis { name, .. }
            | Self::OptionalAxis { name, .. }
            | Self::Shape { name, .. }
            | Self::OptionalBool { name, .. }
            | Self::Dtype { name, .. }
            | Self::Execution { name, .. } => name,
        }
    }
}

#[derive(Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum ExecutionTarget {
    DefaultCpu,
    ExplicitCpu,
}

#[derive(Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
enum Expected {
    Success {
        provenance: Provenance,
        outputs: Vec<ExpectedOutput>,
    },
    Error {
        allowed_stage: AllowedStage,
        reason: String,
        python_exception: PythonException,
        control_case_id: String,
        diagnostic: String,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Provenance {
    NumpyCorroborated,
    MlxPython,
}

#[derive(Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum AllowedStage {
    InvokeOrEval,
    EvalOnly,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PythonException {
    module: String,
    #[serde(rename = "type")]
    exception_type: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedOutput {
    name: String,
    #[serde(rename = "ref")]
    tensor_ref: String,
    dtype: String,
    shape: Vec<i32>,
    policy: String,
    encoding: Option<String>,
    imag_ref: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Qualification {
    schema_version: u32,
    mutations: Vec<Mutation>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Mutation {
    id: String,
    base_case_id: String,
    kind: String,
    expected_class: String,
}

struct LoadedSuite {
    suite: Suite,
    bytes: Vec<u8>,
}

struct LoadedCorpus {
    root: PathBuf,
    corpus: Corpus,
    suites: Vec<LoadedSuite>,
}

#[derive(Clone)]
enum TensorData {
    Bool(Vec<u8>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    C64(Vec<Complex32>),
}

#[derive(Clone)]
struct HostTensor {
    dtype: Dtype,
    shape: Vec<i32>,
    data: TensorData,
}

impl HostTensor {
    fn len(&self) -> usize {
        match &self.data {
            TensorData::Bool(v) | TensorData::U8(v) => v.len(),
            TensorData::U16(v) => v.len(),
            TensorData::U32(v) => v.len(),
            TensorData::U64(v) => v.len(),
            TensorData::I8(v) => v.len(),
            TensorData::I16(v) => v.len(),
            TensorData::I32(v) => v.len(),
            TensorData::I64(v) => v.len(),
            TensorData::F16(v) => v.len(),
            TensorData::BF16(v) => v.len(),
            TensorData::F32(v) => v.len(),
            TensorData::F64(v) => v.len(),
            TensorData::C64(v) => v.len(),
        }
    }

    fn to_array(&self) -> Array {
        match &self.data {
            TensorData::Bool(values) => unsafe {
                Array::from_raw_data(values.as_ptr().cast(), &self.shape, Dtype::Bool)
            },
            TensorData::U8(values) => Array::from_slice(values, &self.shape),
            TensorData::U16(values) => Array::from_slice(values, &self.shape),
            TensorData::U32(values) => Array::from_slice(values, &self.shape),
            TensorData::U64(values) => Array::from_slice(values, &self.shape),
            TensorData::I8(values) => Array::from_slice(values, &self.shape),
            TensorData::I16(values) => Array::from_slice(values, &self.shape),
            TensorData::I32(values) => Array::from_slice(values, &self.shape),
            TensorData::I64(values) => Array::from_slice(values, &self.shape),
            TensorData::F16(values) => Array::from_slice(values, &self.shape),
            TensorData::BF16(values) => Array::from_slice(values, &self.shape),
            TensorData::F32(values) => Array::from_slice(values, &self.shape),
            TensorData::F64(values) => Array::from_slice_f64(values, &self.shape),
            TensorData::C64(values) => Array::from_slice(values, &self.shape),
        }
    }
}

struct DeviceGuard(Device);

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        Device::set_default(&self.0);
    }
}

fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("mlx-tests has a workspace parent")
        .join("conformance")
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn load_corpus() -> Result<LoadedCorpus, Vec<String>> {
    let root = corpus_root();
    let corpus: Corpus = read_json(&root.join("corpus.json")).map_err(|error| vec![error])?;
    let mut failures = Vec::new();
    if corpus.schema_version != 1 {
        failures.push(format!(
            "corpus schema_version {} is unsupported",
            corpus.schema_version
        ));
    }
    if corpus.canonical_device != "cpu" {
        failures.push("canonical_device must be cpu".into());
    }
    if corpus.corpus_seed.is_empty() || !corpus.generator_digest.starts_with("sha256:") {
        failures.push("corpus identity fields are invalid".into());
    }
    let _ = &corpus.fixture_shards;
    if corpus.rng.algorithm != "numpy.PCG64"
        || corpus.rng.case_seed_hash != "sha256(corpus_seed || NUL || case_id)"
        || corpus.rng.seed_bytes != 16
        || corpus.rng.byte_order != "little"
    {
        failures.push("RNG declaration is invalid".into());
    }
    if corpus.environment.python != "3.12.14"
        || corpus.environment.architecture != "arm64"
        || corpus.environment.mlx_package != "0.32.2"
        || corpus.environment.mlx_metal_package != "0.32.2"
        || corpus.environment.mlx_runtime != "0.32.2"
        || corpus.environment.numpy != "2.2.6"
    {
        failures.push("environment provenance does not match the reference lock".into());
    }
    let required_policies = [
        "exact_bits",
        "exact_numeric",
        "elementwise_float",
        "low_precision_float",
        "reduction_float",
    ];
    if corpus
        .tolerance_policies
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != required_policies.into_iter().collect()
    {
        failures.push("tolerance registry names are invalid".into());
    }
    let mut suites = Vec::new();
    let mut suite_names = BTreeSet::new();
    for suite_rel in &corpus.suites {
        if suite_rel == "suites/gguf.json" {
            continue;
        }
        let expected_prefix = "suites/";
        if !suite_rel.starts_with(expected_prefix) || suite_rel.contains("..") {
            failures.push(format!("invalid suite path {suite_rel}"));
            continue;
        }
        let suite_path = root.join(suite_rel);
        let suite: Suite = match read_json(&suite_path) {
            Ok(value) => value,
            Err(error) => {
                failures.push(error);
                continue;
            }
        };
        if suite.schema_version != 1 {
            failures.push(format!("{} has unsupported schema", suite.name));
        }
        let expected_suite_path = format!("suites/{}.json", suite.name);
        if *suite_rel != expected_suite_path || !suite_names.insert(suite.name.clone()) {
            failures.push(format!("suite identity mismatch for {suite_rel}"));
        }
        let expected_fixture = format!("fixtures/{}.safetensors", suite.name);
        if suite.fixture != expected_fixture {
            failures.push(format!(
                "{} must name exactly {expected_fixture}",
                suite.name
            ));
        }
        let fixture_path = root.join(&suite.fixture);
        match fs::read(&fixture_path) {
            Ok(bytes) => suites.push(LoadedSuite { suite, bytes }),
            Err(error) => failures.push(format!("{}: {error}", fixture_path.display())),
        }
    }
    if failures.is_empty() {
        Ok(LoadedCorpus {
            root,
            corpus,
            suites,
        })
    } else {
        Err(failures)
    }
}

pub(super) fn dtype_from_name(name: &str) -> Result<Dtype, String> {
    match name {
        "BOOL" => Ok(Dtype::Bool),
        "U8" => Ok(Dtype::Uint8),
        "U16" => Ok(Dtype::Uint16),
        "U32" => Ok(Dtype::Uint32),
        "U64" => Ok(Dtype::Uint64),
        "I8" => Ok(Dtype::Int8),
        "I16" => Ok(Dtype::Int16),
        "I32" => Ok(Dtype::Int32),
        "I64" => Ok(Dtype::Int64),
        "F16" => Ok(Dtype::Float16),
        "BF16" => Ok(Dtype::Bfloat16),
        "F32" => Ok(Dtype::Float32),
        "F64" => Ok(Dtype::Float64),
        "C64" => Ok(Dtype::Complex64),
        _ => Err(format!("unknown dtype {name}")),
    }
}

fn checked_shape(shape: &[usize]) -> Result<(Vec<i32>, usize), String> {
    let converted = shape
        .iter()
        .map(|&dim| i32::try_from(dim).map_err(|_| format!("shape dimension {dim} exceeds i32")))
        .collect::<Result<Vec<_>, _>>()?;
    let product = shape
        .iter()
        .try_fold(1usize, |product, dim| product.checked_mul(*dim))
        .ok_or_else(|| "shape product overflow".to_string())?;
    Ok((converted, product))
}

fn decode_words<T>(bytes: &[u8], width: usize, parse: impl Fn(&[u8]) -> T) -> Vec<T> {
    bytes.chunks_exact(width).map(parse).collect()
}

fn decode_tensor(safe: &SafeTensors<'_>, tensor_ref: &str) -> Result<HostTensor, String> {
    let view = safe
        .tensor(tensor_ref)
        .map_err(|error| format!("{tensor_ref}: {error}"))?;
    let (shape, count) = checked_shape(view.shape())?;
    let data = view.data();
    let width = match view.dtype() {
        SafeDtype::BOOL | SafeDtype::U8 | SafeDtype::I8 => 1,
        SafeDtype::U16 | SafeDtype::I16 | SafeDtype::F16 | SafeDtype::BF16 => 2,
        SafeDtype::U32 | SafeDtype::I32 | SafeDtype::F32 => 4,
        SafeDtype::U64 | SafeDtype::I64 | SafeDtype::F64 => 8,
        other => {
            return Err(format!(
                "{tensor_ref}: unsupported safetensors dtype {other:?}"
            ))
        }
    };
    let expected_bytes = count
        .checked_mul(width)
        .ok_or_else(|| format!("{tensor_ref}: byte count overflow"))?;
    if data.len() != expected_bytes {
        return Err(format!(
            "{tensor_ref}: expected {expected_bytes} bytes, got {}",
            data.len()
        ));
    }
    let (dtype, decoded) = match view.dtype() {
        SafeDtype::BOOL => {
            if let Some((index, value)) = data.iter().enumerate().find(|(_, value)| **value > 1) {
                return Err(format!(
                    "{tensor_ref}: invalid bool byte {value} at {index}"
                ));
            }
            (Dtype::Bool, TensorData::Bool(data.to_vec()))
        }
        SafeDtype::U8 => (Dtype::Uint8, TensorData::U8(data.to_vec())),
        SafeDtype::I8 => (
            Dtype::Int8,
            TensorData::I8(data.iter().map(|value| *value as i8).collect()),
        ),
        SafeDtype::U16 => (
            Dtype::Uint16,
            TensorData::U16(decode_words(data, 2, |v| u16::from_le_bytes([v[0], v[1]]))),
        ),
        SafeDtype::I16 => (
            Dtype::Int16,
            TensorData::I16(decode_words(data, 2, |v| i16::from_le_bytes([v[0], v[1]]))),
        ),
        SafeDtype::F16 => (
            Dtype::Float16,
            TensorData::F16(decode_words(data, 2, |v| {
                f16::from_bits(u16::from_le_bytes([v[0], v[1]]))
            })),
        ),
        SafeDtype::BF16 => (
            Dtype::Bfloat16,
            TensorData::BF16(decode_words(data, 2, |v| {
                bf16::from_bits(u16::from_le_bytes([v[0], v[1]]))
            })),
        ),
        SafeDtype::U32 => (
            Dtype::Uint32,
            TensorData::U32(decode_words(data, 4, |v| {
                u32::from_le_bytes(v.try_into().unwrap())
            })),
        ),
        SafeDtype::I32 => (
            Dtype::Int32,
            TensorData::I32(decode_words(data, 4, |v| {
                i32::from_le_bytes(v.try_into().unwrap())
            })),
        ),
        SafeDtype::F32 => (
            Dtype::Float32,
            TensorData::F32(decode_words(data, 4, |v| {
                f32::from_bits(u32::from_le_bytes(v.try_into().unwrap()))
            })),
        ),
        SafeDtype::U64 => (
            Dtype::Uint64,
            TensorData::U64(decode_words(data, 8, |v| {
                u64::from_le_bytes(v.try_into().unwrap())
            })),
        ),
        SafeDtype::I64 => (
            Dtype::Int64,
            TensorData::I64(decode_words(data, 8, |v| {
                i64::from_le_bytes(v.try_into().unwrap())
            })),
        ),
        SafeDtype::F64 => (
            Dtype::Float64,
            TensorData::F64(decode_words(data, 8, |v| {
                f64::from_bits(u64::from_le_bytes(v.try_into().unwrap()))
            })),
        ),
        _ => unreachable!(),
    };
    Ok(HostTensor {
        dtype,
        shape,
        data: decoded,
    })
}

fn decode_ref(
    safe: &SafeTensors<'_>,
    tensor_ref: &str,
    encoding: Option<&str>,
    imag_ref: Option<&str>,
) -> Result<HostTensor, String> {
    match encoding {
        None => {
            if imag_ref.is_some() {
                return Err(format!("{tensor_ref}: imag_ref without encoding"));
            }
            decode_tensor(safe, tensor_ref)
        }
        Some("complex64_split") => {
            let imag_ref =
                imag_ref.ok_or_else(|| format!("{tensor_ref}: complex encoding needs imag_ref"))?;
            let real = decode_tensor(safe, tensor_ref)?;
            let imag = decode_tensor(safe, imag_ref)?;
            if real.dtype != Dtype::Float32
                || imag.dtype != Dtype::Float32
                || real.shape != imag.shape
            {
                return Err(format!("{tensor_ref}: invalid complex64 split pair"));
            }
            let (TensorData::F32(real_values), TensorData::F32(imag_values)) =
                (real.data, imag.data)
            else {
                unreachable!()
            };
            Ok(HostTensor {
                dtype: Dtype::Complex64,
                shape: real.shape,
                data: TensorData::C64(
                    real_values
                        .into_iter()
                        .zip(imag_values)
                        .map(|(re, im)| Complex32::new(re, im))
                        .collect(),
                ),
            })
        }
        Some(other) => Err(format!("{tensor_ref}: unknown encoding {other}")),
    }
}

fn preflight(loaded: &LoadedCorpus) -> Vec<String> {
    let mut failures = Vec::new();
    let mut global_ids = BTreeSet::new();
    let mut global_previous = None;
    let mut controls = Vec::new();
    let mut success_ids = BTreeSet::new();
    for loaded_suite in &loaded.suites {
        let safe = match SafeTensors::deserialize(&loaded_suite.bytes) {
            Ok(value) => value,
            Err(error) => {
                failures.push(format!(
                    "{} fixture decode: {error}",
                    loaded_suite.suite.name
                ));
                continue;
            }
        };
        let mut previous = None;
        let mut used = BTreeSet::new();
        for case in &loaded_suite.suite.cases {
            if previous.is_some_and(|value: &String| value >= &case.id) {
                failures.push(format!(
                    "{} case IDs are not sorted",
                    loaded_suite.suite.name
                ));
            }
            previous = Some(&case.id);
            if !global_ids.insert(case.id.clone()) {
                failures.push(format!("duplicate case ID {}", case.id));
            }
            if global_previous
                .as_ref()
                .is_some_and(|value: &String| value >= &case.id)
            {
                failures.push("case IDs are not globally sorted".into());
            }
            global_previous = Some(case.id.clone());
            if case.semantic_op != case.recipe {
                failures.push(format!(
                    "{} semantic_op and recipe must name the same registry entry",
                    case.id
                ));
            }
            if !ADAPTERS.contains(&case.rust_call.as_str()) {
                failures.push(format!("{} missing adapter {}", case.id, case.rust_call));
            }
            let mut names = BTreeSet::new();
            let mut executions = 0;
            for arg in &case.args {
                if !names.insert(arg.name()) {
                    failures.push(format!("{} duplicate argument {}", case.id, arg.name()));
                }
                match arg {
                    Arg::Tensor {
                        tensor_ref,
                        encoding,
                        imag_ref,
                        ..
                    } => {
                        used.insert(tensor_ref.clone());
                        if let Some(imag) = imag_ref {
                            used.insert(imag.clone());
                        }
                        if let Err(error) =
                            decode_ref(&safe, tensor_ref, encoding.as_deref(), imag_ref.as_deref())
                        {
                            failures.push(format!("{} input: {error}", case.id));
                        }
                    }
                    Arg::Scalar {
                        scalar_type,
                        value,
                        bits,
                        real_bits,
                        imag_bits,
                        ..
                    } => {
                        let valid = match scalar_type.as_str() {
                            "bool" => {
                                value.as_ref().is_some_and(serde_json::Value::is_boolean)
                                    && bits.is_none()
                                    && real_bits.is_none()
                                    && imag_bits.is_none()
                            }
                            "i32" => {
                                value
                                    .as_ref()
                                    .and_then(serde_json::Value::as_i64)
                                    .and_then(|v| i32::try_from(v).ok())
                                    .is_some()
                                    && bits.is_none()
                                    && real_bits.is_none()
                                    && imag_bits.is_none()
                            }
                            "f32" => {
                                value.is_none()
                                    && parse_hex_u32(bits.as_deref()).is_ok()
                                    && real_bits.is_none()
                                    && imag_bits.is_none()
                            }
                            "complex64" => {
                                value.is_none()
                                    && bits.is_none()
                                    && parse_hex_u32(real_bits.as_deref()).is_ok()
                                    && parse_hex_u32(imag_bits.as_deref()).is_ok()
                            }
                            _ => false,
                        };
                        if !valid {
                            failures.push(format!("{} has invalid scalar {}", case.id, arg.name()));
                        }
                    }
                    Arg::Shape { values, .. } => {
                        if values.iter().filter(|value| **value == -1).count() > 1
                            || values.iter().any(|value| *value < -1)
                        {
                            failures.push(format!("{} has invalid shape argument", case.id));
                        }
                    }
                    Arg::Dtype { value, .. } => {
                        if let Err(error) = dtype_from_name(value) {
                            failures.push(format!("{}: {error}", case.id));
                        }
                    }
                    Arg::Execution { name, target } => {
                        executions += 1;
                        if name != "execution" {
                            failures.push(format!(
                                "{} execution argument must be named execution",
                                case.id
                            ));
                        }
                        let explicit_call = case.rust_call.contains("explicit_cpu");
                        if explicit_call != (*target == ExecutionTarget::ExplicitCpu) {
                            failures.push(format!(
                                "{} execution target does not match rust_call",
                                case.id
                            ));
                        }
                    }
                    _ => {}
                }
            }
            if executions != 1 {
                failures.push(format!(
                    "{} must have exactly one execution argument",
                    case.id
                ));
            }
            match &case.expected {
                Expected::Success {
                    provenance,
                    outputs,
                } => {
                    let _ = provenance;
                    success_ids.insert(case.id.clone());
                    for (index, output) in outputs.iter().enumerate() {
                        if output.name != format!("output{index}") {
                            failures.push(format!(
                                "{} has invalid output order at {}",
                                case.id, output.name
                            ));
                        }
                        used.insert(output.tensor_ref.clone());
                        if let Some(imag) = &output.imag_ref {
                            used.insert(imag.clone());
                        }
                        if !loaded
                            .corpus
                            .tolerance_policies
                            .contains_key(&output.policy)
                        {
                            failures.push(format!(
                                "{} output {} has unknown policy {}",
                                case.id, output.name, output.policy
                            ));
                        }
                        match decode_ref(
                            &safe,
                            &output.tensor_ref,
                            output.encoding.as_deref(),
                            output.imag_ref.as_deref(),
                        ) {
                            Ok(tensor) => {
                                match dtype_from_name(&output.dtype) {
                                    Ok(dtype) if dtype != tensor.dtype => failures.push(format!(
                                        "{} output {} dtype metadata mismatch",
                                        case.id, output.name
                                    )),
                                    Err(error) => failures.push(format!("{}: {error}", case.id)),
                                    _ => {}
                                }
                                if output.shape != tensor.shape {
                                    failures.push(format!(
                                        "{} output {} shape metadata mismatch",
                                        case.id, output.name
                                    ));
                                }
                            }
                            Err(error) => failures.push(format!("{} output: {error}", case.id)),
                        }
                    }
                }
                Expected::Error {
                    allowed_stage: _,
                    reason,
                    python_exception,
                    control_case_id,
                    diagnostic,
                } => {
                    if reason.is_empty()
                        || diagnostic.is_empty()
                        || python_exception.module.is_empty()
                        || python_exception.exception_type.is_empty()
                    {
                        failures.push(format!(
                            "{} has incomplete expected-error provenance",
                            case.id
                        ));
                    }
                    controls.push((case.id.clone(), control_case_id.clone()));
                }
            }
        }
        let names = safe
            .names()
            .into_iter()
            .map(str::to_string)
            .collect::<BTreeSet<_>>();
        for missing in used.difference(&names) {
            failures.push(format!(
                "{} missing tensor {missing}",
                loaded_suite.suite.name
            ));
        }
        for orphan in names.difference(&used) {
            failures.push(format!(
                "{} orphan tensor {orphan}",
                loaded_suite.suite.name
            ));
        }
    }
    for (case_id, control) in controls {
        if !success_ids.contains(&control) {
            failures.push(format!(
                "{case_id} control case {control} is not a success case"
            ));
        }
    }
    failures.sort();
    failures
}

fn parse_hex_u32(value: Option<&str>) -> Result<u32, String> {
    let value = value.ok_or_else(|| "missing bit payload".to_string())?;
    let digits = value
        .strip_prefix("0x")
        .ok_or_else(|| format!("invalid bit payload {value}"))?;
    if digits.len() != 8 {
        return Err(format!("invalid bit payload {value}"));
    }
    u32::from_str_radix(digits, 16).map_err(|_| format!("invalid bit payload {value}"))
}

pub(super) struct Args<'a> {
    case: &'a Case,
    safe: &'a SafeTensors<'a>,
    consumed: Vec<bool>,
}

impl<'a> Args<'a> {
    pub(super) fn new(case: &'a Case, safe: &'a SafeTensors<'a>) -> Self {
        Self {
            case,
            safe,
            consumed: vec![false; case.args.len()],
        }
    }

    pub(super) fn take(&mut self, name: &str) -> Result<&'a Arg, String> {
        let index = self
            .case
            .args
            .iter()
            .position(|arg| arg.name() == name)
            .ok_or_else(|| format!("missing argument {name}"))?;
        if self.consumed[index] {
            return Err(format!("argument {name} consumed twice"));
        }
        self.consumed[index] = true;
        Ok(&self.case.args[index])
    }

    pub(super) fn tensor(&mut self, name: &str) -> Result<Array, String> {
        match self.take(name)? {
            Arg::Tensor {
                tensor_ref,
                encoding,
                imag_ref,
                ..
            } => decode_ref(
                self.safe,
                tensor_ref,
                encoding.as_deref(),
                imag_ref.as_deref(),
            )
            .map(|tensor| tensor.to_array()),
            _ => Err(format!("argument {name} is not a tensor")),
        }
    }

    pub(super) fn scalar(&mut self, name: &str) -> Result<ScalarValue, String> {
        match self.take(name)? {
            Arg::Scalar {
                scalar_type,
                value,
                bits,
                real_bits,
                imag_bits,
                ..
            } => match scalar_type.as_str() {
                "bool" => value
                    .as_ref()
                    .and_then(serde_json::Value::as_bool)
                    .map(ScalarValue::Bool)
                    .ok_or_else(|| format!("invalid bool scalar {name}")),
                "i32" => value
                    .as_ref()
                    .and_then(serde_json::Value::as_i64)
                    .and_then(|v| i32::try_from(v).ok())
                    .map(ScalarValue::I32)
                    .ok_or_else(|| format!("invalid i32 scalar {name}")),
                "f32" => Ok(ScalarValue::F32(f32::from_bits(parse_hex_u32(
                    bits.as_deref(),
                )?))),
                "complex64" => Ok(ScalarValue::C64(Complex32::new(
                    f32::from_bits(parse_hex_u32(real_bits.as_deref())?),
                    f32::from_bits(parse_hex_u32(imag_bits.as_deref())?),
                ))),
                _ => Err(format!("unknown scalar type {scalar_type}")),
            },
            _ => Err(format!("argument {name} is not a scalar")),
        }
    }

    pub(super) fn axes(&mut self, name: &str) -> Result<Vec<i32>, String> {
        match self.take(name)? {
            Arg::Axes { values, .. } => Ok(values.clone()),
            _ => Err(format!("argument {name} is not axes")),
        }
    }

    pub(super) fn axis(&mut self, name: &str) -> Result<i32, String> {
        match self.take(name)? {
            Arg::Axis { value, .. } => Ok(*value),
            _ => Err(format!("argument {name} is not an axis")),
        }
    }

    pub(super) fn optional_axis(&mut self, name: &str) -> Result<Option<i32>, String> {
        match self.take(name)? {
            Arg::OptionalAxis { value, .. } => Ok(*value),
            _ => Err(format!("argument {name} is not an optional axis")),
        }
    }

    pub(super) fn shape(&mut self, name: &str) -> Result<Vec<i32>, String> {
        match self.take(name)? {
            Arg::Shape { values, .. } => Ok(values.clone()),
            _ => Err(format!("argument {name} is not a shape")),
        }
    }

    pub(super) fn optional_bool(&mut self, name: &str) -> Result<Option<bool>, String> {
        match self.take(name)? {
            Arg::OptionalBool { value, .. } => Ok(*value),
            _ => Err(format!("argument {name} is not an optional bool")),
        }
    }

    pub(super) fn execution(&mut self) -> Result<ExecutionTarget, String> {
        match self.take("execution")? {
            Arg::Execution { target, .. } => Ok(*target),
            _ => Err("execution is not an execution argument".into()),
        }
    }

    pub(super) fn finish(self) -> Result<(), String> {
        let unconsumed = self
            .case
            .args
            .iter()
            .zip(self.consumed)
            .filter_map(|(arg, consumed)| (!consumed).then_some(arg.name()))
            .collect::<Vec<_>>();
        if unconsumed.is_empty() {
            Ok(())
        } else {
            Err(format!("unconsumed arguments: {}", unconsumed.join(", ")))
        }
    }
}

pub(super) enum ScalarValue {
    Bool(bool),
    I32(i32),
    F32(f32),
    C64(Complex32),
}

pub(super) fn mlx_error<T>(result: mlx_rs::error::Result<T>) -> Result<T, String> {
    result.map_err(|error| error.to_string())
}

fn logical_offsets(shape: &[i32], strides: &[usize], count: usize) -> Result<Vec<usize>, String> {
    if shape.len() != strides.len() {
        return Err(format!(
            "shape rank {} does not match stride rank {}",
            shape.len(),
            strides.len()
        ));
    }
    let dimensions = shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension).map_err(|_| format!("negative dimension {dimension}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let logical_count = dimensions.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| "logical element count overflow".to_string())
    })?;
    if logical_count != count {
        return Err(format!(
            "shape contains {logical_count} elements but array size is {count}"
        ));
    }
    let max_linear_offset =
        dimensions
            .iter()
            .zip(strides)
            .try_fold(0usize, |offset, (&dimension, &stride)| {
                let contribution = dimension
                    .saturating_sub(1)
                    .checked_mul(stride)
                    .ok_or_else(|| "maximum linear offset overflow".to_string())?;
                offset
                    .checked_add(contribution)
                    .ok_or_else(|| "maximum linear offset overflow".to_string())
            })?;
    (0..count)
        .map(|mut logical_offset| {
            let mut linear_offset = 0usize;
            for (&dimension, &stride) in dimensions.iter().zip(strides).rev() {
                let index = logical_offset % dimension;
                logical_offset /= dimension;
                linear_offset = linear_offset
                    .checked_add(index.checked_mul(stride).ok_or_else(|| {
                        "logical linear offset multiplication overflow".to_string()
                    })?)
                    .ok_or_else(|| "logical linear offset overflow".to_string())?;
            }
            if linear_offset > max_linear_offset {
                return Err("logical linear offset exceeds maximum".to_string());
            }
            Ok(linear_offset)
        })
        .collect()
}

fn observe(array: &Array) -> Result<HostTensor, String> {
    let dtype = array.dtype();
    let shape = array.shape().to_vec();
    let count = array.size();
    let empty = || match dtype {
        Dtype::Bool => TensorData::Bool(Vec::new()),
        Dtype::Uint8 => TensorData::U8(Vec::new()),
        Dtype::Uint16 => TensorData::U16(Vec::new()),
        Dtype::Uint32 => TensorData::U32(Vec::new()),
        Dtype::Uint64 => TensorData::U64(Vec::new()),
        Dtype::Int8 => TensorData::I8(Vec::new()),
        Dtype::Int16 => TensorData::I16(Vec::new()),
        Dtype::Int32 => TensorData::I32(Vec::new()),
        Dtype::Int64 => TensorData::I64(Vec::new()),
        Dtype::Float16 => TensorData::F16(Vec::new()),
        Dtype::Bfloat16 => TensorData::BF16(Vec::new()),
        Dtype::Float32 => TensorData::F32(Vec::new()),
        Dtype::Float64 => TensorData::F64(Vec::new()),
        Dtype::Complex64 => TensorData::C64(Vec::new()),
    };
    let data = if count == 0 {
        empty()
    } else {
        let offsets = logical_offsets(&shape, array.strides(), count)?;
        macro_rules! read_data {
            ($accessor:path, $convert:expr) => {{
                let pointer = $accessor(array.as_ptr());
                if pointer.is_null() {
                    return Err(format!("null data pointer while observing {dtype:?}"));
                }
                offsets
                    .iter()
                    .map(|&offset| $convert(*pointer.add(offset)))
                    .collect()
            }};
        }
        // MLX data pointers are offset-adjusted but stride-blind, and reshape may return strided views, so no operation-level materialization is trustworthy.
        unsafe {
            match dtype {
                Dtype::Bool => TensorData::Bool(read_data!(mlx_sys::mlx_array_data_bool, u8::from)),
                Dtype::Uint8 => {
                    TensorData::U8(read_data!(mlx_sys::mlx_array_data_uint8, |value| value))
                }
                Dtype::Uint16 => {
                    TensorData::U16(read_data!(mlx_sys::mlx_array_data_uint16, |value| value))
                }
                Dtype::Uint32 => {
                    TensorData::U32(read_data!(mlx_sys::mlx_array_data_uint32, |value| value))
                }
                Dtype::Uint64 => {
                    TensorData::U64(read_data!(mlx_sys::mlx_array_data_uint64, |value| value))
                }
                Dtype::Int8 => {
                    TensorData::I8(read_data!(mlx_sys::mlx_array_data_int8, |value| value))
                }
                Dtype::Int16 => {
                    TensorData::I16(read_data!(mlx_sys::mlx_array_data_int16, |value| value))
                }
                Dtype::Int32 => {
                    TensorData::I32(read_data!(mlx_sys::mlx_array_data_int32, |value| value))
                }
                Dtype::Int64 => {
                    TensorData::I64(read_data!(mlx_sys::mlx_array_data_int64, |value| value))
                }
                Dtype::Float16 => TensorData::F16(read_data!(
                    mlx_sys::mlx_array_data_float16,
                    |value: mlx_sys::float16_t| f16::from_bits(value.0)
                )),
                Dtype::Bfloat16 => TensorData::BF16(read_data!(
                    mlx_sys::mlx_array_data_bfloat16,
                    bf16::from_bits
                )),
                Dtype::Float32 => {
                    TensorData::F32(read_data!(mlx_sys::mlx_array_data_float32, |value| value))
                }
                Dtype::Float64 => {
                    TensorData::F64(read_data!(mlx_sys::mlx_array_data_float64, |value| value))
                }
                Dtype::Complex64 => TensorData::C64(read_data!(
                    mlx_sys::mlx_array_data_complex64,
                    |value: mlx_sys::mlx_complex64_t| Complex32::new(value.re, value.im)
                )),
            }
        }
    };
    Ok(HostTensor { dtype, shape, data })
}

#[derive(Debug)]
struct Mismatch {
    class: &'static str,
    detail: String,
}

fn compare_float(
    expected: f64,
    got: f64,
    expected_bits: u64,
    got_bits: u64,
    policy: &Policy,
) -> Result<f64, Mismatch> {
    let Policy::Float {
        atol,
        rtol,
        nan_equal,
        infinity_sign,
        signed_zero,
        complex,
    } = policy
    else {
        return Err(Mismatch {
            class: "policy",
            detail: "float output used exact_numeric".into(),
        });
    };
    if *complex != ComplexRule::Componentwise {
        return Err(Mismatch {
            class: "policy",
            detail: "unsupported complex rule".into(),
        });
    }
    if expected.is_nan() || got.is_nan() {
        return if expected.is_nan() && got.is_nan() && *nan_equal {
            Ok(0.0)
        } else {
            Err(Mismatch {
                class: "nan",
                detail: format!("expected {expected:?}, got {got:?}"),
            })
        };
    }
    if expected.is_infinite() || got.is_infinite() {
        return if expected == got || (!infinity_sign && expected.is_infinite() && got.is_infinite())
        {
            Ok(0.0)
        } else {
            Err(Mismatch {
                class: "infinity_sign",
                detail: format!("expected {expected:?}, got {got:?}"),
            })
        };
    }
    if expected == 0.0
        && got == 0.0
        && *signed_zero
        && expected.is_sign_negative() != got.is_sign_negative()
    {
        return Err(Mismatch {
            class: "signed_zero",
            detail: format!("expected bits 0x{expected_bits:x}, got 0x{got_bits:x}"),
        });
    }
    let error = (expected - got).abs();
    let limit = *atol + *rtol * expected.abs();
    if error > limit {
        let class = if *atol > 0.0 && expected == 0.0 {
            "value_absolute"
        } else if *rtol > 0.0 && expected != 0.0 {
            "value_relative"
        } else {
            "value"
        };
        Err(Mismatch { class, detail: format!("expected {expected:?} (0x{expected_bits:x}), got {got:?} (0x{got_bits:x}), error {error:e}, limit {limit:e}") })
    } else {
        Ok(error)
    }
}

fn compare_float_sequence(
    values: impl IntoIterator<Item = (String, f64, f64, u64, u64)>,
    policy: &Policy,
    output_name: &str,
) -> Result<(), Mismatch> {
    let mut first = None;
    let mut max_error = 0.0f64;
    for (location, expected, got, expected_bits, got_bits) in values {
        match compare_float(expected, got, expected_bits, got_bits, policy) {
            Ok(error) => max_error = max_error.max(error),
            Err(mut error) => {
                if expected.is_finite() && got.is_finite() {
                    max_error = max_error.max((expected - got).abs());
                }
                if first.is_none() {
                    error.detail = format!(
                        "{output_name}: first bad {location}, {}; max observed error pending",
                        error.detail
                    );
                    first = Some(error);
                }
            }
        }
    }
    if let Some(mut error) = first {
        error.detail = error.detail.replace(
            "max observed error pending",
            &format!("max observed error {max_error:e}"),
        );
        Err(error)
    } else {
        Ok(())
    }
}

fn compare_tensor(
    expected: &HostTensor,
    got: &HostTensor,
    policy: &Policy,
    output_name: &str,
) -> Result<(), Mismatch> {
    if expected.dtype != got.dtype {
        return Err(Mismatch {
            class: "dtype",
            detail: format!(
                "{output_name}: expected {:?}, got {:?}",
                expected.dtype, got.dtype
            ),
        });
    }
    if expected.shape != got.shape {
        return Err(Mismatch {
            class: "shape",
            detail: format!(
                "{output_name}: expected {:?}, got {:?}",
                expected.shape, got.shape
            ),
        });
    }
    if expected.len() != got.len() {
        return Err(Mismatch {
            class: "size",
            detail: format!(
                "{output_name}: expected {}, got {}",
                expected.len(),
                got.len()
            ),
        });
    }
    macro_rules! exact {
        ($a:expr,$b:expr) => {
            if let Some((index, (a, b))) = $a.iter().zip($b).enumerate().find(|(_, (a, b))| a != b)
            {
                return Err(Mismatch {
                    class: "value",
                    detail: format!(
                        "{output_name}: first bad element {index}, expected {a:?}, got {b:?}"
                    ),
                });
            }
        };
    }
    match (&expected.data, &got.data) {
        (TensorData::Bool(a), TensorData::Bool(b)) | (TensorData::U8(a), TensorData::U8(b)) => {
            exact!(a, b)
        }
        (TensorData::U16(a), TensorData::U16(b)) => exact!(a, b),
        (TensorData::U32(a), TensorData::U32(b)) => exact!(a, b),
        (TensorData::U64(a), TensorData::U64(b)) => exact!(a, b),
        (TensorData::I8(a), TensorData::I8(b)) => exact!(a, b),
        (TensorData::I16(a), TensorData::I16(b)) => exact!(a, b),
        (TensorData::I32(a), TensorData::I32(b)) => exact!(a, b),
        (TensorData::I64(a), TensorData::I64(b)) => exact!(a, b),
        (TensorData::F16(a), TensorData::F16(b)) => {
            compare_float_sequence(
                a.iter().zip(b).enumerate().map(|(index, (a, b))| {
                    (
                        format!("element {index}"),
                        a.to_f64(),
                        b.to_f64(),
                        a.to_bits() as u64,
                        b.to_bits() as u64,
                    )
                }),
                policy,
                output_name,
            )?;
        }
        (TensorData::BF16(a), TensorData::BF16(b)) => {
            compare_float_sequence(
                a.iter().zip(b).enumerate().map(|(index, (a, b))| {
                    (
                        format!("element {index}"),
                        a.to_f64(),
                        b.to_f64(),
                        a.to_bits() as u64,
                        b.to_bits() as u64,
                    )
                }),
                policy,
                output_name,
            )?;
        }
        (TensorData::F32(a), TensorData::F32(b)) => {
            compare_float_sequence(
                a.iter().zip(b).enumerate().map(|(index, (a, b))| {
                    (
                        format!("element {index}"),
                        *a as f64,
                        *b as f64,
                        a.to_bits() as u64,
                        b.to_bits() as u64,
                    )
                }),
                policy,
                output_name,
            )?;
        }
        (TensorData::F64(a), TensorData::F64(b)) => {
            compare_float_sequence(
                a.iter().zip(b).enumerate().map(|(index, (a, b))| {
                    (format!("element {index}"), *a, *b, a.to_bits(), b.to_bits())
                }),
                policy,
                output_name,
            )?;
        }
        (TensorData::C64(a), TensorData::C64(b)) => {
            compare_float_sequence(
                a.iter().zip(b).enumerate().flat_map(|(index, (a, b))| {
                    [("real", a.re, b.re), ("imag", a.im, b.im)].map(
                        move |(component, expected, got)| {
                            (
                                format!("element {index} {component}"),
                                expected as f64,
                                got as f64,
                                expected.to_bits() as u64,
                                got.to_bits() as u64,
                            )
                        },
                    )
                }),
                policy,
                output_name,
            )?;
        }
        _ => {
            return Err(Mismatch {
                class: "dtype",
                detail: format!("{output_name}: host representation mismatch"),
            })
        }
    }
    Ok(())
}

enum OperationFailure {
    Invoke(String),
    Eval(String),
    Observe(String),
    Panic,
}

fn invoke_case(case: &Case, safe: &SafeTensors<'_>) -> Result<Vec<HostTensor>, OperationFailure> {
    let invoked = catch_unwind(AssertUnwindSafe(|| dispatch(case, safe)))
        .map_err(|_| OperationFailure::Panic)?;
    let outputs = invoked.map_err(OperationFailure::Invoke)?;
    for output in &outputs {
        output
            .eval()
            .map_err(|error| OperationFailure::Eval(error.to_string()))?;
    }
    outputs
        .iter()
        .map(observe)
        .collect::<Result<Vec<_>, _>>()
        .map_err(OperationFailure::Observe)
}

fn run_case(
    case: &Case,
    safe: &SafeTensors<'_>,
    policies: &BTreeMap<String, Policy>,
) -> Vec<String> {
    let result = invoke_case(case, safe);
    match (&case.expected, result) {
        (
            Expected::Error {
                allowed_stage: AllowedStage::InvokeOrEval,
                ..
            },
            Err(OperationFailure::Invoke(_)),
        ) => Vec::new(),
        (
            Expected::Error {
                allowed_stage: AllowedStage::InvokeOrEval | AllowedStage::EvalOnly,
                ..
            },
            Err(OperationFailure::Eval(_)),
        ) => Vec::new(),
        (Expected::Error { .. }, Err(OperationFailure::Panic)) => vec![format!(
            "{}: panic never satisfies expected operation error",
            case.id
        )],
        (Expected::Error { .. }, Err(OperationFailure::Observe(error))) => vec![format!(
            "{}: observation failure never satisfies expected operation error: {error}",
            case.id
        )],
        (
            Expected::Error {
                allowed_stage: AllowedStage::EvalOnly,
                ..
            },
            Err(OperationFailure::Invoke(error)),
        ) => vec![format!(
            "{}: error_stage: expected eval error, invocation failed: {error}",
            case.id
        )],
        (Expected::Error { .. }, Ok(_)) => {
            vec![format!("{}: expected_error: operation succeeded", case.id)]
        }
        (Expected::Success { .. }, Err(OperationFailure::Invoke(error))) => {
            vec![format!("{}: invocation failure: {error}", case.id)]
        }
        (Expected::Success { .. }, Err(OperationFailure::Eval(error))) => {
            vec![format!("{}: evaluation failure: {error}", case.id)]
        }
        (Expected::Success { .. }, Err(OperationFailure::Observe(error))) => {
            vec![format!("{}: observation failure: {error}", case.id)]
        }
        (Expected::Success { .. }, Err(OperationFailure::Panic)) => {
            vec![format!("{}: adapter panicked", case.id)]
        }
        (Expected::Success { outputs, .. }, Ok(got)) => {
            if outputs.len() != got.len() {
                return vec![format!(
                    "{}: output_count: expected {}, got {}",
                    case.id,
                    outputs.len(),
                    got.len()
                )];
            }
            let mut failures = Vec::new();
            for (index, (output, got)) in outputs.iter().zip(&got).enumerate() {
                let expected = match decode_ref(
                    safe,
                    &output.tensor_ref,
                    output.encoding.as_deref(),
                    output.imag_ref.as_deref(),
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        failures.push(format!(
                            "{}: output {} fixture decode: {error}",
                            case.id, output.name
                        ));
                        continue;
                    }
                };
                let policy = &policies[&output.policy];
                if let Err(error) = compare_tensor(&expected, got, policy, &output.name) {
                    failures.push(format!(
                        "{}: output {index}: {}: {}",
                        case.id, error.class, error.detail
                    ));
                }
            }
            failures
        }
    }
}

fn with_cpu_defaults<T>(f: impl FnOnce() -> T) -> T {
    let previous = Device::try_default().expect("read default device");
    let _guard = DeviceGuard(previous);
    Device::set_default(&Device::cpu());
    with_stream(&Stream::cpu(), f)
}

fn assert_failures(mut failures: Vec<String>) {
    failures.sort();
    if !failures.is_empty() {
        panic!("{}", failures.join("\n"));
    }
}

pub(super) fn committed_corpus() {
    let loaded = match load_corpus() {
        Ok(value) => value,
        Err(failures) => return assert_failures(failures),
    };
    assert_failures(preflight(&loaded));
    with_cpu_defaults(|| {
        let mut failures = Vec::new();
        for loaded_suite in &loaded.suites {
            let safe =
                SafeTensors::deserialize(&loaded_suite.bytes).expect("preflight decoded fixture");
            for case in &loaded_suite.suite.cases {
                failures.extend(run_case(case, &safe, &loaded.corpus.tolerance_policies));
            }
        }
        assert_failures(failures);
    });
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GgufSuite {
    schema_version: u32,
    name: String,
    fixture: String,
    cases: Vec<GgufCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GgufCase {
    pub(super) id: String,
    pub(super) rust_call: String,
    pub(super) recipe: GgufRecipe,
    pub(super) expected: GgufExpected,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum GgufRecipe {
    Load {
        path: String,
        execution: ExecutionTarget,
        #[serde(default)]
        dequantize: Option<GgufDequantize>,
    },
    Absence {
        path: String,
        array_key: String,
        metadata_key: String,
    },
    WrongKind {
        path: String,
        key: String,
        requested: GgufKind,
    },
    TensorRejects {
        accepted: Vec<String>,
        dtypes: Vec<String>,
    },
    MetadataRejects {
        accepted: Vec<String>,
        dtypes: Vec<String>,
        ranks: Vec<usize>,
        empty: bool,
    },
    ConstructSave {
        path: String,
        same_spelling: String,
        metadata_value: String,
        non_contiguous_shape: Vec<i32>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GgufDequantize {
    pub(super) group_size: i32,
    pub(super) bits: i32,
}

#[derive(Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum GgufKind {
    Array,
    String,
    Strings,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GgufExpected {
    pub(super) status: String,
    #[serde(default)]
    pub(super) variant: Option<String>,
    #[serde(default)]
    pub(super) variants: Vec<String>,
    #[serde(default)]
    pub(super) expected_kind: Option<GgufKind>,
    #[serde(default)]
    pub(super) actual_kind: Option<GgufKind>,
    #[serde(default)]
    pub(super) array_keys: Option<Vec<String>>,
    #[serde(default)]
    pub(super) arrays: Vec<GgufExpectedArray>,
    #[serde(default)]
    pub(super) metadata: Vec<GgufExpectedMetadata>,
    #[serde(default)]
    pub(super) array_absent: Option<bool>,
    #[serde(default)]
    pub(super) metadata_absent: Option<bool>,
    #[serde(default)]
    pub(super) dequantized: Option<GgufExpectedArray>,
    #[serde(default)]
    pub(super) duplicate_array_variant: Option<String>,
    #[serde(default)]
    pub(super) duplicate_metadata_variant: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GgufExpectedArray {
    pub(super) key: String,
    #[serde(rename = "ref")]
    pub(super) tensor_ref: String,
    pub(super) dtype: String,
    pub(super) shape: Vec<i32>,
    pub(super) policy: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GgufExpectedMetadata {
    pub(super) key: String,
    pub(super) kind: GgufKind,
    #[serde(default)]
    pub(super) value: Option<serde_json::Value>,
    #[serde(default, rename = "ref")]
    pub(super) tensor_ref: Option<String>,
    #[serde(default)]
    pub(super) dtype: Option<String>,
    #[serde(default)]
    pub(super) shape: Option<Vec<i32>>,
    #[serde(default)]
    pub(super) policy: Option<String>,
}

pub(super) enum GgufObservedMetadata {
    Array(Array),
    String(String),
    Strings(Vec<String>),
}

pub(super) struct GgufObservation {
    pub(super) array_keys: Vec<String>,
    pub(super) arrays: BTreeMap<String, Array>,
    pub(super) metadata: BTreeMap<String, GgufObservedMetadata>,
    pub(super) array_absent: Option<bool>,
    pub(super) metadata_absent: Option<bool>,
    pub(super) errors: Vec<String>,
    pub(super) error_kinds: Vec<(GgufKind, GgufKind)>,
    pub(super) dequantized: Option<Array>,
}

fn compare_gguf_array(
    case_id: &str,
    expected: &GgufExpectedArray,
    got: &Array,
    safe: &SafeTensors<'_>,
    policies: &BTreeMap<String, Policy>,
) -> Result<(), String> {
    let fixture = decode_tensor(safe, &expected.tensor_ref)?;
    if fixture.dtype != dtype_from_name(&expected.dtype)? || fixture.shape != expected.shape {
        return Err(format!(
            "{case_id}: GGUF fixture declaration mismatch for {}",
            expected.key
        ));
    }
    got.eval().map_err(|error| error.to_string())?;
    let observed = observe(got)?;
    compare_tensor(
        &fixture,
        &observed,
        &policies[&expected.policy],
        &expected.key,
    )
    .map_err(|error| format!("{case_id}: {}: {}", error.class, error.detail))
}

fn run_gguf_case(
    root: &Path,
    case: &GgufCase,
    safe: &SafeTensors<'_>,
    policies: &BTreeMap<String, Policy>,
) -> Vec<String> {
    let observed = match super::adapters::dispatch_gguf(root, case) {
        Ok(value) => value,
        Err(error) => return vec![format!("{}: {error}", case.id)],
    };
    compare_gguf_observation(case, &observed, safe, policies)
}

fn compare_gguf_observation(
    case: &GgufCase,
    observed: &GgufObservation,
    safe: &SafeTensors<'_>,
    policies: &BTreeMap<String, Policy>,
) -> Vec<String> {
    let expected = &case.expected;
    let mut failures = Vec::new();
    if expected.status == "error" {
        if let Some(variant) = &expected.variant {
            if !observed.errors.contains(variant) {
                failures.push(format!(
                    "{}: error_variant: expected {variant:?}, got {:?}",
                    case.id, observed.errors
                ));
            }
        }
        for variant in &expected.variants {
            if !observed.errors.contains(variant) {
                failures.push(format!("{}: error_variant: missing {variant}", case.id));
            }
        }
        if let (Some(expected_kind), Some(actual_kind)) =
            (expected.expected_kind, expected.actual_kind)
        {
            if !observed.error_kinds.contains(&(expected_kind, actual_kind)) {
                failures.push(format!(
                    "{}: wrong_kind_fields: expected wrong-kind fields did not match",
                    case.id
                ));
            }
        }
        return failures;
    }
    if let Some(array_keys) = &expected.array_keys {
        if &observed.array_keys != array_keys {
            failures.push(format!("{}: array_keys: keys differ", case.id));
        }
    }
    for item in &expected.arrays {
        match observed.arrays.get(&item.key) {
            Some(array) => {
                if let Err(error) = compare_gguf_array(&case.id, item, array, safe, policies) {
                    failures.push(error);
                }
            }
            None => failures.push(format!(
                "{}: array_missing: missing array {}",
                case.id, item.key
            )),
        }
    }
    for item in &expected.metadata {
        let got = observed.metadata.get(&item.key);
        match (item.kind, got) {
            (GgufKind::String, Some(GgufObservedMetadata::String(value)))
                if item.value.as_ref().and_then(serde_json::Value::as_str) == Some(value) => {}
            (GgufKind::String, Some(GgufObservedMetadata::String(_))) => failures.push(format!(
                "{}: metadata_value: metadata {} differs",
                case.id, item.key
            )),
            (GgufKind::Strings, Some(GgufObservedMetadata::Strings(values))) => {
                let expected_values = item
                    .value
                    .as_ref()
                    .and_then(serde_json::Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(serde_json::Value::as_str)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if expected_values != values.iter().map(String::as_str).collect::<Vec<_>>() {
                    failures.push(format!(
                        "{}: metadata_value: metadata {} differs",
                        case.id, item.key
                    ));
                }
            }
            (GgufKind::Array, Some(GgufObservedMetadata::Array(array))) => {
                let declaration = GgufExpectedArray {
                    key: item.key.clone(),
                    tensor_ref: item.tensor_ref.clone().unwrap_or_default(),
                    dtype: item.dtype.clone().unwrap_or_default(),
                    shape: item.shape.clone().unwrap_or_default(),
                    policy: item.policy.clone().unwrap_or_default(),
                };
                if let Err(error) =
                    compare_gguf_array(&case.id, &declaration, array, safe, policies)
                {
                    failures.push(error);
                }
            }
            (_, None) => failures.push(format!(
                "{}: metadata_missing: metadata {} is missing",
                case.id, item.key
            )),
            _ => failures.push(format!(
                "{}: metadata_kind: metadata {} kind differs",
                case.id, item.key
            )),
        }
    }
    if observed.array_absent != expected.array_absent {
        failures.push(format!("{}: array_absence: result differs", case.id));
    }
    if observed.metadata_absent != expected.metadata_absent {
        failures.push(format!("{}: metadata_absence: result differs", case.id));
    }
    for expected_variant in [
        expected.duplicate_array_variant.as_ref(),
        expected.duplicate_metadata_variant.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if !observed.errors.contains(expected_variant) {
            failures.push(format!(
                "{}: duplicate_variant: missing {expected_variant}",
                case.id
            ));
        }
    }
    match (&expected.dequantized, &observed.dequantized) {
        (Some(declaration), Some(array)) => {
            if let Err(error) = compare_gguf_array(&case.id, declaration, array, safe, policies) {
                failures.push(error);
            }
        }
        (Some(_), None) => failures.push(format!(
            "{}: dequantized_missing: observation is missing",
            case.id
        )),
        (None, Some(_)) => failures.push(format!(
            "{}: dequantized_unexpected: observation is unexpected",
            case.id
        )),
        (None, None) => {}
    }
    failures
}

pub(super) fn gguf_committed_corpus() {
    let root = corpus_root();
    let suite: GgufSuite = match read_json(&root.join("suites/gguf.json")) {
        Ok(value) => value,
        Err(error) => return assert_failures(vec![error]),
    };
    if suite.schema_version != 1
        || suite.name != "gguf"
        || suite.fixture != "fixtures/gguf.safetensors"
    {
        return assert_failures(vec!["invalid GGUF suite identity".into()]);
    }
    let bytes = fs::read(root.join(&suite.fixture)).expect("read GGUF safetensors fixture");
    let safe = SafeTensors::deserialize(&bytes).expect("decode GGUF safetensors fixture");
    let corpus: Corpus = read_json(&root.join("corpus.json")).expect("read corpus");
    let _ = &corpus.gguf_fixtures;
    with_cpu_defaults(|| {
        let failures = suite
            .cases
            .iter()
            .flat_map(|case| run_gguf_case(&root, case, &safe, &corpus.tolerance_policies))
            .collect();
        assert_failures(failures);
    });
}

fn gguf_failure_class(case_id: &str, failures: Vec<String>) -> Result<String, String> {
    let [failure] = failures.as_slice() else {
        return Err(format!(
            "GGUF mutation produced {} failures instead of one: {failures:?}",
            failures.len()
        ));
    };
    let prefix = format!("{case_id}: ");
    let detail = failure
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("GGUF failure did not start with {prefix:?}: {failure}"))?;
    let (class, _) = detail
        .split_once(':')
        .ok_or_else(|| format!("GGUF failure did not identify a class: {failure}"))?;
    Ok(class.into())
}

fn perturb_gguf_array(array: &Array) -> Result<Array, String> {
    let delta = Array::from_f32(1.0)
        .as_dtype(array.dtype())
        .map_err(|error| error.to_string())?;
    ops::add(array, delta).map_err(|error| error.to_string())
}

fn qualify_gguf_mutation(
    mutation: &Mutation,
    base: &GgufCase,
    safe: &SafeTensors<'_>,
    mut observed: GgufObservation,
    policies: &BTreeMap<String, Policy>,
) -> Result<String, String> {
    match mutation.kind.as_str() {
        "gguf_array_dtype_changed_values_equal" => {
            let array = observed
                .arrays
                .get("tensor.f32")
                .ok_or("GGUF dtype mutation base is missing tensor.f32")?;
            let converted = array
                .as_dtype(Dtype::Float64)
                .map_err(|error| error.to_string())?;
            observed.arrays.insert("tensor.f32".into(), converted);
        }
        "gguf_array_beyond_tolerance" => {
            let array = observed
                .arrays
                .get("tensor.f32")
                .ok_or("GGUF value mutation base is missing tensor.f32")?;
            let perturbed = perturb_gguf_array(array)?;
            observed.arrays.insert("tensor.f32".into(), perturbed);
        }
        "gguf_array_key_removed" => {
            let original_len = observed.array_keys.len();
            observed.array_keys.retain(|key| key != "tensor.f32");
            if observed.array_keys.len() == original_len {
                return Err("GGUF key mutation base is missing tensor.f32".into());
            }
        }
        "gguf_metadata_kind_swapped" => {
            let Some(GgufObservedMetadata::String(value)) = observed.metadata.remove("text") else {
                return Err("GGUF metadata kind mutation base is missing string text".into());
            };
            observed
                .metadata
                .insert("text".into(), GgufObservedMetadata::Strings(vec![value]));
        }
        "gguf_metadata_entry_missing" => {
            if observed.metadata.remove("text").is_none() {
                return Err("GGUF metadata missing mutation base has no text entry".into());
            }
        }
        "gguf_error_variant_mismatch" => {
            if observed.errors.is_empty() {
                return Err("GGUF error variant mutation base has no error".into());
            }
            observed.errors.clear();
        }
        "gguf_wrong_kind_fields" => {
            if observed.error_kinds.is_empty() {
                return Err("GGUF wrong-kind mutation base has no kind fields".into());
            }
            observed.error_kinds.clear();
        }
        "gguf_dequantized_observation_dropped" => {
            if observed.dequantized.take().is_none() {
                return Err("GGUF dequantized-drop mutation base has no observation".into());
            }
        }
        "gguf_dequantized_beyond_tolerance" => {
            let dequantized = observed
                .dequantized
                .take()
                .ok_or("GGUF dequantized value mutation base has no observation")?;
            observed.dequantized = Some(perturb_gguf_array(&dequantized)?);
        }
        other => return Err(format!("unknown GGUF mutation kind {other}")),
    }
    gguf_failure_class(
        &base.id,
        compare_gguf_observation(base, &observed, safe, policies),
    )
}

fn qualification_results(loaded: &LoadedCorpus, qualification: &Qualification) -> Vec<String> {
    let mut failures = Vec::new();
    if qualification.schema_version != 1 {
        failures.push("qualification schema_version is unsupported".into());
    }
    let cases = loaded
        .suites
        .iter()
        .flat_map(|suite| {
            suite
                .suite
                .cases
                .iter()
                .map(move |case| (case.id.as_str(), (case, &suite.bytes)))
        })
        .collect::<BTreeMap<_, _>>();
    let gguf_suite: GgufSuite = match read_json(&loaded.root.join("suites/gguf.json")) {
        Ok(value) => value,
        Err(error) => return vec![error],
    };
    if gguf_suite.schema_version != 1
        || gguf_suite.name != "gguf"
        || gguf_suite.fixture != "fixtures/gguf.safetensors"
    {
        return vec!["invalid GGUF qualification suite identity".into()];
    }
    let gguf_bytes = match fs::read(loaded.root.join(&gguf_suite.fixture)) {
        Ok(value) => value,
        Err(error) => return vec![error.to_string()],
    };
    let gguf_safe = match SafeTensors::deserialize(&gguf_bytes) {
        Ok(value) => value,
        Err(error) => return vec![error.to_string()],
    };
    let gguf_cases = gguf_suite
        .cases
        .iter()
        .map(|case| (case.id.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    let mut ids = BTreeSet::new();
    for mutation in &qualification.mutations {
        if !ids.insert(&mutation.id) {
            failures.push(format!("duplicate mutation {}", mutation.id));
            continue;
        }
        if mutation.base_case_id.starts_with("gguf.") {
            let Some(base) = gguf_cases.get(mutation.base_case_id.as_str()) else {
                failures.push(format!(
                    "{} missing base case {}",
                    mutation.id, mutation.base_case_id
                ));
                continue;
            };
            let class = match super::adapters::dispatch_gguf(&loaded.root, base) {
                Ok(observed) => qualify_gguf_mutation(
                    mutation,
                    base,
                    &gguf_safe,
                    observed,
                    &loaded.corpus.tolerance_policies,
                ),
                Err(_) => Err("GGUF qualification base did not execute successfully".into()),
            };
            match class {
                Ok(actual) if actual == mutation.expected_class => {}
                Ok(actual) => failures.push(format!(
                    "{}: expected class {}, got {actual}",
                    mutation.id, mutation.expected_class
                )),
                Err(error) => failures.push(format!("{}: {error}", mutation.id)),
            }
            continue;
        }
        let Some((base, bytes)) = cases.get(mutation.base_case_id.as_str()) else {
            failures.push(format!(
                "{} missing base case {}",
                mutation.id, mutation.base_case_id
            ));
            continue;
        };
        let safe = SafeTensors::deserialize(bytes).expect("preflight decoded fixture");
        let class = if mutation.kind == "error_to_valid" {
            let Expected::Error {
                control_case_id, ..
            } = &base.expected
            else {
                failures.push(format!("{} base is not an error case", mutation.id));
                continue;
            };
            let Some((control, control_bytes)) = cases.get(control_case_id.as_str()) else {
                failures.push(format!("{} control case is missing", mutation.id));
                continue;
            };
            let control_safe =
                SafeTensors::deserialize(control_bytes).expect("preflight decoded control fixture");
            match invoke_case(control, &control_safe) {
                Ok(_) => Ok("expected_error".into()),
                Err(_) => Err("valid inversion control did not execute successfully".into()),
            }
        } else {
            match invoke_case(base, &safe) {
                Ok(observed) => qualify_mutation(
                    mutation,
                    base,
                    &safe,
                    &observed,
                    &loaded.corpus.tolerance_policies,
                ),
                Err(_) => Err("passing qualification base did not execute successfully".into()),
            }
        };
        match class {
            Ok(actual) if actual == mutation.expected_class => {}
            Ok(actual) => failures.push(format!(
                "{}: expected class {}, got {actual}",
                mutation.id, mutation.expected_class
            )),
            Err(error) => failures.push(format!("{}: {error}", mutation.id)),
        }
    }
    failures
}

fn first_expected(
    case: &Case,
    safe: &SafeTensors<'_>,
) -> Result<(HostTensor, ExpectedOutput), String> {
    let Expected::Success { outputs, .. } = &case.expected else {
        return Err("base is not a success case".into());
    };
    let output = outputs
        .first()
        .ok_or_else(|| "base has no output".to_string())?
        .clone();
    let tensor = decode_ref(
        safe,
        &output.tensor_ref,
        output.encoding.as_deref(),
        output.imag_ref.as_deref(),
    )?;
    Ok((tensor, output))
}

fn output_layout_class(outputs: &[ExpectedOutput], observed: &[HostTensor]) -> Result<(), String> {
    if outputs.len() != observed.len() {
        return Err("output_count".into());
    }
    if outputs
        .iter()
        .enumerate()
        .any(|(index, output)| output.name != format!("output{index}"))
    {
        return Err("output_order".into());
    }
    Ok(())
}

fn qualify_mutation(
    mutation: &Mutation,
    base: &Case,
    safe: &SafeTensors<'_>,
    observed: &[HostTensor],
    policies: &BTreeMap<String, Policy>,
) -> Result<String, String> {
    let (mut expected, output) = first_expected(base, safe).unwrap_or_else(|_| {
        (
            HostTensor {
                dtype: Dtype::Float32,
                shape: vec![],
                data: TensorData::F32(vec![0.0]),
            },
            ExpectedOutput {
                name: "output0".into(),
                tensor_ref: String::new(),
                dtype: "F32".into(),
                shape: vec![],
                policy: "elementwise_float".into(),
                encoding: None,
                imag_ref: None,
            },
        )
    });
    let result = match mutation.kind.as_str() {
        "dtype_changed_values_equal" => {
            expected.dtype = Dtype::Float64;
            expected.data = match expected.data {
                TensorData::F32(v) => TensorData::F64(v.into_iter().map(f64::from).collect()),
                _ => return Err("dtype base is not F32".into()),
            };
            compare_tensor(
                &expected,
                &observed[0],
                &policies[&output.policy],
                &output.name,
            )
            .unwrap_err()
            .class
        }
        "shape_changed_same_count" => {
            expected.shape = vec![1, expected.len() as i32];
            compare_tensor(
                &expected,
                &observed[0],
                &policies[&output.policy],
                &output.name,
            )
            .unwrap_err()
            .class
        }
        "output_removed" => {
            let Expected::Success { outputs, .. } = &base.expected else {
                return Err("output mutation base is not successful".into());
            };
            let mut mutated = outputs.clone();
            mutated.pop();
            return Ok(output_layout_class(&mutated, observed).unwrap_err());
        }
        "output_added" => {
            let Expected::Success { outputs, .. } = &base.expected else {
                return Err("output mutation base is not successful".into());
            };
            let mut mutated = outputs.clone();
            mutated.push(outputs[0].clone());
            return Ok(output_layout_class(&mutated, observed).unwrap_err());
        }
        "output_reordered" => {
            let Expected::Success { outputs, .. } = &base.expected else {
                return Err("output mutation base is not successful".into());
            };
            let mut mutated = outputs.clone();
            mutated.swap(0, 1);
            return Ok(output_layout_class(&mutated, observed).unwrap_err());
        }
        "beyond_absolute" => {
            compare_float(
                0.0,
                policies[&output.policy].atol() * 2.0 + f64::EPSILON,
                0,
                (policies[&output.policy].atol() * 2.0 + f64::EPSILON).to_bits(),
                &policies[&output.policy],
            )
            .unwrap_err()
            .class
        }
        "beyond_relative" => {
            compare_float(
                1000.0,
                1000.0 + policies[&output.policy].rtol() * 2000.0,
                1000.0f64.to_bits(),
                (1000.0 + policies[&output.policy].rtol() * 2000.0).to_bits(),
                &policies[&output.policy],
            )
            .unwrap_err()
            .class
        }
        "nan_vs_finite" => {
            compare_float(
                f64::NAN,
                1.0,
                f64::NAN.to_bits(),
                1.0f64.to_bits(),
                &policies[&output.policy],
            )
            .unwrap_err()
            .class
        }
        "positive_inf_vs_negative_inf" => {
            compare_float(
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY.to_bits(),
                f64::NEG_INFINITY.to_bits(),
                &policies[&output.policy],
            )
            .unwrap_err()
            .class
        }
        "swapped_subtraction" => {
            let mut args = Args::new(base, safe);
            let lhs = args.tensor("input0")?;
            let rhs = args.tensor("input1")?;
            let swapped = mlx_error(ops::subtract(&rhs, &lhs))?;
            swapped.eval().map_err(|error| error.to_string())?;
            let swapped = observe(&swapped)?;
            let mismatch =
                compare_tensor(&expected, &swapped, &policies[&output.policy], &output.name)
                    .expect_err("swapped subtraction mutation unexpectedly matched");
            if !mismatch.class.starts_with("value") {
                return Err(format!(
                    "swapped subtraction rejected as {}, not value",
                    mismatch.class
                ));
            }
            "value"
        }
        "wrong_axis" => {
            let mut args = Args::new(base, safe);
            let input = args.tensor("input0")?;
            let wrong = mlx_error(ops::sum_axis(&input, 0, false))?;
            wrong.eval().map_err(|error| error.to_string())?;
            let wrong = observe(&wrong)?;
            compare_tensor(&expected, &wrong, &policies[&output.policy], &output.name)
                .expect_err("wrong axis mutation unexpectedly matched")
                .class
        }
        "error_to_valid" => "expected_error",
        "f16_decoder" => match expected.data {
            TensorData::F16(v) => {
                let raw = safe
                    .tensor(&output.tensor_ref)
                    .map_err(|error| error.to_string())?;
                let bits = decode_words(raw.data(), 2, |bytes| {
                    u16::from_le_bytes([bytes[0], bytes[1]])
                });
                if !bits
                    .iter()
                    .zip(&v)
                    .all(|(bits, value)| *bits == value.to_bits())
                {
                    return Err("F16 calibration did not preserve raw bits".into());
                }
                "decoder_f16"
            }
            _ => return Err("F16 calibration did not decode raw bits".into()),
        },
        "bf16_decoder" => match expected.data {
            TensorData::BF16(v) => {
                let raw = safe
                    .tensor(&output.tensor_ref)
                    .map_err(|error| error.to_string())?;
                let bits = decode_words(raw.data(), 2, |bytes| {
                    u16::from_le_bytes([bytes[0], bytes[1]])
                });
                if !bits
                    .iter()
                    .zip(&v)
                    .all(|(bits, value)| *bits == value.to_bits())
                {
                    return Err("BF16 calibration did not preserve raw bits".into());
                }
                "decoder_bf16"
            }
            _ => return Err("BF16 calibration did not decode raw bits".into()),
        },
        "empty_tensor" => {
            if expected.len() == 0 && expected.shape == observed[0].shape && observed[0].len() == 0
            {
                "empty"
            } else {
                return Err("empty tensor calibration failed".into());
            }
        }
        "endianness" => match expected.data {
            TensorData::I32(v) => {
                let raw = safe
                    .tensor(&output.tensor_ref)
                    .map_err(|error| error.to_string())?;
                let values = decode_words(raw.data(), 4, |bytes| {
                    i32::from_le_bytes(bytes.try_into().unwrap())
                });
                let calibrated = values
                    .iter()
                    .zip(&v)
                    .all(|(decoded, value)| decoded == value);
                if !calibrated || !v.iter().any(|value| *value != value.swap_bytes()) {
                    return Err("endianness calibration failed".into());
                }
                "endianness"
            }
            _ => return Err("endianness calibration requires I32".into()),
        },
        other => return Err(format!("unknown mutation kind {other}")),
    };
    Ok(result.into())
}

trait PolicyValues {
    fn atol(&self) -> f64;
    fn rtol(&self) -> f64;
}
impl PolicyValues for Policy {
    fn atol(&self) -> f64 {
        match self {
            Policy::Float { atol, .. } => *atol,
            Policy::ExactNumeric => 0.0,
        }
    }
    fn rtol(&self) -> f64 {
        match self {
            Policy::Float { rtol, .. } => *rtol,
            Policy::ExactNumeric => 0.0,
        }
    }
}

pub(super) fn harness_qualification() {
    let loaded = match load_corpus() {
        Ok(value) => value,
        Err(failures) => return assert_failures(failures),
    };
    assert_failures(preflight(&loaded));
    let qualification: Qualification = match read_json(&loaded.root.join("qualification.json")) {
        Ok(value) => value,
        Err(error) => return assert_failures(vec![error]),
    };
    with_cpu_defaults(|| {
        assert_failures(qualification_results(&loaded, &qualification));
    });
}
