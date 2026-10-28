# Reviewed GGUF reference boundary

`gguf_writer.py` writes GGUF v3 directly using Python's standard library. Tensor
dimensions on disk are reversed. It packs Q4 low nibbles for columns 0..15 and high
nibbles for columns 16..31; Q8 payload bytes are signed. Each 32-value block stores
an F16 scale; Q4_1 also stores an F16 minimum. Norms remain F32.

The basis is the existing `llama-base` or `qwen3-base` checkpoint, with its exact
tokenizer sidecars, prompt strings and IDs. Tied cases omit the output group. For
untied Qwen3, the output is the existing embedding with rows rolled by seven. Qwen3
Q/K norm vectors use `1 + 8 * (base - 1)` so a norm swap has a value witness under
the fixed F16 tolerance. Other base weights are reused before storage conversion.
These transformations are fixture witnesses, not a model conversion product.

`gguf_bridge.py` reads the core-converted container with `mx.load`. It maps each
external tensor to an explicit checkpoint key and maps GGUF metadata to the pinned
upstream architecture's `ModelArgs`. The upstream affine constructors call
`mx.quantize`; the bridge therefore allocates their module instances without those
constructors, sets group size 32 and the observed 4/8 bits, and injects all three
converted slots. Strict upstream loading verifies complete assignment. It never
quantizes initialized model weights.

Llama Q/K use the inverse of the pinned exporter's operation, on packed weights,
scales and biases alike. One head with dimension 16 has reviewed exported row order:

```text
[0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15]
```

The writer's Llama F32 names and actual Q/K arrays are checked against
`mlx_lm.gguf.translate_weight_names` and `permute_weights` in mlx-lm 0.31.3
(upstream commit `ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`). The literal witness
is independent of both helpers; changing a writer and inverse together cannot
redefine it.

Qwen3 keeps row order `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]`. The reviewed
external convention is the ordinary dense Qwen3 path through `Qwen3Model`,
`Qwen2Model.modify_tensors`, and the generic name mapping in llama.cpp's
[Qwen converter](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py).
That path does not apply Llama's permutation. The latest file-history entry
observed during this review was `9a7570587ce908b0073a0458877205b80627f393`;
fetching its immutable source was unavailable in this sandbox. Immutable-source
qualification remains pending until the launcher supplies it. The live source
review alone must not be called a pinned converter qualification.

`numpy_reference/gguf.py` independently parses the binary, reverses dimensions,
unpacks integer lanes and reconstructs affine slots. Q4_0 bias is F16(-8*d), Q8_0
bias is F16(-128*d), and Q4_1 uses its stored minimum. Before any forward comparison,
the converted U32 words and F16 scales/biases must match Python core bit for bit.
Dequantized witnesses use `F16(F16(scale*q)+bias)`, rounding both operations in
F16 under ruling J. NumPy's existing decoder then computes in F32 from these
decoded values.

The generator qualifies Python against NumPy before freezing any fixture. Pure
F32 uses atol=rtol=2e-4. `gguf-f16-v1` uses atol=rtol=5e-3 for F16 storage.
`gguf-affine-v1` uses atol=2e-2, rtol=2e-3 for Q4_0/Q4_1/Q8_0 logits and cache,
calibrated under ruling K as documented in SCHEMA.md. Per-key reference errors
are frozen as provenance, and greedy-ID equality is a separate exact requirement.
A mismatch stops generation and retains its diagnostic staging tree.
It never widens tolerances. Native Python dtypes are recorded separately from the
F32 comparison payloads and are exact requirements for Rust observations.

Mutation outputs come from executing a changed Python bridge, not editing an
expected scalar to invent sensitivity. Every fixture must kill orientation and
gate/up mutations; Qwen3 adds Q/K norm swaps, tied Q4 adds bypassed affine output,
and mixed Q4_1 adds a shape-preserving wrong-bits reinterpretation of its Q8 head.
Each stores one failed numerical tensor and its expected comparator class.
Affine mutations must exceed the combined tolerance bound by at least 10x;
their actual margins are reported and frozen. A weaker affine mutation or an
inert mutation stops generation and requires witness redesign.

`gguf_cases.json` records malformed recipes and typed boundary expectations.
`gguf_recipes.materialize` produces temporary copies without changing the base.
An attempted canonical-name alias is an unknown external key: the admitted map
has no alias that could validly collide with another spelling. Q5_0 reuses the
core fixture and requires only `GgufError::Exception`, never exception text.
