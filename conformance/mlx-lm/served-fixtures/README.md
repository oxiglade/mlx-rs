# Stock Qwen decode replay

Run from the repository root on an idle Apple Silicon GPU. `MODEL` is the local checkpoint directory whose file hashes appear in `Qwen3.6-35B-A3B-4bit.json`. Build `cargo build --release --locked -p mlx-lm --example qwen_hybrid` first. Timings exclude loading the model. Each prompt starts with an empty cache.

For the two tool turns, compare the emitted IDs, including the stop token when present:

```bash
set -e
fixture=conformance/mlx-lm/served-fixtures/Qwen3.6-35B-A3B-4bit.json
for i in 0 1; do
  target/release/examples/qwen_hybrid "$MODEL" "$(jq -c ".cases[$i].prompt_ids" "$fixture")" 512 > actual.ids
  diff -u <(jq -r ".cases[$i].output_ids[]" "$fixture") actual.ids
done
```

For the 57,130-token pair, use the checkpoint's tokenizer (`tokenizers==0.23.3`). Replacing the first ID of the second prompt makes the two inputs share no prefix. The reference SHA-256 covers both 32-token continuations in order, one decimal ID and newline per token.

```bash
MODEL="$MODEL" python3 - <<'PY'
import hashlib, json, os
from tokenizers import Tokenizer
t = Tokenizer.from_file(os.path.join(os.environ['MODEL'], 'tokenizer.json'))
records = '\n'.join(f'A{i}: '+hashlib.sha256(f'A{i}'.encode()).hexdigest()[:20] for i in range(2295))
for turn in range(2):
    prompt = f'<|im_start|>system\nReference records for project A:\n{records}<|im_end|>\n<|im_start|>user\nReply with just the project letter. Turn {turn}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
    ids = t.encode(prompt, add_special_tokens=False).ids
    assert len(ids) == 57130
    if turn: ids[0] = 42
    with open(f'prompt-{turn}.json', 'w') as f: json.dump(ids, f, separators=(',', ':'))
PY
target/release/examples/qwen_hybrid "$MODEL" "$(cat prompt-0.json)" 32 "$(cat prompt-1.json)" > actual.ids
echo 'bfa4f60950bb3feaa39c40b22b3a98e7841b1353d1d9fb7f91242c96086bc38b  actual.ids' | shasum -a 256 -c -
```

## Inkling 2 Mini

The same example replays `Inkling-2-Mini.json` against the checkpoint whose file hashes it records:

```bash
set -e
fixture=conformance/mlx-lm/served-fixtures/Inkling-2-Mini.json
for i in 0 1; do
  target/release/examples/qwen_hybrid "$MODEL" "$(jq -c ".cases[$i].prompt_ids" "$fixture")" 512 > actual.ids
  diff -u <(jq -r ".cases[$i].output_ids[]" "$fixture") actual.ids
done
```

`Inkling-2-Mini-image.json` holds one prompt with a 100x70 image. The case's `image.rgb` gives the three bytes of the pixel at column `x` and row `y`, and `prompt_ids` holds one image token for it. Replay it with the checkpoint test:

```bash
MLX_LM_INKLING="$MODEL" cargo test --release --locked -p mlx-lm --test real_checkpoints -- --ignored --exact inkling_replays_the_image_fixture
```

## Chat rendering

`served_chat_reference.py qwen|inkling "$MODEL"` prints a `chat/` fixture: the checkpoint's template and, for each case, the messages, tools and template variables with the text Transformers renders and its token IDs. The `served_chat_template_oracles` unit test renders every case from the recorded template. To compare token IDs against a checkpoint's own tokenizer, name its directory after the fixture and run:

```bash
MLX_LM_SERVED_CHAT="$MODEL" cargo test --locked -p mlx-lm --test real_checkpoints -- --ignored --exact served_chat_fixture_tokenizes
```
