"""Capture a single served checkpoint in its own pinned Python environment."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=('qwen', 'inkling'))
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    path = args.checkpoint
    ink = args.family == 'inkling'
    mx.random.seed(0)
    if ink:
        from mlx_vlm import load, stream_generate
        mx.set_memory_limit(40_000_000_000)
        mx.set_cache_limit(2_000_000_000)
        model, renderer = load(str(path), trust_remote_code=False)
        tok = renderer.tokenizer
    else:
        from mlx_lm import load, stream_generate
        model, renderer = load(str(path))
        tok = renderer
    model.eval()
    tools = [{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{'ticket':{'type':'integer'}},'required':['ticket']}}}]
    messages = [{'role':'user','content':'Call lookup for ticket 413.'}]
    cases = []
    for history in [messages, messages + [
        {'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'lookup','arguments':{'ticket':413}}}]},
        {'role':'tool','name':'lookup','content':'LARCH-481'},
        {'role':'user','content':'Reply with just the lookup result.'},
    ]]:
        prompt = renderer.apply_chat_template(history, tools=tools, tokenize=False, add_generation_prompt=True, **({'reasoning_effort':'high'} if ink else {}))
        ids = tok.encode(prompt, add_special_tokens=False)
        if ink:
            for r in stream_generate(model, renderer, prompt=prompt, input_ids=mx.array([ids]), max_tokens=512, temperature=0., prefill_step_size=512):
                pass
            output = r.token_ids
        else:
            output = []
            for r in stream_generate(model, tok, prompt=mx.array(ids), max_tokens=512, prefill_step_size=2048):
                output.append(r.token)
        cases.append(dict(messages=history, tools=tools, prompt_ids=ids, output_ids=output, finish=r.finish_reason))
    files = sorted([*path.glob('*.safetensors'), *path.glob('*.json'), *path.glob('*.jinja')])
    hashes = {}
    for file in files:
        with file.open('rb') as f:
            hashes[file.name] = hashlib.file_digest(f, 'sha256').hexdigest()
    packages = ['mlx','mlx-metal','mlx-vlm' if ink else 'mlx-lm','transformers','tokenizers']
    print(json.dumps(dict(checkpoint=path.name, sha256=hashes, versions={p:metadata.version(p) for p in packages}, implementation=metadata.distribution(packages[2]).read_text('direct_url.json'), cases=cases), sort_keys=True))


if __name__ == '__main__':
    main()
