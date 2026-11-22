# mlx-lm server

An unpublished HTTP server over the public `mlx-lm` API. The package and its
binary are `mlx-lm-server`.

```text
mlx-lm-server --model DIR [--host ADDR] [--port PORT] [--cache-bytes N]
              [--disk-cache DIR] [--disk-cache-bytes N] [--disk-cache-block N]
              [--max-tokens N] [--chat-template-args JSON] [--reasoning-field NAME]
```

It loads one local safetensors directory, prints `listening on http://ADDR` once
it accepts connections, and serves three routes. `--host` defaults to `127.0.0.1`
and `--port` to 8080; port 0 picks a free port. A model that fails to load ends
the process with status 1.

| Route | Response |
| --- | --- |
| `GET /health` | `{"status": "ok"}` |
| `GET /v1/models` | The loaded model, with its resolved directory as `id`. |
| `POST /v1/chat/completions` | An OpenAI `chat.completion` object, or a stream of chunks with `"stream": true`. |

A chat request renders `messages` and `tools` through the model's chat template,
ending with the assistant prompt, and generates from the result. It reads
`max_completion_tokens` before `max_tokens`, then `temperature`, `top_p`, `top_k`,
`min_p`, `seed` and `stop` (a string or a list). A request without a limit gets
`--max-tokens`, and without that flag its reply may fill the model's context, as in
llama.cpp, vLLM and Ollama. A prompt that leaves no room in the context returns
status 400. The other absent fields take `mlx_lm.server`'s
defaults: temperature zero and no filters, and a zero `top_p`, `top_k` or `min_p`
switches that filter off. `model` is echoed and otherwise ignored, and other fields
are ignored.

The template also sees the variables in `--chat-template-args`, a JSON object, and
those in the request's `chat_template_kwargs`, which override them. A request's
`reasoning_effort` reaches the template under that name, over any variable of that
name, and sets `enable_thinking` to whether it is anything but `"none"`, unless
`chat_template_kwargs` sets that too. This is vLLM's order of precedence.

The reply's message carries `content`, `reasoning` when the model reasoned (named
`reasoning_content` with `--reasoning-field reasoning_content`), and
`tool_calls` with each call's arguments as a JSON string. `finish_reason` is
`length`, `tool_calls` or `stop`, and `usage` counts prompt and completion tokens,
with the prompt tokens taken from the cache in `prompt_tokens_details.cached_tokens`.
A request that cannot be served returns status 400 and `{"error": MESSAGE}`; a
failure during generation returns 500 in the same shape.

With `"stream": true` the reply is server-sent events, framed as `mlx_lm.server`
frames them. Prompt processing sends `: keepalive PROCESSED/TOTAL` comments. Each
`chat.completion.chunk` then carries a `delta` with `role` and the new `content`,
`reasoning` or `tool_calls`; a tool call arrives whole, with its `index`. The last
chunk has an empty delta and the `finish_reason`, and `data: [DONE]` ends the
stream. `"stream_options": {"include_usage": true}` adds a chunk before `[DONE]`
with no choices and the `usage`; it is a `chat.completion.chunk`, as in OpenAI's
API, where `mlx_lm.server` calls it a `chat.completion`. A failure after the
stream has started sends `data: {"error": MESSAGE}` and ends it without `[DONE]`.

The model runs on one thread and serves one request at a time; later requests
wait their turn. A client that disconnects stops its generation at the next
prefill step or token, or leaves the queue.

The server keeps the prompt caches of earlier conversations. A request uses the
cache that holds the most of its prompt. When the prompt continues that cache, only
the tokens after it are prefilled. Otherwise the cache rewinds to the newest
checkpoint the prompt still begins with and prefills from there. A request that
would drop more than half of a cache this way leaves it alone, as llama.cpp does,
and a request no cache serves starts a new one from its first token. After each
request the least recently used caches are dropped until the rest fit in
`--cache-bytes`, 8 GiB by default; the last one used always stays.

Checkpoints are taken during prefill in three places: where the system messages
end, where the last user message ends, and where the messages end when the last
one is not a user's. A cache keeps the first and the three newest of the system and
user checkpoints, and the four newest of the others.

A reply that used the cache can differ from the reply a fresh server gives to the
same request. The cached tokens were prefilled in different steps, which rounds
differently, and a near-tie between two tokens can then fall the other way.

With `--disk-cache DIR` the caches also go to disk, where a later process with the
same model finds them. Once the server has been idle for a second it writes the
conversation of the last request: the K/V up to its first checkpoint and up to its
newest, and those two checkpoints. A request that the files hold more of than any
cache in memory does starts from the longest checkpoint its prompt continues.

The files are in `DIR/NAME-DIGEST`, where the digest is a SHA-256 of a format
number, the server's version, the block size, the bytes of the model's
`config.json` and the paths, sizes and modification times of its weight files. One
server at a time uses that subdirectory, and it leaves the subdirectories of other
models, versions and block sizes as they are. A K/V file holds `--disk-cache-block`
tokens, 2,048 by default, or what a checkpoint has of its last block. Its name is a
SHA-256 of the name of the block before it and its own token IDs, followed by the
count of tokens it ends after, so conversations that begin alike share files. A
checkpoint file takes the name of its last K/V file. Every name ends in a CRC-32 of
the file's bytes; a file that does not match it, or that the library refuses, is
deleted, and the request starts from a shorter checkpoint or from its first token.
Files are written under a temporary name, synced and renamed.

A file's modification time is the end of the last request that used its state. When
the subdirectory would pass `--disk-cache-bytes`, 64 GiB by default, the state used
longest ago loses its checkpoint and then its K/V from the end, and no file of the
conversation being written is dropped. A state too large to fit beside the rest of
its conversation is not written. The budget also applies to the files a starting
server finds.

Weights replaced under the same paths, sizes and modification times are not noticed.

Before the template reads a message, the server prepares it as `mlx_lm.server`
does: a `content` list of text parts becomes one string, an absent or null
`content` becomes empty, and a tool call's `arguments` sent as a JSON string
becomes an object. A content part that is not text, or arguments that are not
JSON, return status 400.

Not supported: images, penalties and `logprobs`.
