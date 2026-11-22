# mlx-lm server

An unpublished HTTP server over the public `mlx-lm` API. The package and its
binary are `mlx-lm-server`.

```text
mlx-lm-server --model DIR [--host ADDR] [--port PORT]
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
`min_p`, `seed` and `stop` (a string or a list). Absent fields take
`mlx_lm.server`'s defaults: 512 tokens, temperature zero and no filters, and a
zero `top_p`, `top_k` or `min_p` switches that filter off. `model` is echoed and
otherwise ignored, and other fields are ignored.

The reply's message carries `content`, `reasoning` when the model reasoned, and
`tool_calls` with each call's arguments as a JSON string. `finish_reason` is
`length`, `tool_calls` or `stop`, and `usage` counts prompt and completion tokens.
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

Messages reach the template as sent. `mlx_lm.server` first joins a list of text
parts into one string, reads a null `content` as empty and parses each tool call's
`arguments` string into an object; this server does none of that yet, so a
conversation that echoes tool calls back in the OpenAI wire format renders
differently.

Not supported: prompt caching between requests; images; penalties, `logprobs` and `chat_template_kwargs`; and Inkling's
`reasoning_content` field name.
