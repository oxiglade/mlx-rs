use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
};

use mlx_lm::{FinishReason, GenerationEvent, GenerationOptions, Model, Prompt};
use serde_json::{json, Value};

const PROMPT: &str = "hello the small fox runs over green hill";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures/qwen3-base")
}

struct Server {
    child: Child,
    address: String,
}
impl Server {
    fn start(model: &Path, flags: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_mlx-lm-server"))
            .args(["--port", "0", "--model"])
            .arg(model)
            .args(flags)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let address = line
            .trim()
            .strip_prefix("listening on http://")
            .unwrap()
            .to_owned();
        Self { child, address }
    }
    fn send(&self, method: &str, path: &str, body: &str) -> (u16, String) {
        let mut stream = TcpStream::connect(&self.address).unwrap();
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            self.address,
            body.len()
        )
        .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        let status = head.split(' ').nth(1).unwrap().parse().unwrap();
        (status, body.to_owned())
    }
    fn request(&self, method: &str, path: &str, body: &str) -> (u16, Value) {
        let (status, body) = self.send(method, path, body);
        (status, serde_json::from_str(&body).unwrap())
    }
    fn chat(&self, body: Value) -> (u16, Value) {
        self.request("POST", "/v1/chat/completions", &body.to_string())
    }
    /// The comments and the data of a streamed reply's events. Each event is one HTTP
    /// chunk, so its lines arrive whole between the chunk sizes.
    fn stream(&self, mut body: Value) -> (Vec<String>, Vec<String>) {
        body["stream"] = true.into();
        let (status, body) = self.send("POST", "/v1/chat/completions", &body.to_string());
        assert_eq!(status, 200, "{body}");
        let lines = |prefix| {
            let lines = body.lines().filter_map(|line| line.strip_prefix(prefix));
            lines.map(str::to_owned).collect()
        };
        (lines(": "), lines("data: "))
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

type Configure = fn(&mut GenerationOptions);

/// A copy of the fixture model that renders chats with `template`.
fn fixture_with_template(template: &str) -> tempfile::TempDir {
    let model = tempfile::tempdir().unwrap();
    for entry in fs::read_dir(fixture()).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), model.path().join(entry.file_name())).unwrap();
    }
    let path = model.path().join("tokenizer_config.json");
    let mut config: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    config["chat_template"] = template.into();
    fs::write(&path, config.to_string()).unwrap();
    model
}

fn tokens(prompt: &str) -> usize {
    let model = Model::from_dir(fixture()).unwrap();
    model.tokenizer().encode(prompt).unwrap().len()
}

/// What the library generates after `prompt`: the prompt's token count, the sampled
/// tokens' count, their text and the finish reason.
fn generated(
    prompt: &str,
    configure: impl FnOnce(&mut GenerationOptions),
) -> (usize, usize, String, FinishReason) {
    let mut model = Model::from_dir(fixture()).unwrap();
    let tokens = model.tokenizer().encode(prompt).unwrap();
    let mut options = GenerationOptions::default();
    configure(&mut options);
    let (mut count, mut text, mut finish) = (0, String::new(), None);
    for event in model.generate(Prompt::Tokens(&tokens), options).unwrap() {
        if let GenerationEvent::Token {
            text: delta,
            finish_reason,
            ..
        } = event.unwrap()
        {
            count += 1;
            text += &delta;
            finish = finish_reason;
        }
    }
    (tokens.len(), count, text, finish.unwrap())
}

#[test]
fn chat_completion_returns_what_the_library_generates() {
    let server = Server::start(&fixture(), &[]);
    assert_eq!(
        server.request("GET", "/health", ""),
        (200, json!({ "status": "ok" }))
    );
    let (status, models) = server.request("GET", "/v1/models", "");
    assert_eq!(status, 200);
    let id = models["data"][0]["id"].as_str().unwrap();
    assert!(id.ends_with("/qwen3-base"), "{id}");

    let messages = json!([{ "role": "user", "content": PROMPT }]);
    let prompt = format!("<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n");
    let cases: [(Value, Configure); 4] = [
        // Without a limit the reply may fill the fixture's 128 positions after the 13 of
        // the prompt.
        (json!({}), |options| {
            options.max_tokens = 115.try_into().unwrap()
        }),
        (
            json!({ "max_tokens": 8, "max_completion_tokens": 3, "model": "other" }),
            |options| options.max_tokens = 3.try_into().unwrap(),
        ),
        (json!({ "max_tokens": 8, "stop": " fox" }), |options| {
            options.max_tokens = 8.try_into().unwrap();
            options.stop.stop_strings = vec![" fox".into()];
        }),
        (
            json!({ "max_tokens": 8, "temperature": 1.5, "seed": 7, "top_k": 40,
                    "top_p": 0, "min_p": 0, "stop": "" }),
            |options| {
                options.max_tokens = 8.try_into().unwrap();
                options.sampling.temperature = 1.5;
                options.sampling.seed = Some(7);
                options.sampling.top_k = Some(40.try_into().unwrap());
            },
        ),
    ];
    let mut finishes = Vec::new();
    for (mut body, configure) in cases {
        let model = body["model"].as_str().unwrap_or(id).to_owned();
        body["messages"] = messages.clone();
        let (status, response) = server.chat(body.clone());
        assert_eq!(status, 200, "{response}");
        let (prompt_tokens, completion_tokens, text, finish) = generated(&prompt, configure);
        let finish = match finish {
            FinishReason::Stop => "stop",
            _ => "length",
        };
        let choice = &response["choices"][0];
        assert_eq!(
            choice["message"],
            json!({ "role": "assistant", "content": (!text.is_empty()).then_some(&text) }),
            "{body}"
        );
        assert_eq!(choice["finish_reason"], finish, "{body}");
        // The cache tests cover how many of the prompt's tokens were cached.
        let expected = |usage: &Value| {
            let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64();
            json!({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": { "cached_tokens": cached.unwrap() },
            })
        };
        assert_eq!(response["usage"], expected(&response["usage"]), "{body}");
        assert_eq!(response["object"], "chat.completion");
        assert_eq!(response["model"], model);
        finishes.push(finish);

        let (comments, data) = server.stream(body.clone());
        assert!(comments[0].starts_with("keepalive 0/"), "{comments:?}");
        assert!(!data.iter().any(|data| data.contains("usage")), "{data:?}");
        body["stream_options"] = json!({ "include_usage": true });
        let (_, data) = server.stream(body.clone());
        let [chunks @ .., last, usage, done] = &data[..] else {
            panic!("{data:?}")
        };
        assert_eq!(done, "[DONE]");
        let parse = |data: &String| serde_json::from_str::<Value>(data).unwrap();
        let (last, usage) = (parse(last), parse(usage));
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"], expected(&usage["usage"]), "{body}");
        assert_eq!(
            last["choices"],
            json!([{ "index": 0, "finish_reason": finish, "delta": { "role": "assistant" } }]),
            "{body}"
        );
        let mut streamed = String::new();
        for chunk in chunks.iter().map(parse) {
            let choice = &chunk["choices"][0];
            assert_eq!(choice["finish_reason"], Value::Null);
            assert_eq!(choice["delta"]["role"], "assistant");
            streamed += choice["delta"]["content"].as_str().unwrap();
            for (field, value) in [
                ("id", &last["id"]),
                ("object", &json!("chat.completion.chunk")),
                ("model", &json!(model)),
            ] {
                assert_eq!(&chunk[field], value);
                assert_eq!(&usage[field], value);
            }
        }
        assert_eq!(streamed, text, "{body}");
    }
    assert_eq!(finishes, ["length", "length", "stop", "length"]);

    let long = [PROMPT; 16].join(" ");
    for (body, diagnostic) in [
        (
            json!({ "messages": messages, "temperature": -1 }).to_string(),
            "temperature",
        ),
        (
            json!({ "messages": messages, "temperature": -1, "stream": true }).to_string(),
            "temperature",
        ),
        (
            json!({ "messages": messages, "max_tokens": 0 }).to_string(),
            "max_tokens",
        ),
        (
            json!({ "messages": [{ "role": "user", "content": [{ "type": "image_url" }] }] })
                .to_string(),
            "text content",
        ),
        (
            json!({ "messages": [{ "role": "assistant", "tool_calls": [
                { "function": { "name": "lookup", "arguments": "{" } },
            ] }] })
            .to_string(),
            "arguments",
        ),
        (
            json!({ "max_tokens": 1, "messages": [{ "role": "user", "content": long }] })
                .to_string(),
            "context",
        ),
        ("{".to_owned(), "JSON"),
    ] {
        let (status, response) = server.request("POST", "/v1/chat/completions", &body);
        assert_eq!(status, 400, "{response}");
        let message = response["error"].as_str().unwrap();
        assert!(message.contains(diagnostic), "{message}");
    }
    let body = json!({ "messages": messages, "max_tokens": 1 });
    let (status, _) = server.chat(body.clone());
    assert_eq!(status, 200);
    // A repeat rewinds to the end of the user message, streamed or not.
    let mut body = body;
    body["stream_options"] = json!({ "include_usage": true });
    let (_, data) = server.stream(body);
    let usage: Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
    assert_eq!(
        usage["usage"]["prompt_tokens_details"]["cached_tokens"],
        tokens(&format!("<|im_start|>user\n{PROMPT}<|im_end|>\n"))
    );
}

const ASSISTANT: &str = "<|im_start|>assistant\n";

type Messages<'a> = [(&'a str, &'a str)];

fn render(messages: &Messages) -> String {
    let turns = messages.iter();
    let turns = turns.map(|(role, content)| format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    turns.collect()
}

/// The token count of the prompt that asks for a reply to `messages`.
fn prompt(messages: &Messages) -> usize {
    tokens(&(render(messages) + ASSISTANT))
}

/// Checks the server's reply against a run without a cache and returns it with the counts
/// of cached and of sampled tokens.
fn ask(server: &Server, messages: &Messages, max_tokens: usize) -> (String, usize, usize) {
    let sent = messages.iter();
    let sent = sent.map(|(role, content)| json!({ "role": role, "content": content }));
    let sent: Vec<_> = sent.collect();
    let (status, response) = server.chat(json!({ "messages": sent, "max_tokens": max_tokens }));
    assert_eq!(status, 200, "{response}");
    let (prompt_tokens, sampled, text, _) = generated(&(render(messages) + ASSISTANT), |options| {
        options.max_tokens = max_tokens.try_into().unwrap()
    });
    assert_eq!(
        response["choices"][0]["message"]["content"],
        json!((!text.is_empty()).then_some(&text)),
        "{messages:?}"
    );
    let usage = &response["usage"];
    assert_eq!(usage["prompt_tokens"], prompt_tokens);
    let cached = usage["prompt_tokens_details"]["cached_tokens"].as_u64();
    (text, cached.unwrap() as usize, sampled)
}

#[test]
fn a_request_continues_from_the_longest_cached_state_it_extends() {
    let server = Server::start(&fixture(), &[]);
    let length = |messages: &Messages| tokens(&render(messages));
    // Long enough that every request below keeps more than half of the cache it reuses.
    let brief = ["be brief"; 20].join(" ");
    let (system, user) = (("system", brief.as_str()), ("user", "hello the small fox"));
    let (reply, cached, sampled) = ask(&server, &[system, user], 4);
    assert_eq!(cached, 0);

    // The cache holds the prompt and every sampled token but the last.
    let again = ("user", "runs over");
    let follow_up = [system, user, ("assistant", reply.as_str()), again];
    assert_eq!(
        ask(&server, &follow_up, 1).1,
        prompt(&follow_up[..2]) + sampled - 1
    );
    // After a reply of one token the cache ends where the prompt ends. A repeat has to
    // prefill a token, so it rewinds to the user message's end and takes no second
    // checkpoint there.
    for _ in 0..2 {
        assert_eq!(ask(&server, &follow_up, 1).1, length(&follow_up));
    }

    // The first round differs from the follow-up after the first user message. Each later
    // round extends the prompt before it.
    let tool = ("tool", "a");
    let mut rounds = vec![system, user];
    let mut expected = length(&rounds);
    for _ in 0..5 {
        rounds.extend([("assistant", "b"), tool]);
        assert_eq!(ask(&server, &rounds, 1).1, expected);
        expected = prompt(&rounds);
    }
    // A branch off the second round ends past the later rounds' ends, which a request
    // that leaves the branch midway must not rewind to.
    for answer in ["c c c c c c c c c c c c", "c c c c c c c c b"] {
        let cached = ask(
            &server,
            &[&rounds[..6], &[("assistant", answer), tool]].concat(),
            4,
        )
        .1;
        assert_eq!(cached, length(&rounds[..6]));
    }
    // Only the ends of the last four requests are kept, so the first round's is gone.
    let branch = [("assistant", "c"), tool];
    let cached = ask(&server, &[&rounds[..4], &branch].concat(), 4).1;
    assert_eq!(cached, length(&rounds[..2]));

    // The system message's checkpoint is taken where the first message after it starts
    // to differ from an empty user message.
    let shared = tokens(&(render(&[system]) + "<|im_start|>user\n"));
    let mut turns = vec![system, ("user", "world")];
    let mut expected = shared;
    for _ in 0..4 {
        assert_eq!(ask(&server, &turns, 1).1, expected);
        expected = prompt(&turns);
        turns.extend([("assistant", "b"), again]);
    }
    // Of the ends of user messages, the last three are kept beside the system message's.
    let cached = ask(&server, &[&turns[..2], &branch].concat(), 4).1;
    assert_eq!(cached, shared);

    assert_eq!(ask(&server, &[("system", "answer yes"), user], 4).1, 0);
}

#[test]
fn conversations_keep_their_caches_while_they_fit_the_budget() {
    let system = ("system", "be brief");
    let long = [
        system,
        ("user", "hello the small fox runs over green hill again"),
    ];
    let other = [system, ("user", "world")];
    let more = [("assistant", "b"), ("user", "again")];

    // The other chat shares only the system message with the long one. Continuing from
    // there would drop most of the long one, so it starts its own cache and both go on.
    let server = Server::start(&fixture(), &[]);
    assert_eq!(ask(&server, &long, 1).1, 0);
    assert_eq!(ask(&server, &other, 1).1, 0);
    assert_eq!(
        ask(&server, &[&long[..], &more].concat(), 1).1,
        prompt(&long)
    );
    assert_eq!(
        ask(&server, &[&other[..], &more].concat(), 1).1,
        prompt(&other)
    );

    // Without a budget only the last chat's cache stays, and a request that fails is not
    // a chat. The long chat's last request continues from the other chat's system
    // message, which is more than half of that cache.
    let server = Server::start(&fixture(), &["--cache-bytes", "0"]);
    assert_eq!(ask(&server, &long, 1).1, 0);
    let (status, _) = server.chat(json!({ "messages": [], "stop": [""] }));
    assert_eq!(status, 400);
    let longer = [&long[..], &more].concat();
    assert_eq!(ask(&server, &longer, 1).1, prompt(&long));
    assert_eq!(ask(&server, &other, 1).1, 0);
    let shared = tokens(&(render(&[system]) + "<|im_start|>user\n"));
    assert_eq!(ask(&server, &[&longer[..], &more].concat(), 1).1, shared);
}

#[test]
fn messages_reach_the_template_as_mlx_lm_server_prepares_them() {
    let model = fixture_with_template(concat!(
        "{%- for message in messages %}",
        "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] }}",
        "{%- for call in message['tool_calls'] or [] %}",
        "{{- ' ' + call['function']['arguments']['word'] }}",
        "{%- endfor %}{{- '<|im_end|>\n' }}{%- endfor %}",
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}",
    ));

    let server = Server::start(model.path(), &[]);
    let call = |arguments: Value| json!({ "id": "1", "type": "function", "function": { "name": "lookup", "arguments": arguments } });
    let (status, response) = server.chat(json!({ "max_tokens": 4, "messages": [
        { "role": "user", "content": [
            { "type": "text", "text": "hello the" },
            { "type": "text", "text": " small fox" },
        ] },
        { "role": "assistant", "content": null, "tool_calls": [call(json!("{\"word\": \"hill\"}"))] },
        { "role": "assistant", "tool_calls": [call(json!({ "word": "green" }))] },
        { "role": "user", "content": "again" },
    ] }));
    assert_eq!(status, 200, "{response}");
    let (prompt_tokens, _, text, _) = generated(
        "<|im_start|>user\nhello the small fox<|im_end|>\n<|im_start|>assistant\n hill<|im_end|>\n\
         <|im_start|>assistant\n green<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n\
         <|im_start|>assistant\n",
        |options| options.max_tokens = 4.try_into().unwrap(),
    );
    assert_eq!(response["usage"]["prompt_tokens"], prompt_tokens);
    assert_eq!(response["choices"][0]["message"]["content"], text);
}

#[test]
fn flags_set_what_a_request_leaves_out() {
    let flags = [
        "--max-tokens",
        "3",
        "--chat-template-args",
        r#"{"enable_thinking": false}"#,
    ];
    let server = Server::start(&fixture(), &flags);
    let open = format!("<|im_start|>user\n{PROMPT}<|im_end|>\n{ASSISTANT}");
    let closed = format!("{open}<think>\n\n</think>\n\n");
    let request = json!({ "max_tokens": 5, "chat_template_kwargs": { "enable_thinking": true } });
    for (mut body, prompt, max_tokens) in [(json!({}), closed, 3), (request, open, 5)] {
        body["messages"] = json!([{ "role": "user", "content": PROMPT }]);
        let (status, response) = server.chat(body.clone());
        assert_eq!(status, 200, "{response}");
        let (prompt_tokens, _, text, _) = generated(&prompt, |options| {
            options.max_tokens = max_tokens.try_into().unwrap()
        });
        assert_eq!(response["usage"]["prompt_tokens"], prompt_tokens, "{body}");
        assert_eq!(response["usage"]["completion_tokens"], max_tokens, "{body}");
        assert_eq!(response["choices"][0]["message"]["content"], text, "{body}");
    }
}

#[test]
fn reasoning_effort_reaches_the_template_and_reasoning_takes_the_named_field() {
    let model = fixture_with_template(concat!(
        "{{- '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n' }}",
        "{{- '<|im_start|>assistant\n' + reasoning_effort + '\n' }}",
        "{%- if enable_thinking %}{{- '<think>\n' }}{%- endif %}",
    ));
    let flags = [
        "--reasoning-field",
        "reasoning_content",
        "--chat-template-args",
        r#"{"reasoning_effort": "again", "enable_thinking": false}"#,
    ];
    let server = Server::start(model.path(), &flags);
    let start = format!("<|im_start|>user\n{PROMPT}<|im_end|>\n{ASSISTANT}");
    // The request's own `enable_thinking` stands; its `reasoning_effort` kwarg does not.
    let kwargs = json!({ "enable_thinking": true, "reasoning_effort": "green hill" });
    let thinks = json!({ "enable_thinking": true });
    for (effort, kwargs, prompt, field) in [
        (
            json!("over"),
            json!({}),
            format!("{start}over\n<think>\n"),
            "reasoning_content",
        ),
        (
            json!("none"),
            json!({}),
            format!("{start}none\n"),
            "content",
        ),
        (
            json!("none"),
            kwargs,
            format!("{start}none\n<think>\n"),
            "reasoning_content",
        ),
        (
            Value::Null,
            thinks,
            format!("{start}again\n<think>\n"),
            "reasoning_content",
        ),
    ] {
        let body = json!({
            "messages": [{ "role": "user", "content": PROMPT }],
            "max_tokens": 3,
            "reasoning_effort": effort,
            "chat_template_kwargs": kwargs,
        });
        let (status, response) = server.chat(body.clone());
        assert_eq!(status, 200, "{response}");
        let (prompt_tokens, _, text, _) = generated(&prompt, |options| {
            options.max_tokens = 3.try_into().unwrap()
        });
        assert_eq!(response["usage"]["prompt_tokens"], prompt_tokens, "{body}");
        let mut message = json!({ "role": "assistant", "content": null });
        message[field] = text.into();
        assert_eq!(response["choices"][0]["message"], message, "{body}");
        let (_, data) = server.stream(body);
        assert!(data[0].contains(&format!("\"{field}\"")), "{data:?}");
    }
}

#[test]
fn a_model_that_does_not_load_ends_the_process() {
    let output = Command::new(env!("CARGO_BIN_EXE_mlx-lm-server"))
        .arg("--model")
        .arg(fixture().join("config.json"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).starts_with("mlx-lm-server: "));
}
