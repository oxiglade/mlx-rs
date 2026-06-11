//! Streams assistant deltas to a writer, colouring reasoning vs. answer.
//!
//! `<think>` / `</think>` tag literals are suppressed; the text between
//! them renders dim (reasoning) and the rest renders in the answer colour.
//! On a non-TTY writer (tests, pipes) no escape codes are emitted, so the
//! output is the plain tag-stripped text.

use std::io::{self, IsTerminal, Write};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

const C_ANSWER: &str = "\x1b[1;32m"; // bold green
const C_THINK: &str = "\x1b[2m"; // dim
const C_RESET: &str = "\x1b[0m";

/// Forwards streamed deltas to `out`, stripping `<think>` tags and
/// colouring reasoning (dim) vs. answer (bold green) when `tty` is set.
pub struct ThinkStream<W: Write> {
    out: W,
    tty: bool,
    /// Trailing bytes that could be a tag prefix straddling the next push;
    /// everything before this flushes immediately.
    buf: String,
    /// Currently inside a `<think>…</think>` span.
    thinking: bool,
    /// The colour escape currently written to the terminal, so we only
    /// emit a new one on a real state change.
    active_colour: Option<&'static str>,
}

impl<W: Write> ThinkStream<W> {
    pub fn new(out: W) -> Self
    where
        W: IsTerminal,
    {
        let tty = out.is_terminal();
        Self::with_tty(out, tty)
    }

    /// Build with an explicit TTY flag (tests pass `false` to get plain
    /// text with no escape codes).
    pub fn with_tty(out: W, tty: bool) -> Self {
        Self {
            out,
            tty,
            buf: String::new(),
            thinking: false,
            active_colour: None,
        }
    }

    pub fn push(&mut self, delta: &str) -> io::Result<()> {
        self.buf.push_str(delta);
        loop {
            match first_tag(&self.buf) {
                Some((pos, len)) => {
                    let before = self.buf[..pos].to_owned();
                    let after = self.buf[pos + len..].to_owned();
                    self.emit(&before)?;
                    // Toggle on whichever tag matched.
                    self.thinking = self.buf[pos..pos + len].starts_with(THINK_OPEN);
                    self.buf = after;
                }
                None => {
                    let hold = tail_partial_tag_len(&self.buf, THINK_OPEN)
                        .max(tail_partial_tag_len(&self.buf, THINK_CLOSE));
                    let split = self.buf.len() - hold;
                    if split > 0 {
                        let flushable = self.buf[..split].to_owned();
                        self.buf.drain(..split);
                        self.emit(&flushable)?;
                    }
                    return Ok(());
                }
            }
        }
    }

    pub fn finish(&mut self) -> io::Result<()> {
        if !self.buf.is_empty() {
            let trailing = std::mem::take(&mut self.buf);
            self.emit(&trailing)?;
        }
        if self.active_colour.is_some() {
            self.out.write_all(C_RESET.as_bytes())?;
            self.active_colour = None;
        }
        self.out.flush()
    }

    pub fn into_inner(self) -> W {
        self.out
    }

    /// Write `text` in the current think/answer colour, switching the
    /// terminal colour only when the state actually changes. Flushes so
    /// each delta appears token-by-token rather than buffered to a newline.
    fn emit(&mut self, text: &str) -> io::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if self.tty {
            let want = if self.thinking { C_THINK } else { C_ANSWER };
            if self.active_colour != Some(want) {
                self.out.write_all(want.as_bytes())?;
                self.active_colour = Some(want);
            }
        }
        self.out.write_all(text.as_bytes())?;
        self.out.flush()
    }
}

/// `(byte_pos, byte_len)` of the earliest `<think>` / `</think>`.
fn first_tag(buf: &str) -> Option<(usize, usize)> {
    let open = buf.find(THINK_OPEN).map(|p| (p, THINK_OPEN.len()));
    let close = buf.find(THINK_CLOSE).map(|p| (p, THINK_CLOSE.len()));
    match (open, close) {
        (Some(a), Some(b)) if a.0 <= b.0 => Some(a),
        (Some(_), Some(b)) => Some(b),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Length of the longest tail of `buf` that is a strict prefix of
/// `needle`. The streamer holds back exactly that many chars between
/// pushes so a split tag can still match.
fn tail_partial_tag_len(buf: &str, needle: &str) -> usize {
    let max = (needle.len() - 1).min(buf.len());
    for k in (1..=max).rev() {
        if buf.is_char_boundary(buf.len() - k) && needle.starts_with(&buf[buf.len() - k..]) {
            return k;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(input: &str) -> String {
        let mut s = ThinkStream::with_tty(Vec::<u8>::new(), false);
        s.push(input).unwrap();
        s.finish().unwrap();
        String::from_utf8(s.into_inner()).unwrap()
    }

    fn render_split(chunks: &[&str]) -> String {
        let mut s = ThinkStream::with_tty(Vec::<u8>::new(), false);
        for c in chunks {
            s.push(c).unwrap();
        }
        s.finish().unwrap();
        String::from_utf8(s.into_inner()).unwrap()
    }

    #[test]
    fn passthrough_when_no_tags() {
        assert_eq!(render("plain text\n"), "plain text\n");
    }

    #[test]
    fn strips_open_and_close_pair() {
        assert_eq!(
            render("hi <think>reason</think> after\n"),
            "hi reason after\n"
        );
    }

    #[test]
    fn strips_stray_close_only() {
        // qwen 3.6 pattern: only `</think>` arrives; opener was in the prompt.
        assert_eq!(render("reason</think>answer\n"), "reasonanswer\n");
    }

    #[test]
    fn strips_stray_open_only() {
        assert_eq!(render("<think>only opener"), "only opener");
    }

    #[test]
    fn tag_split_across_pushes() {
        assert_eq!(
            render_split(&["before <thi", "nk>reason</thi", "nk> after\n"]),
            "before reason after\n"
        );
    }

    #[test]
    fn tag_at_very_start() {
        assert_eq!(render("</think>plain text\n"), "plain text\n");
    }

    #[test]
    fn no_false_positive_on_partial_tag_at_eof() {
        // `finish()` flushes a held-back tag prefix as raw content.
        assert_eq!(render("text <thi"), "text <thi");
    }

    #[test]
    fn tty_colours_think_and_answer() {
        let mut s = ThinkStream::with_tty(Vec::<u8>::new(), true);
        s.push("answer <think>reasoning</think> more\n").unwrap();
        s.finish().unwrap();
        let out = String::from_utf8(s.into_inner()).unwrap();
        assert!(out.starts_with(C_ANSWER));
        assert!(out.contains(C_THINK));
        assert!(out.ends_with(C_RESET));
        // tag literals are still stripped under colour.
        assert!(!out.contains("<think>") && !out.contains("</think>"));
    }
}
