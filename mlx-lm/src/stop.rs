use crate::GenerationError;

#[derive(Debug)]
pub(crate) struct StopStringFilter {
    stops: Vec<String>,
    held: String,
    stopped: bool,
}

impl StopStringFilter {
    pub(crate) fn new(stops: Vec<String>) -> Result<Self, GenerationError> {
        if let Some(index) = stops.iter().position(String::is_empty) {
            return Err(GenerationError::EmptyStopString { index });
        }
        Ok(Self {
            stops,
            held: String::new(),
            stopped: false,
        })
    }

    pub(crate) fn process(&mut self, delta: &str) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }
        self.held.push_str(delta);
        if let Some(start) = self
            .stops
            .iter()
            .filter_map(|stop| self.held.find(stop))
            .min()
        {
            self.held.truncate(start);
            self.stopped = true;
            return (std::mem::take(&mut self.held), true);
        }

        let held_len = self
            .stops
            .iter()
            .filter_map(|stop| {
                (1..=self.held.len().min(stop.len() - 1))
                    .rev()
                    .find(|&len| self.held.as_bytes().ends_with(&stop.as_bytes()[..len]))
            })
            .max()
            .unwrap_or(0);
        // A matching UTF-8 prefix starts at a character boundary in the held text.
        let suffix = self.held.split_off(self.held.len() - held_len);
        (std::mem::replace(&mut self.held, suffix), false)
    }

    pub(crate) fn finish(mut self, final_delta: &str) -> (String, bool) {
        let (mut text, stopped) = self.process(final_delta);
        text.push_str(&self.held);
        (text, stopped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Deserialize)]
    struct Fixture {
        schema_version: u32,
        stop_strings: BTreeMap<String, Case>,
        errors: BTreeMap<String, ErrorCase>,
    }

    #[derive(Deserialize)]
    struct Case {
        stops: Vec<String>,
        deltas: Vec<String>,
        events: Vec<Event>,
        final_delta: String,
        finish_flush: Option<String>,
        stopped: bool,
        finish_reason: String,
    }

    #[derive(Deserialize)]
    struct Event {
        text: String,
        held: String,
        stopped: bool,
    }

    #[derive(Deserialize)]
    struct ErrorCase {
        stops: Vec<String>,
        expected_error: String,
    }

    #[test]
    fn oracle_text_cases() -> Result<(), Box<dyn std::error::Error>> {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../conformance/mlx-lm/fixtures/llama-base/text_cases.json"
        ))?;
        assert_eq!(fixture.schema_version, 1);
        assert!(!fixture.stop_strings.is_empty());
        for (name, case) in fixture.stop_strings {
            let mut filter = StopStringFilter::new(case.stops)?;
            let mut consumed = 0;
            for delta in &case.deltas {
                let event = &case.events[consumed];
                let (text, stopped) = filter.process(delta);
                assert_eq!(text, event.text, "{name}/{consumed}: text");
                assert_eq!(filter.held, event.held, "{name}/{consumed}: held");
                assert_eq!(stopped, event.stopped, "{name}/{consumed}: stopped");
                consumed += 1;
                if stopped {
                    break;
                }
            }
            assert_eq!(consumed, case.events.len(), "{name}: events");
            if let Some(flush) = case.finish_flush {
                assert!(!filter.stopped, "{name}: premature stop");
                let (text, stopped) = filter.finish(&case.final_delta);
                assert_eq!(text, flush, "{name}: finish flush");
                assert_eq!(stopped, case.stopped, "{name}: final stop");
            } else {
                assert!(filter.stopped, "{name}: missing stop");
                assert!(case.stopped, "{name}: final stop");
                for delta in &case.deltas[consumed..] {
                    assert_eq!(filter.process(delta), (String::new(), true), "{name}");
                }
                assert_eq!(filter.process("ignored"), (String::new(), true), "{name}");
                assert_eq!(filter.finish("ignored"), (String::new(), true), "{name}");
            }
            assert_eq!(
                case.finish_reason,
                if case.stopped { "stop" } else { "length" },
                "{name}: finish reason"
            );
        }
        assert!(!fixture.errors.is_empty());
        for (name, case) in fixture.errors {
            assert_eq!(case.expected_error, "EmptyStopString", "{name}");
            let expected_index = case.stops.iter().position(String::is_empty).unwrap();
            assert!(
                matches!(
                    StopStringFilter::new(case.stops),
                    Err(GenerationError::EmptyStopString { index }) if index == expected_index
                ),
                "{name}"
            );
        }
        Ok(())
    }

    #[test]
    fn empty_stop_reports_first_original_index() {
        assert!(matches!(
            StopStringFilter::new(vec!["END".into(), "END".into(), "".into(), "".into()]),
            Err(GenerationError::EmptyStopString { index: 2 })
        ));
    }

    #[test]
    fn longest_prefix_empty_delta_and_unicode_boundaries() -> Result<(), GenerationError> {
        let mut filter = StopStringFilter::new(vec!["é!".into(), "雪é?".into()])?;
        assert_eq!(filter.process("x雪é"), ("x".into(), false));
        assert_eq!(filter.held, "雪é");
        assert_eq!(filter.process(""), (String::new(), false));
        assert_eq!(filter.held, "雪é");
        assert_eq!(filter.process("雪"), ("雪é".into(), false));
        assert_eq!(filter.held, "雪");
        assert_eq!(filter.finish("é"), ("雪é".into(), false));
        Ok(())
    }
}
