#[path = "conformance/adapters.rs"]
mod adapters;
#[path = "conformance/oracle.rs"]
mod oracle;

#[test]
fn committed_corpus() {
    oracle::committed_corpus();
}

#[test]
fn committed_gguf_corpus() {
    oracle::gguf_committed_corpus();
}

#[test]
fn harness_qualification() {
    oracle::harness_qualification();
}
