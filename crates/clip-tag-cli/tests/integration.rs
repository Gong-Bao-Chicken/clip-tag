#[test]
#[ignore = "requires downloaded CLIP model and vocab cache"]
fn fixture_corpus_tags_deterministically() {
    let corpus =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/corpus/sample.jpg");

    let run = || {
        assert_cmd::Command::cargo_bin("clip-tag")
            .unwrap()
            .arg(&corpus)
            .arg("--json")
            .arg("--top-k")
            .arg("5")
            .output()
            .expect("run clip-tag")
    };

    let a = run();
    let b = run();
    assert!(a.status.success());
    assert_eq!(a.stdout, b.stdout);
}
