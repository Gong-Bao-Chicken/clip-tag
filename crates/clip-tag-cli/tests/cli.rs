fn bin() -> assert_cmd::Command {
    assert_cmd::Command::cargo_bin("clip-tag").unwrap()
}

fn corpus_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/corpus")
        .canonicalize()
        .expect("fixtures corpus")
}

#[test]
fn help_succeeds() {
    bin().arg("--help").assert().success();
}

#[test]
fn directory_requires_recursive() {
    bin()
        .arg(corpus_dir())
        .assert()
        .failure()
        .stderr(predicates::str::contains("--recursive"));
}

#[test]
fn threshold_must_be_unit_interval() {
    bin()
        .arg(corpus_dir())
        .arg("--recursive")
        .arg("--threshold")
        .arg("1.5")
        .assert()
        .failure()
        .stderr(predicates::str::contains(
            "--threshold must be in [0.0, 1.0]",
        ));
}

#[test]
fn diversity_threshold_must_be_unit_interval() {
    bin()
        .arg(corpus_dir())
        .arg("--recursive")
        .arg("--diversity-threshold")
        .arg("1.5")
        .assert()
        .failure()
        .stderr(predicates::str::contains(
            "--diversity-threshold must be in [0.0, 1.0]",
        ));
}

#[test]
fn top_k_must_be_positive() {
    bin()
        .arg(corpus_dir())
        .arg("--recursive")
        .arg("--top-k")
        .arg("0")
        .assert()
        .failure()
        .stderr(predicates::str::contains("--top-k must be >= 1"));
}

#[test]
fn provider_must_be_known_value() {
    bin()
        .arg(corpus_dir())
        .arg("--recursive")
        .arg("--provider")
        .arg("gpu")
        .assert()
        .failure()
        .stderr(predicates::str::contains("invalid value 'gpu'"));
}
