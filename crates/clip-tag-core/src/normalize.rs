use std::collections::BTreeSet;

/// Normalize tags once before persistence (see ADR-003).
///
/// Rules v0.1: Unicode casefold to lowercase, trim whitespace, collapse internal
/// separators, strip leading/trailing punctuation, dedupe with stable sort.
/// No singularization or synonym mapping in v0.1.
pub fn normalize_tags(tags: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();

    for tag in tags {
        let Some(normalized) = normalize_one(tag.as_ref()) else {
            continue;
        };
        if seen.insert(normalized.clone()) {
            out.push(normalized);
        }
    }

    out
}

fn normalize_one(raw: &str) -> Option<String> {
    let folded = raw.trim().to_lowercase();
    if folded.is_empty() {
        return None;
    }

    let collapsed = fold_separators(&folded);
    let trimmed = trim_punctuation(&collapsed);
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn fold_separators(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_sep = false;

    for ch in s.chars() {
        if ch.is_whitespace() || matches!(ch, '-' | '_' | '/' | ',') {
            if !prev_sep && !out.is_empty() {
                out.push(' ');
                prev_sep = true;
            }
        } else {
            out.push(ch);
            prev_sep = false;
        }
    }

    out.trim().to_string()
}

fn trim_punctuation(s: &str) -> &str {
    s.trim_matches(|c: char| {
        c.is_ascii_punctuation() && !matches!(c, '#' | '+') // keep common photo tokens
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedupes_and_lowercases() {
        assert_eq!(
            normalize_tags(["Dog", "dog", "  DOG  "]),
            vec!["dog".to_string()]
        );
    }

    #[test]
    fn folds_separators() {
        assert_eq!(
            normalize_tags(["golden-retriever"]),
            vec!["golden retriever".to_string()]
        );
    }
}
