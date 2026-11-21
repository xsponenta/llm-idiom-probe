# Grammar review of a Ukrainian idiom minimal-pair dataset

Each input line is one group: {"item_id", "idiom", "meaning_here", "variants": [{"id", "condition", "expect", "sentence"}]}.
In each sentence the changed / tested part is marked [[...]]. You are a careful native-level Ukrainian editor.

For EVERY variant output one JSON line: {"id": "...", "ok": true/false, "fix": "..." or null, "note": "..." or null}

ok = false when the sentence (with the [[...]] part, read without the brackets) has a GRAMMAR problem caused by the
marked part: wrong case / case government (e.g. "не дав всі гроші" -> "усіх грошей"), gender/number/person agreement,
verb aspect or tense clash with the rest of the sentence, wrong preposition, wrong word form or misspelling,
broken word order, missing/duplicated word at the junction with the context.
Also ok = false if the marked part clearly does not match "expect" (e.g. an "antonym" that is not opposite at all,
a "synonym" that changes the meaning) — note "meaning: ...".

Do NOT flag:
- stylistic awkwardness, mild oddness of antonym sentences ("he generously forgave" may sound odd but is grammatical);
- archaic / dialect / literary forms that come from the original text;
- "собака" with feminine agreement (colloquially fine); PARTIAL being a non-existent idiom (that is intended).

fix: ONLY for conditions SYN_LIT, SYN_LIT_MATCHED, SYN_IDIOM, ANT_LIT, ANT_IDIOM, PARTIAL, NEUTRAL_SYN_LIT,
NEUTRAL_ANT_LIT: give the corrected text of the [[...]] part only (minimal change, same meaning). Use null if the
problem is outside the brackets or cannot be fixed minimally. For ORIG, NEUTRAL, LIT_CONTEXT always fix = null.
note: very short reason in Ukrainian when ok=false, else null.

Be strict about grammar and precise: do not invent problems. Valid JSON lines only, one per variant id.
