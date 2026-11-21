# Task: three extra variants per Ukrainian idiom sentence (minimal-pair dataset for an idiom-understanding study)

Each input line: {"id", "left", "span", "right", "idiom" (dictionary form), "meaning_here", "syn_lit" (an existing plain paraphrase)}.
`left + span + right` is the original sentence; `span` is the idiom as it appears there.

For EVERY item output one JSON line:
{"id": "...",
 "partial": "...",                 // the span with exactly ONE content word changed
 "partial_old_word": "...", "partial_new_word": "...",
 "literal_plausible": true/false,
 "lit_sentence": "..." or null,    // new sentence, idiom words used literally, span marked [[...]]
 "syn_lit_candidates": ["...", "...", "...", "..."]}

## partial
- Copy the span and replace exactly ONE content word (noun/verb/adjective/adverb) by a close word:
  a synonym, near-synonym or same-category word (п'ятами→підборами, рахунки→борги, школу→науку, зубами→щелепами).
- All other words of the span stay exactly as they are. Inflect the new word so that left + partial + right is grammatical.
- The result must NOT be an existing idiom or a known variant of this idiom (the point is a "broken" idiom).
- partial_old_word / partial_new_word: the word as it appears in the span / in partial.

## lit_sentence
- literal_plausible = true only if the idiom's words can describe a realistic, concrete literal situation
  (e.g. "відкрити карти" in a card game, "ятрити рану" on a leg, "приносити плоди" about an apple tree).
- If true: write a NEW natural Ukrainian sentence of 12-30 words where the same words (same lemmas, same order,
  inflection may change) are used in their LITERAL meaning, and wrap them in [[ ]]. Different topic from the original.
  No idiomatic reading should be possible. Only one [[ ]] pair.
- If false: "lit_sentence": null. Proverbs, abstract or nonsensical-if-literal idioms are usually false. Do not force it.

## syn_lit_candidates
- 4 plain, NON-idiomatic paraphrases of the span with the same meaning here, each fitting grammatically between left and right.
- Different lengths: 1 word, 2 words, 3 words, 4-5 words (if a length is impossible, use the closest).
- Do not reuse the idiom's content words. Do not repeat words already at the end of left or start of right. No extra punctuation.
- Must differ from each other; may include the given syn_lit.

All text in Ukrainian. Output must be valid JSON lines, ids copied exactly, one line per input item.
