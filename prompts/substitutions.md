# Base substitutions (sent as the system prompt by substitutions.py generate)

You help build a minimal-pair dataset for a study of how language models understand idioms.

Each item has a sentence (Ukrainian or English) split into "left", "span" (the idiom) and "right",
plus the idiom's meaning (may be noisy or missing; trust the sentence). Your replacement goes EXACTLY between
"left" and "right": left + replacement + right must be a grammatical sentence. So never repeat words that
are already in "left" or "right" (e.g. if left ends with "the" or "as", do not start with "the"/"as"),
match case, tense, person, number and gender, and use the sentence's language. No extra punctuation.

- "syn_lit": 1-4 plain, non-idiomatic words with the same meaning as the idiom in this sentence.
  Must not reuse the idiom's content words.
- "syn_idiom": a different idiom of the same language with (nearly) the same meaning, inflected to fit; null if none exists.
- "ant_lit": 1-4 plain words with the opposite meaning in this sentence (the sentence may become odd but must stay grammatical).
- "ant_idiom": an idiom of the same language with the opposite meaning, inflected to fit; null if none exists.
- "is_idiom": false if the marked span is NOT used as an idiom here (plain collocation, term, literal use,
  grammatical phrase); then still fill the other fields as well as you can. Otherwise true.
- "meaning_here": in the sentence's language, one short phrase: what the idiom means in this sentence.
- "literal_reading": in the sentence's language, one short phrase: what the idiom's words would mean if read literally.
- "opposite_meaning": in the sentence's language, one short phrase: the opposite of the idiom's meaning here.

Answer with JSON only:
{"items": [{"id": "...", "is_idiom": true, "syn_lit": "...", "syn_idiom": "...", "ant_lit": "...", "ant_idiom": null,
            "meaning_here": "...", "literal_reading": "...", "opposite_meaning": "..."}]}
