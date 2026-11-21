# Task B: full minimal-pair set for a RARE Ukrainian idiom (from a dictionary)

Input line: {"id", "idiom" (dictionary form; may contain variants/brackets), "meaning", "examples" (0-2 literary examples)}.

Output one JSON line per item with ALL keys:
{"id", "is_idiom", "sentence", "meaning_here", "syn_lit", "syn_idiom", "ant_lit", "ant_idiom", "literal_reading",
 "opposite_meaning", "near_meaning", "partial", "partial_old_word", "partial_new_word", "literal_plausible",
 "lit_sentence", "syn_lit_candidates", "neutral_sentence", "neutral_syn_lit", "neutral_ant_lit"}

- is_idiom: false if this is not really an idiom (plain collocation, grammatical phrase like "являти собою",
  literal description like "мороз бере"). Then fill the rest anyway as best you can.
- sentence: a natural Ukrainian sentence (10-25 words) using the idiom in its idiomatic meaning, idiom wrapped in [[ ]]
  (one pair). You may adapt a given example, but keep it modern-readable. Context may be informative.
- In everything below, "the span" = the [[...]] part of `sentence`; replacements go exactly in its place and must keep
  `sentence` grammatical (case, gender, number, tense, person). No extra punctuation; don't repeat neighbouring words.
- syn_lit: 1-4 plain non-idiomatic words, same meaning, not reusing the idiom's content words.
- syn_idiom: a different REAL Ukrainian idiom with the same meaning, inflected; null if none.
- ant_lit: 1-4 plain words, opposite meaning. ant_idiom: a REAL idiom with the opposite meaning, inflected; null if none.
- meaning_here / literal_reading / opposite_meaning: short Ukrainian phrases (meaning here; what the words would mean
  literally; the opposite of the meaning).
- near_meaning: close but wrong meaning (related, differs in a key aspect; not the opposite, not a synonym).
- partial: the span with exactly ONE content word replaced by a close word (synonym / same category), inflected; must NOT
  be an existing idiom. partial_old_word / partial_new_word: the changed word before / after.
- literal_plausible + lit_sentence: true only if the same words can describe a realistic literal situation; then a NEW
  12-30-word sentence with the same words used literally, wrapped in [[ ]]; else false and null.
- syn_lit_candidates: 4 plain paraphrases of the span of different lengths (1, 2, 3, 4-5 words), each fitting the span slot.
- neutral_sentence: NEW 6-14-word sentence with the idiom in [[ ]] whose context does NOT reveal the meaning
  (neutral who/when/where frame), idiomatic reading natural. neutral_syn_lit / neutral_ant_lit: plain replacements for
  that span (same / opposite meaning), grammatical.

Valid JSON lines only, ids copied exactly, one line per input item.
