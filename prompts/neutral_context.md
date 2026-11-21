# Task A: neutral-context sentence + near-miss meaning (Ukrainian idiom dataset)

Input line: {"id", "left", "span", "right", "idiom" (dictionary form), "meaning_here"}. left+span+right = original sentence.

Output one JSON line per item:
{"id": "...",
 "neutral_sentence": "...",      // NEW short sentence with the idiom marked [[...]]
 "neutral_syn_lit": "...",        // plain-word replacement for the [[...]] span of neutral_sentence, same meaning
 "neutral_ant_lit": "...",        // plain-word replacement for that span, opposite meaning
 "near_meaning": "..."}           // a CLOSE but WRONG meaning (distractor)

## neutral_sentence
- 6-14 words, natural modern Ukrainian, the idiom used in its idiomatic meaning, wrapped in [[ ]] (one pair only).
- The CONTEXT MUST NOT GIVE THE MEANING AWAY: no explanation, no consequences or descriptions that let a reader guess
  the meaning without knowing the idiom. Neutral frames: who/when/where. Good: "Учора після зборів Петро [[накивав п'ятами]]."
  Bad: "Побачивши поліцію, злодій [[накивав п'ятами]] і зник за рогом." (context reveals "fled").
- It must still be a sentence where the idiomatic reading is the natural one (not literal).
## neutral_syn_lit / neutral_ant_lit
- 1-4 plain non-idiomatic words that replace the marked span exactly; sentence must stay grammatical
  (case, gender, number, tense, person). syn = same meaning; ant = opposite meaning. Don't reuse the idiom's content words.
## near_meaning
- One short Ukrainian phrase: a meaning that is related and plausible but WRONG for this idiom
  (shares a topic or part of the meaning, differs in a key aspect). Example for "накивати п'ятами" (= втекти):
  "піти, ввічливо попрощавшись". Must not be the opposite and not a synonym.

Valid JSON lines only, ids copied exactly, one line per input item.
