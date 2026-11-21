# UK idiom minimal pairs, v1

Датасет мінімальних пар для дослідження того, як мовні моделі обробляють українські фразеологізми.
Кожне речення містить ідіому, а варіанти замінюють лише її спан (контекст фіксований), або переносять
ідіому в інший контекст (нейтральний чи буквальний).

- **Файл:** `data/processed/substitutions_v1.parquet` (копія для перегляду: `substitutions_v1.csv`)
- **sha256:** `a05911927f7639f235f4c90eb8a62c620fdb03cd870d1974094089d3fb0ea56b`
- **Обсяг:** 824 ідіоми, 824 речення, 8 717 варіантів. Робоча вибірка без граматичних прапорців — 8 593.
- **Мова:** українська.

## Умови

| Рамка | Умова | Що це | gold | К-сть |
|---|---|---|---|---|
| original | ORIG | оригінальне речення з ідіомою | same | 824 |
| original | SYN_LIT | буквальний синонім (1–4 слова) | same | 824 |
| original | SYN_LIT_MATCHED | буквальний синонім, підібраний за к-стю токенів | same | 824 |
| original | SYN_IDIOM | інша ідіома з тим самим значенням | same | 660 |
| original | ANT_LIT | буквальний антонім | opposite | 824 |
| original | ANT_IDIOM | ідіома з протилежним значенням | opposite | 369 |
| original | RAND | випадкові слова тієї ж довжини | none | 824 |
| original | PARTIAL | ідіома з одним заміненим словом | partial (без gold) | 819 |
| neutral | NEUTRAL | нове коротке речення, контекст не підказує значення | same | 824 |
| neutral | NEUTRAL_SYN_LIT | буквальний синонім у нейтральному реченні | same | 824 |
| neutral | NEUTRAL_ANT_LIT | буквальний антонім у нейтральному реченні | opposite | 824 |
| literal | LIT_CONTEXT | ті самі слова в прямому значенні (дизайн B) | literal | 277 |

Для T2 (чи збігається значення з оригіналом) варіант порівнюється з `orig_sentence` своєї рамки.

## Джерела речень

| `source` | Речень | Походження речення | Ліцензія |
|---|---|---|---|
| simidioms | 432 | природне (приклади SimIdioms) | лише академічне використання |
| uk_wiktionary | 157 | природне (літературні приклади, спан виділений у словнику) | CC BY-SA 4.0 |
| parseme | 51 | природне (новини, спан розмічений у PARSEME 2.0) | CC BY-NC-SA 4.0 |
| uk_wiktionary_rare | 184 | згенероване (ідіома зі словника без прикладу) | CC BY-SA 4.0 (ідіома, значення) |

SimIdioms бралися лише ідіоми, які три LLM-анотатори одноголосно визнали ідіомами
(`data/labels/idiomatic/consensus.parquet`). Кандидати, позначені генератором як не-ідіоми, виключені;
дві словникові одиниці з етнічними образами виключені вручну.

## Як відтворити

```bash
# 0. сирі дані: data_first_touch.ipynb (SimIdioms, MAGPIE, Kobza 10k)
.venv/bin/python build_dataset.py                 # спани ідіом у реченнях -> data/processed/idiom_sentences.parquet
.venv/bin/python labeling.py export idiomatic     # чи це ідіома: завдання для анотаторів
.venv/bin/python agent_label.py run idiomatic     # 3 LLM-анотатори (OpenRouter)
.venv/bin/python labeling.py consensus idiomatic
.venv/bin/python idiom_sources.py                 # uk/en.wiktionary + PARSEME -> data/interim/uk_idiom_inventory.parquet
.venv/bin/python substitutions.py select          # 750 речень-кандидатів -> data/interim/substitution_sources.parquet
.venv/bin/python substitutions.py generate        # базові заміни (OpenRouter) -> data/interim/substitutions_llm.jsonl
.venv/bin/python substitutions.py build           # -> data/processed/substitutions.parquet
```

Токени для SYN_LIT_MATCHED рахуються токенізатором Qwen3 (`data/raw/tokenizers/qwen3`); кандидати збережені
в `syn_lit_candidates`, тож підбір можна перерахувати під токенізатор цільової моделі.

## Походження: усі генерації та розмітка

Увесь текстовий вміст, крім природних речень-джерел, створений мовними моделями. Використовувались два канали:

- **OpenRouter API** (платно, витрати логуються в `data/labels/spending.jsonl`; разом **$2.58**).
- **Claude Opus 5.5 у Claude Code** (агенти, що працювали з файлами напряму, без API-витрат). У даних ці
  записи мають мітку `llm-agent` у колонці `generator`, `model` або `reviewer`.

### Генерація датасету

| Крок | Модель / канал | Обсяг | Промпт | Результат | Витрати |
|---|---|---|---|---|---|
| Базові заміни: SYN_LIT, SYN_IDIOM, ANT_LIT, ANT_IDIOM, `meaning_here`, `literal_reading`, `opposite_meaning`, `is_idiom` | Claude Sonnet 5.5, OpenRouter, temperature 0.3 | 610 речень (93 запити по 10) | `prompts/substitutions.md` | `substitutions_llm.jsonl` | $1.18 |
| Ті самі поля для решти речень (OpenRouter відмовив через ліміт балансу) | Claude Opus 5.5, Claude Code (`llm-agent`) | 140 речень | `prompts/substitutions.md` | `substitutions_llm.jsonl` | — |
| PARTIAL, LIT_CONTEXT, 4 кандидати для SYN_LIT_MATCHED | Claude Opus 5.5, Claude Code | 642 речення | `prompts/extra_variants.md` | `substitutions_extra.jsonl` | — |
| Нейтральні речення + NEUTRAL_SYN_LIT / NEUTRAL_ANT_LIT, `near_meaning` | Claude Opus 5.5, Claude Code | 640 речень | `prompts/neutral_context.md` | `substitutions_neutral.jsonl` | — |
| Рідкісні ідіоми: речення і повний набір умов | Claude Opus 5.5, Claude Code | 233 ідіоми → 184 у датасеті | `prompts/rare_idioms.md` | `substitutions_rare.jsonl` | — |
| Граматична перевірка і мінімальні виправлення (окремі агенти, що не писали ці дані) | Claude Opus 5.5, Claude Code | 7 253 варіанти | `prompts/grammar_review.md` | `grammar_review.jsonl` | — |
| RAND | без LLM: випадкові слова з речень датасету, seed 42 | 824 | — | — | — |

Колонка `generator` у датасеті: `anthropic/claude-sonnet-5.5` — 2 975 варіантів, `llm-agent` (Claude Opus 5.5) — 5 742.

### Розмітка

| Завдання | Хто розмічав | Обсяг | Результат | Витрати |
|---|---|---|---|---|
| `idiomatic`: чи є вираз ідіомою (відбір ідіом SimIdioms) | 3 LLM-анотатори через OpenRouter: Claude Haiku 5.5, GPT-6 Luna, Gemini 3.1 Flash Lite (temperature 0) | 1 740 ідіом; одностайно 1 448 (1 210 «ідіома», 238 «ні»), спірні 292; κ між моделями 0.61–0.76 | `data/labels/idiomatic/` | $0.30 |
| `usage`: як вжито ідіому в реченні (Kobza, лематичні збіги; у v1 не використано) | ті самі 3 моделі | 4 155 речень; одностайно 2 772, спірні 1 383; κ 0.35–0.48 | `data/labels/usage/` | $0.75 |
| `pair`: еквівалентність пар UK–EN (у v1 не використано, видалено) | ті самі 3 моделі | 281 пара | — | $0.06 |
| `subst_check`: чи коректний варіант для своєї умови | GPT-6 Luna (OpenRouter) | 4 026 варіантів; відкинуто як ненадійний (κ = 0.13 з людиною) | `data/labels/subst_check/gpt-6-luna.jsonl` | $0.20 |
| `subst_check`, ручна сліпа перевірка | автор | 253 варіанти з 40 випадкових речень | `data/labels/subst_check/human.jsonl` | — |
| `kobza`: вживання ідіом у корпусі (на потім) | автор | 60 з 500 | `data/labels/kobza/human.jsonl` | — |

### Пілоти (у датасет не пішли)

| Що | Моделі (OpenRouter) | Обсяг | Витрати |
|---|---|---|---|
| Порівняння генераторів речень | Gemini 3 Flash, Claude Sonnet 5.5, DeepSeek V4 Pro | 8 ідіом | $0.06 |
| Пробна оцінка моделей (T1/T2) на 20 реченнях | GPT-6 Luna, DeepSeek V4.1 Flash, Gemma 3 4B, Llama 3.1 8B, Ministral 3B, Mistral Nemo | 198 запитань на модель | $0.04 |

## Контроль якості

- **Граматика:** окремі агенти Claude Opus 5.5, які не створювали цих даних, прочитали 7 253 варіанти
  (усі, крім RAND і природних оригіналів).
  Проблеми знайдено в 452 (6.2%): 328 мінімально виправлено, 124 позначено `flag_grammar` — їх виключати.
  Найслабші умови: SYN_IDIOM (22% проблем до виправлення), ANT_IDIOM (17%), ANT_LIT (13%), SYN_LIT (11%).
- **Ручна перевірка:** 253 варіанти з 40 випадкових речень (`data/labels/subst_check/human.jsonl`):
  246 коректні, 7 з проблемами. LLM-перевіряльник знайшов 6 із цих 7 і ще 14 справжніх помилок,
  пропущених при ручній розмітці.
- **LLM-суддя GPT-6 Luna** на тих самих даних виявився ненадійним (κ = 0.13 з людиною, 10% його «проблем»
  підтверджено), тому його вердикти не використовуються.
- **Автоматичні прапорці** `flag_*`: `span_by_lemma_match` (межі спану з лематичного пошуку),
  `token_gap_ge3`, `duplicate` (SYN_LIT_MATCHED збігся з SYN_LIT), `doubtful` (сумнівні за самооцінкою
  LLM-проходу) та інші; переважно м'які, не обов'язково помилки.

## Основні колонки

`item_id`, `variant_id`, `source`, `idiom`, `meaning` (словникове), `meaning_here` (у цьому реченні),
`literal_reading`, `opposite_meaning`, `near_meaning`, `condition`, `gold`, `frame`, `sentence`,
`span_start`/`span_end`, `replacement`, `orig_sentence`, `sentence_origin`, `n_tok_span`/`n_tok_idiom`,
`generator`, `grammar_ok`, `grammar_note`, `flag_*`.

## Обмеження

- Ідіоми-синоніми й антоніми часто відсутні (немає справжньої ідіоми) або наближені.
- Нейтральні й буквальні речення та речення для рідкісних ідіом згенеровані, а не взяті з корпусу.
- `meaning` із SimIdioms місцями хибне; для оцінки використовується `meaning_here`.
- Українська частина SimIdioms — лише для академічних досліджень; дані не публікуються в git.
