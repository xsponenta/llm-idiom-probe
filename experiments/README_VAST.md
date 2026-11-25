# Експерименти E1–E4 на Vast.ai

| Скрипт | Що робить | Вихід (`data/results/gpu/<model>/`) |
|---|---|---|
| `behavioral.py` | **E1**: T1/T2 без генерації. T1 — ймовірність літер A–F на наступному токені; T2 — ймовірність усього слова «так» / «ні» (teacher forcing). Питання будуються так само, як у `evaluate.py` (seed 42), тож результати порівнянні з OpenRouter | `e1.jsonl` |
| `lm_probe.py` | Один прохід по реченнях без chat-шаблону. **E2** — сюрпризал спану, 3 токенів після нього і всього речення. **E3** — косинус до ORIG (і до NEUTRAL у нейтральній рамці) по шарах, три пулінги: середнє по спану, по токенах після спану і по всьому реченню. **E4** — ознаки (середнє по спану, усі шари) для ORIG / NEUTRAL / LIT_CONTEXT. `SAVE_ACTS=1` зберігає ще й усі активації (див. `RESULTS.md`) | `lm_probe/shard_*.parquet`, `*.npz` |
| `analyze.py` | Таблиці й графіки E1–E4. E4: логістична регресія по шарах, GroupKFold за ідіомою (у тесті лише нові ідіоми) + контроль з перемішаними мітками | `data/results/gpu/summary/` |
| `run_all.sh` | Усі моделі підряд, потім `analyze.py`. Перезапуск безпечний: готове пропускається | `data/results/gpu/logs/` |

Робоча вибірка — без `flag_grammar` (8 593 варіанти).

## 1. Інстанс

- **GPU: від 48 ГБ** (A6000 48 GB, L40S, A100 40/80 GB). Моделі на 12–14B у bf16 займають 24–28 ГБ, на 24 ГБ (RTX 4090) вони не влізуть. Якщо є тільки 4090, ганяйте `MODELS="qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b gemma3-4b mamaylm-4b llama3.1-8b"`.
- **Диск: від 100 ГБ.** Кожну модель після прогону видаляємо з кешу (`CLEANUP=1`), тож одночасно на диску лежить лише одна.
- Образ: будь-який PyTorch (CUDA 12.x).

## 2. Підготовка

```bash
git clone https://github.com/xsponenta/llm-idiom-probe && cd llm-idiom-probe
pip install -r experiments/requirements.txt
mkdir -p data/processed
# з локальної машини (датасет не в git): scp -P <port> data/processed/substitutions_v1.parquet root@<ip>:~/llm-idiom-probe/data/processed/
export HF_TOKEN=hf_...        # потрібен для gemma*, llama*: спершу прийміть ліцензію на сторінці моделі на HF
```

## 3. Спершу швидкий тест (5 хв)

```bash
python experiments/behavioral.py --models qwen3-0.6b --limit-items 20
python experiments/lm_probe.py   --models qwen3-0.6b --limit-items 20
python experiments/analyze.py --tag _limit20
```

Тест пише в ті самі папки, тому перед нічним запуском видаліть його результати: `rm -rf data/results/gpu`.
Заодно подивіться в лозі, скільки секунд пішло на 20 ідіом. Повний прогін займе приблизно в 40 разів більше на кожну модель.

## 4. Нічний запуск

```bash
nohup bash experiments/run_all.sh > run.log 2>&1 &
tail -f run.log                                  # прогрес; деталі в data/results/gpu/logs/<model>.log
```

Інший список моделей: `MODELS="qwen3-1.7b gemma3-12b mamaylm-12b" nohup bash experiments/run_all.sh > run.log 2>&1 &`.
Скорочені назви моделей і їхні HF id є в `experiments/common.py` (`MODELS`); можна передати й повний HF id.

Якщо машина перезавантажилась або процес упав, запустіть ту саму команду ще раз: готові питання й шарди пропускаються.

## 5. Забрати результати

```bash
# на локальній машині
scp -P <port> -r root@<ip>:~/llm-idiom-probe/data/results/gpu ./data/results/
```

Файли `*.npz` (ознаки для E4) важать сотні МБ на модель. Якщо вони не потрібні локально, достатньо забрати `summary/`, `*.parquet` і `e1.jsonl`.

## Що в підсумку

- `e1_T1_accuracy.*`, `e1_T2_accuracy.*`: точність за умовами.
- `e1_format_mass`: яку частку ймовірності модель віддає на допустимі відповіді (якщо мала, модель «не тримає формат»).
- `e1_partial_reading`: як модель читає PARTIAL.
- `e1_T1_orig_by_source`: природні речення проти згенерованих рідкісних ідіом.
- `e2_surprisal`: сюрпризал за умовами. Окремо в `summary.txt` різниця PARTIAL − ORIG по кожній ідіомі.
- `e3_similarity.png`, `e3_<model>.csv`: косинус до ORIG по шарах. У `summary.txt` — шар, з якого SYN_LIT ближчий до оригіналу, ніж PARTIAL.
- `e4_probe.*`: ROC-AUC проби «переносне / пряме» на нових ідіомах по шарах і контроль із перемішаними мітками.
