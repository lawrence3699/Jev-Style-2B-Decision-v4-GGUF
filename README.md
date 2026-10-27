---
license: apache-2.0
base_model: chaoliangUNSW/Jev-Style-2B-Decision-v4
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-classification
tags:
  - decision-model
  - decision-making
  - jev-style
  - system-one
  - calibration
  - long-context
  - qwen3.5
  - gguf
  - llama.cpp
  - on-device
  - llm-routing
  - guardrails
---

# Jev-Style-2B-Decision-v4-GGUF

**Website:** [jevstyle.com](https://jevstyle.com) · **GitHub:** [jev-style](https://github.com/lawrence3699/jev-style)

**Jev-Style decision series:** [v1 · 2B](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF) → [v2 · 2B](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2-GGUF) → [v3 · 0.8B](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF) → [v3 · 2B](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF) → **v4 · 2B**

**Jev-style decisions for llama.cpp.** These are the GGUF builds of
[Jev-Style-2B-Decision-v4](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4) for text and JSON states:
F16, Q8_0 and a **1.27 GB** Q4_K_M, with a small bundled scorer (`jev-score-v2`) that reads the decision at one
verdict slot per option. Images need the main repository's PyTorch runtime. Full results, protocols, training data
and licences are on the [main model card](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4).

| | **Jev-Style 2B v4** | Jev-Style 2B v3 |
|---|---:|---:|
| Decision Index 0.3 public index (self-run, complete) | **44.37** | 34.05 |
| JevBench v1.4.1, public items (231) | **74.9%** | 73.6% |
| Context per call | **32,768 tokens** | 25,600 tokens |

<sub>Results of the model, measured with the main repository's PyTorch runtime: Decision Index public suite 0.3
with the kit's own runner and scorer (bfloat16 on one A100, one complete run of 140,178 requests; v3: its public
numbers on the 0.4.1 board); JevBench v1.4.1 public items with the official harness (float32 on Apple MPS, self-run
once, 95% CI 68.9–80.0%; v3: the number on its own card). Neither is an official board entry. These GGUF files were not re-run on the benchmarks;
they reproduce that runtime's scores within the pre-declared gates in the Files table. Protocols:
[main card](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4#results).</sub>

## Files

| File | Quantization | Size | Same top-1 as PyTorch FP32 (1,000 questions) | Max abs Δp | Accuracy (FP32: 84.4%) | Long fixture (97 questions) | Gate |
|---|---|---:|---:|---:|---:|---:|:---:|
| `Jev-Style-2B-Decision-v4-F16.gguf` | F16 | 3.78 GB | 99.7% | 0.029 | 84.5% | 97 / 97 | PASS |
| `Jev-Style-2B-Decision-v4-Q8_0.gguf` | Q8_0 | 2.01 GB | 99.6% | 0.062 | 84.3% | 97 / 97 | PASS |
| `Jev-Style-2B-Decision-v4-Q4_K_M.gguf` | Q4_K_M | 1.27 GB | 97.4% | 0.321 | 83.6% | 95 / 97 | PASS |

- **For long documents use Q8_0 or F16.** Q4_K_M is noticeably noisier on long inputs.
- F16 is the runtime's default.

<sub>Reference: the main repository's reference scorer in float32 on CUDA (TF32 off, plain PyTorch Qwen3.5 layers),
exact block attention, on the released weights, text only; gates declared before any format was built or scored.
Gate fixture: 1,000 questions from our own held-out development rows (≤ 4,096 tokens; 645 choice, 316 true/false,
39 score; up to 151 options), accuracy = top-1 in the gold set. Long fixture: 70 requests / 97 questions,
development rows of 4,097 to 32,768 tokens plus edge cases (255 options, 15 catalogue questions, an input of
exactly 32,768 tokens); the column counts questions with the same top-1 as the reference. On the long fixture the scores of every question (F16 and Q8_0,
97 / 97) are closer to the block-attention reference than to plain-causal and prefix-causal controls of the same
scorer, so the files run the attention the model was trained with, and `score_many` gave exactly the scores of each
question asked alone. For 4-bit the pre-declared gate is the accuracy drop (≤ 1.0 point); top-1 agreement is
reported only. Scored with `jev-score-v2` on an NVIDIA L4 (CUDA, all layers on the GPU); the scorer is the 2B v3
one, unchanged (checked on Metal for the 2B v3; these v4 files were checked on CUDA only). Converted and quantised
with unmodified llama.cpp tools at commit 441df11f. Sizes are the files (GB = 10^9 bytes). Records:
`validation/parity/`.</sub>

## Speed

**Read once, then ask.** On an NVIDIA L4 (GGUF F16), a single question about an input of 16,385–32,768 tokens (median 21,646) took a median 3.01 s; the same request again, with the state already computed, took 54 ms.

| Input (state + question + options) | Requests | Median input | F16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|---:|---:|
| up to 1,024 tokens | 871 | 250 tokens | 58 ms | 50 ms | 48 ms |
| 1,025–4,096 tokens | 129 | 1,437 tokens | 199 ms | 188 ms | 195 ms |
| 4,097–16,384 tokens | 19 | 4,728 tokens | 643 ms | 600 ms | 619 ms |
| 16,385–32,768 tokens | 25 | 21,646 tokens | 3.01 s | 2.82 s | 2.90 s |
| 16,385–32,768 tokens, state already computed | 25 | 21,646 tokens | 54 ms | 47 ms | 45 ms |

<sub>NVIDIA L4 (Colab), CUDA 13.0, llama.cpp 441df11f, all layers on the GPU, flash attention auto; nothing else ran on the GPU. Wall time of one scoring call (the `jev-score-v2` request and the readout; the input was rendered and the model loaded beforehand), the state computed from scratch, once per fixture request: the 1,000 gate-fixture requests and the single-question requests of the long fixture (Files table). 'State already computed' = the same request again right after, its state still cached in `jev-score-v2`; for every long-fixture request and file the scores were identical to the first call's. The three files were timed one after another on the same VM.</sub>

## Quick start

The files are standard Qwen3.5 text models, so llama.cpp loads them, but chat or text generation does **not** give
you the model's decisions: this model needs block attention and a readout at one verdict slot per option. The
bundled `jev_score_v2.cpp` (a small program on stock libllama) does exactly that. Use this repository's runtime,
tokenizer and GGUF files together: the runtime refuses the 2B v3 and 0.8B v3 GGUFs.

```bash
pip install -U huggingface_hub
hf download chaoliangUNSW/Jev-Style-2B-Decision-v4-GGUF --local-dir jev-v4-2b-gguf
cd jev-v4-2b-gguf
pip install -r requirements.txt                 # tokenizers 0.23.3, numpy 2.5.4

# Build the scorer once against a llama.cpp checkout (tested at commit 441df11f65ea0b6d0c72965aaf70c8241070ddcb).
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout 441df11f65ea0b6d0c72965aaf70c8241070ddcb
sh build_jev_score.sh llama.cpp                 # Metal on macOS; NVIDIA: LLAMA_CMAKE_FLAGS="-DGGML_CUDA=ON" sh build_jev_score.sh llama.cpp
                                                # -> build/jev-score-v2 (or pass --jev-score PATH / set JEV_SCORE_V2_BIN)
```

One question (`--quant` picks the file: `F16` default, `Q8_0`, `Q4_K_M`):

```sh
python jev_style_decision_gguf.py \
  --state "The user said: 'Write a reply to Bob with the Q3 report attached, but let me check it before it goes out.' The draft is written and the report is attached." \
  --question "What should the assistant do next?" \
  --options '{"send": "press Send now", "ask_user": "ask the user to review the draft", "discard": "delete the draft"}'
# -> "answer": "ask_user", probability 0.728 (send 0.265, discard 0.008)
```

A state can be JSON (`--state-json`); with no `--options` the question is true/false; `--qtype score` takes a list
of 2–10 levels, lowest first:

```sh
python jev_style_decision_gguf.py --quant Q8_0 \
  --state-json '{"app": "Calendar", "event": "Team sync", "start": "2026-10-02 15:00", "invitees": 14, "accepted": 3}' \
  --question "Is attendance for this meeting likely to be low?"
# -> "answer": "true", probability 0.906
python jev_style_decision_gguf.py --quant Q4_K_M \
  --state "Ticket: the checkout page returns HTTP 500 for every customer since the last deploy." \
  --question "How urgent is this ticket?" --qtype score --options '["not urgent", "normal", "urgent", "critical"]'
# -> "answer": "3" (the level index of "critical"), probability 0.862
```

The output has `answer`, `probabilities` (softmax of the scores at the calibrated temperature of the question type:
choice 0.770, true/false 0.825, score 0.584; `--temperature 1.0` gives uncalibrated probabilities), the raw `scores`
(logit(" yes") − logit(" no") per option), token counts and `catalogue_overflow`.

Many questions, as JSON lines (consecutive lines with the same state compute that state once; a malformed or
over-budget line gets an `"error"` field, nothing is truncated):

```sh
cat > requests.jsonl <<'EOF'
{"id": "q1", "state": "Order #4411: payment received. Items are still in the warehouse waiting to be packed.", "question": "Has the order shipped?"}
{"id": "q2", "state": "Order #4411: payment received. Items are still in the warehouse waiting to be packed.", "question": {"t": "choice", "ins": "What is the order status?", "crit": {"awaiting_payment": null, "processing": "paid, not yet handed to the courier", "shipped": "handed to the courier"}}}
{"id": "q3", "state": "Hi, I was charged twice for my subscription this month.", "question": "Which intent is this?", "options": ["billing_question", "cancel_subscription", "technical_issue"]}
EOF
python jev_style_decision_gguf.py --quant Q8_0 --jsonl requests.jsonl
# -> one JSON object per line: q1 "false" 0.983, q2 "processing" 0.996, q3 "billing_question" 0.996
```

From Python:

```python
from jev_style_decision_gguf import JevStyleDecisionGGUF

state = {"app": "Mail", "draft_to": "bob@example.com", "attachment": "Q3-report.pdf", "sent": False}
with JevStyleDecisionGGUF(quant="Q8_0") as model:
    one = model.decide(state, "Is a report attached to the draft?")
    many = model.score_many(state, [
        {"t": "noul", "ins": "Has the email been sent?", "crit": None},
        {"t": "choice", "ins": "What is attached?", "crit": {"report": "a report", "photo": "a photo", "nothing": None}},
    ])
print(one["answer"], round(one["top_probability"], 3))   # true 0.889
print([r["answer"] for r in many])                       # ['false', 'report']
```

<sub>Outputs of the commands above as written, run once in a fresh folder laid out like this repository: a new Python 3.12 environment with only `requirements.txt`, and `jev-score-v2` built by the command above from a fresh llama.cpp checkout (Colab VM, NVIDIA L4, CUDA); rounded. On the CPU (`--ngl 0`, Q8_0) the first example gives ask_user 0.724.</sub>

- `score_many(state, questions)` computes the state once and reuses it for every question; the results match
  calling `decide` per question (on the long fixture of the Files table the scores were identical).
- `--ngl 0` runs on the CPU, with flash attention off. `--verify` checks the sha256 of the tokenizer,
  `readout_config.json` and the selected GGUF against `manifest.json` before loading.
- The runtime checks the GGUF metadata and refuses GGUFs of a different architecture, size or name (for example the
  2B v3 and 0.8B v3 files).
- The input format, block attention and the readout are described on the
  [main card](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4#input-format-and-readout).

### With the jev-style package

[jev-style](https://github.com/lawrence3699/jev-style) 0.6.0 or later downloads this repository at a pinned revision
and runs this runtime behind a local systemone-compatible API, with a Playground, demos and MCP tools. It uses the
`jev-score-v2` binary built above:

```bash
pip install jev-style
jev-style serve --release 2b-v4 --backend gguf --quant Q8_0 --scorer /absolute/path/to/build/jev-score-v2
```

```python
from jev_style import JevStyle, noul

js = JevStyle.from_pretrained("chaoliangUNSW/Jev-Style-2B-Decision-v4-GGUF", quant="Q8_0",
                              scorer="/absolute/path/to/build/jev-score-v2")
out = js.decide({"app": "Calendar", "event": "Team sync", "start": "2026-10-02 15:00", "invitees": 14, "accepted": 3},
                {"low": noul("Is attendance for this meeting likely to be low?")})
out["answers"]["low"]["noul"]        # 0.906, as with the command line above
```

## Scope and limits

- **Runtime required.** Stock llama.cpp, Ollama and LM Studio can load these files but cannot produce the decision
  scores, and they would run causal attention. Use `jev_style_decision_gguf.py` with `jev-score-v2`.
- **Text only.** These files hold the language model; images in the state need the PyTorch runtime of the
  [main repository](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4).
- **32,768 tokens** is the limit for the whole input (state + question + options + readout); larger inputs raise
  `InputBudgetError`, nothing is truncated. At most 255 options per question.
- **Decision Index.** The training data includes the official train splits of several benchmarks that the index
  tests on, so results on those benchmarks are not zero-shot; they are named on the
  [main card](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4#training-data-and-licences).
- **Decisions only.** The model scores the options you give it. It does not generate text and takes no actions.

## Disclaimers and licence

Apache-2.0. Built on Qwen/Qwen3.5-2B (Apache-2.0); `NOTICE` lists the modifications. **Soft labels distilled from
TypeSafe Jev 1.14 via its public API** are the targets of part of the training rows; no Jev weights or code are
included. Some training data has terms of its own and some training rows were written by Anthropic Claude and OpenAI
GPT models, whose providers' terms of use may restrict how models trained on them may be used: see [Training data
and licences](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4#training-data-and-licences) on the main
card. `jev_score_v2.cpp` is compiled against libllama (MIT); llama.cpp is not redistributed here. Not affiliated
with, endorsed by or connected to TypeSafe AI or Jev, the Laya authors or the Qwen team.

**AI disclosure:** code written with AI coding assistants (Claude Code) under my direction; I designed the project,
trained the models and verified the results.

## Contact

I welcome internship, employment, and research collaboration opportunities. Please contact me at [**yanchaoliang369@gmail.com**](mailto:yanchaoliang369@gmail.com).

欢迎提供实习、工作及科研合作机会，请邮件联系：[yanchaoliang369@gmail.com](mailto:yanchaoliang369@gmail.com)。
