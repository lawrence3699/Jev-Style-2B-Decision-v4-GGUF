# Jev-Style-2B-Decision-v4 — GGUF / MLX format gates (pre-declared 2026-10-28 00:10 AEDT, BEFORE any v4 format was built or scored)

**Reference.** The main repository's reference scorer (`src/macjev`: RendererV2 → `train.evaluate.score_rows`, exact
block attention) in float32 on CUDA with TF32 off and the plain PyTorch Qwen3.5 layers (no flash-linear-attention or
causal-conv1d kernels), on the released weights `chaoliangUNSW/Jev-Style-2B-Decision-v4` @ `3f2c063f`. Text only (the
vision tower is not used). Probabilities: softmax(scores / T) with the temperature of the question's type from
`readout_config.json` (choice 0.7698 / noul 0.8247 / score 0.5836).

**Release set** (the same as the 2B v3): GGUF F16 / Q8_0 / Q4_K_M and MLX bf16 / 8bit (affine, group 64).
- GGUF: the language model of the released checkpoint saved as a text-only `Qwen3_5ForCausalLM` (vision tower dropped,
  no other change), `convert_hf_to_gguf.py --no-mtp --outtype f16`, `llama-quantize` (llama.cpp 441df11f, unmodified);
  scored by `jev-score-v2` (the v3 scorer, unchanged) with all layers on a CUDA GPU (Colab), flash attention "default".
- MLX: `mlx_lm.convert` (mlx 0.32.2, mlx-lm 0.31.3) of the same text-only export + the FP32 (1 + w) norm sidecar;
  scored by the MLX runtime on this Mac (Apple M1 Max, one run).

**Fixtures** (frozen 2026-10-28 00:12 by `scripts/formats/make_fixtures.py`; our own held-out development rows
`data/eval-v3/v4_dev_cal`, text rows only, never Decision Index suite rows; sha256 in `fixtures/fixtures.stats.json`):
- gate fixture `gate_fixture_n1000.jsonl` (sha256 `0b497f05d34500a2b1244147d553716c4db8ce0dbcdf5f5067bf2319204a9345`):
  1,000 single-question requests, ≤ 4,096 tokens (max 3,942; 516,376 tokens), drawn per source; 645 choice / 316 noul /
  39 score, up to 151 options.
- base fixture `base_fixture.jsonl` (sha256 `e3ba04f6f3f62b6b29fa380c7fb3463cdd28e7652f988433e182e2cb5b4c9488`):
  70 requests / 97 questions (1,383,602 tokens): every dev text row between 4,097 and 32,768 tokens (60 requests;
  rows sharing a state are one request) + 10 code-built edge requests (255 options with a catalogue; question blocks of
  exactly 2,048 / 2,049 tokens; states of exactly 2,048 / 2,049 tokens; one input of exactly 32,768 tokens; five
  question shapes on one state; special-token and non-ASCII text; an empty state; a JSON state). 15 catalogue questions.

**Gates** (thresholds unchanged from the 2B v3, measured on the gate fixture; n = 1,000 → 0.1 pp per row):
- render check: each runtime's own renderer gives the same token ids, blocks and slots as RendererV2 on all 1,097
  fixture questions (exact);
- top-1 agreement with the reference: GGUF F16 and MLX bf16 ≥ 0.99; GGUF Q8_0 and MLX 8bit ≥ 0.98;
- |dNLL| ≤ 0.02 for GGUF F16, GGUF Q8_0, MLX bf16, MLX 8bit (dNLL = format − reference of the mean −log Σ_gold p over
  the rows with a gold set, calibrated temperatures);
- accuracy drop (reference − format, top-1 in the gold set): GGUF Q8_0 and MLX 8bit ≤ 0.3 pp; GGUF Q4_K_M ≤ 1.0 pp;
- GGUF Q4_K_M: top-1 agreement and dNLL are reported only (as for the 2B v3);
- block-mask proof on the base fixture, gate for GGUF F16 and MLX bf16 (the full-precision formats test the
  runtimes' attention): every question is closer to the block reference than to both controls of the same reference
  scorer, (a) plain causal attention over the whole input and (b) prefix-causal (state blocks as trained, causal inside
  the question part); distance = max over the question's options of |score − control score|. Reported for the other
  formats;
- state reuse: `score_many` over each base request equals scoring each question alone (max |Δscore| ≤ 1e-4).

**Recipes.** Recipe 1 (primary): stock `llama-quantize` defaults; `mlx_lm.convert` defaults + the FP32 norm sidecar.
Recipe 2 (declared fallback, used ONLY if a quantized format fails recipe 1): keep the tied embedding / readout rows at
f16 (`llama-quantize --token-embedding-type f16`; MLX: a quantization predicate that leaves `embed_tokens` unquantized).
Same gates, same fixtures. If recipe 2 also fails, that format is NOT shipped. A failure of F16 / bf16 is a runtime
defect: it is fixed and that format re-scored on the same fixtures. Gates are not relaxed after seeing results.

**Card numbers** come only from these runs; protocol and n are stated next to each number; losses go to the internal
report only.

Correction (2026-10-28 00:13 AEDT, still before any format was built or scored): the first fixture files (00:08,
sha256 0ae34ba8… / d333c4c1…) were written with sorted JSON keys, which reordered the options of choice questions and
the keys of JSON states relative to the development rows. The fixtures were rebuilt without key sorting (same rows,
same counts; the sha256 values above); the render check then matched all 1,097 questions token for token.
