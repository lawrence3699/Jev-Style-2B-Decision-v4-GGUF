"""Jev-Style-2B-Decision-v4: typed decisions with llama.cpp (GGUF F16 / Q8_0 / Q4_K_M), text only.

Self-contained runtime for chaoliangUNSW/Jev-Style-2B-Decision-v4-GGUF (Apache-2.0). No dependency on any training
code: rendering (render v2 + attention blocks), verdict readout and calibration are implemented below and
reproduce the reference implementation used for evaluation (see release_config.json -> "runtime_parity").
Needs numpy, tokenizers and the jev-score-v2 binary (build it with `sh build_jev_score.sh /path/to/llama.cpp`).

    python jev_style_decision_gguf.py --state "..." --question "Which option?" --options '["a", "b"]'

    from jev_style_decision_gguf import JevStyleDecisionGGUF
    with JevStyleDecisionGGUF(quant="Q8_0") as m:
        m.decide(state, "Is the task finished?")                       # true/false question
        m.score_many(state, [q1, q2, q3])                              # one state, computed once

Probabilities use the calibrated temperature of the question's type (readout_config.json ->
temperatures.by_qtype: choice 0.770, noul 0.825, score 0.584) unless temperature=... is given
(1.0 = uncalibrated scores).

Text only: images in the state need the main repository's PyTorch runtime. The input protocol is the one of the
2B v3 (render v2 + block attention) and jev-score-v2 is the v3 scorer, unchanged; the v4 runtime, tokenizer and
GGUF files must be used together (the v3 GGUFs have the same architecture and are refused by name).
"""
# ----------------------------------------------------------------------------------------------
# Shared core (byte-identical in jev_style_decision_gguf.py and jev_style_decision_mlx.py): input rendering (v2),
# verdict readout, calibration, budgets, errors, manifest check, CLI/JSONL. The main repository's PyTorch runtime calls
# the reference code instead; this renderer gives the same token ids, blocks and slots as that code for text
# (validation/runtime/render_check.json).
#
# Input layout ("macjev-render-v2-long-options", layout "sb"; token segments are encoded separately
# and concatenated, so slot positions are exact):
#
#   state prefix     State:\n<state>\n\n
#   short form       Question [<type>]: <question>\nOptions:\n
#   (question +      - <option 1> ->\n ... - <option K> ->\n
#    options + slots <= 2,048 tokens)
#   overflow form    Question [<type>]: <question>\nOptions:\n
#   (otherwise)      Option 1: <option 1>\n ... Option K: <option K>\n            (the catalogue)
#                    Judge each numbered option in the complete catalogue above:\n
#                    Option 1 ->\n ... Option K ->\n                               (the rubric)
#
# Attention blocks ([start, stop) token ranges): the state is cut into consecutive 2,048-token
# blocks; the short form is one block; in the overflow form the catalogue and the rubric are each cut
# into 2,048-token blocks. In the 6 full-attention layers every block attends to all earlier tokens
# and to itself with NO causal mask inside the block (block-causal); the Gated-DeltaNet layers are
# ordinary recurrent layers. A backend must compute exactly these blocks (never merged, never re-split).
#
# Score of option k = logit(" yes") - logit(" no") at its " ->" slot = h_slot . (W_yes - W_no) in
# float32 (final normed hidden state, tied embedding rows). Probabilities = softmax(scores / T) in
# canonical option order. T = the temperature of the question's type in readout_config.json
# (temperatures.by_qtype: choice / noul / score, fitted on 3,325 calibration rows) unless temperature=...
# overrides it (1.0 = uncalibrated scores). There are no category / group temperatures and no separate
# question/options budget, so the 0.8B v3 runtime's --category and --head-max options do not exist here.
#
# Budget: the complete input (state + question + options + readout) <= 32,768 tokens. Larger inputs
# raise InputBudgetError; nothing is ever truncated. Text inside the state, question and options is
# tokenised with special tokens disabled, so e.g. "<|im_end|>" in user text can never act as a
# control token. A non-finite score (NaN / inf) raises NonFiniteScoreError: no probabilities are made.
#
# Text only: this runtime does not read images ("<|image_1|>" in a state is plain text here). For images in the
# state use the main repository's PyTorch runtime (jev_style_decision.py).
# ----------------------------------------------------------------------------------------------
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

MODEL_NAME = "Jev-Style-2B-Decision-v4"
TEMPLATE_VERSION = "macjev-render-v2-long-options"
READOUT_FORMAT = "macjev-readout-v2"
LAYOUT = "sb"
BLOCK = 2048                    # attention block size (a processing unit, not a content limit)
CONTEXT_LIMIT = 32_768          # state + question + options + readout, all included
QTYPES = ("choice", "score", "noul")
JSONL_GROUP_MAX = 256           # consecutive JSONL rows with one state scored in one backend call
HERE = Path(__file__).resolve().parent


class InputBudgetError(ValueError):
    """The rendered input exceeds the token budget. Nothing was truncated."""


class QuestionError(ValueError):
    """The question/options are malformed."""


class NonFiniteScoreError(FloatingPointError):
    """The model produced a non-finite decision score (NaN or inf). No probabilities are returned."""


# -- questions ------------------------------------------------------------------------------------
def option_names(question):
    """Canonical option identifiers, in the order the probabilities are returned."""
    if not isinstance(question, dict):
        raise QuestionError("question must be a dict {'t', 'ins', 'crit'}")
    t, crit = question.get("t"), question.get("crit")
    if not isinstance(question.get("ins"), str) or not question["ins"].strip():
        raise QuestionError("question text ('ins') must be a non-empty string")
    if t == "choice":
        if not isinstance(crit, dict) or not crit:
            raise QuestionError("choice needs a non-empty dict {option name: description or None}")
        if len(crit) > 255:
            raise QuestionError(f"choice supports at most 255 options (got {len(crit)})")
        return [str(k) for k in crit]
    if t == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise QuestionError("score needs a list of 2..10 level descriptions")
        return [str(i) for i in range(len(crit))]
    if t == "noul":
        if crit is not None and not isinstance(crit, dict):
            raise QuestionError("noul criteria must be None or {'false': ..., 'true': ...}")
        return ["false", "true"]
    raise QuestionError(f"unknown question type {t!r} (expected one of {QTYPES})")


def make_question(question, options=None, qtype=None):
    """Build a typed question.

    * ``question`` already a dict {"t", "ins", "crit"}: validated and returned.
    * ``qtype="choice"`` (default when ``options`` is given): ``options`` = {name: description or None}
      or a list of names.
    * ``qtype="score"``: ``options`` = list of 2..10 level descriptions (level 0 first).
    * ``qtype="noul"`` (default when no options): a true/false statement; ``options`` may be
      {"false": "...", "true": "..."} to describe the two outcomes.
    """
    if isinstance(question, dict):
        q = dict(question)
    else:
        if qtype is None:
            qtype = "choice" if options is not None else "noul"
        if qtype == "choice":
            if isinstance(options, (list, tuple)):
                if len(set(map(str, options))) != len(options):
                    raise QuestionError("duplicate option names")
                crit = {str(o): None for o in options}
            else:
                crit = options
        elif qtype == "score":
            crit = list(options) if options is not None else None
        else:
            crit = options
        q = {"t": qtype, "ins": question, "crit": crit}
    option_names(q)
    return q


def serialize_state(state):
    """Strings pass through unchanged; any other JSON value is serialised (ensure_ascii=False)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def _criterion(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(question):
    t, crit = question["t"], question.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else f"{k}: {_criterion(v)}" for k, v in crit.items()]
    if t == "score":
        return [f"level {i}: {_criterion(c)}" for i, c in enumerate(crit)]
    crit = crit or {}
    false_c, true_c = crit.get("false"), crit.get("true")
    return ["false: " + (_criterion(false_c) if false_c not in (None, "") else "no, the statement does not hold"),
            "true: " + (_criterion(true_c) if true_c not in (None, "") else "yes, the statement holds")]


# -- tokenizer + renderer -----------------------------------------------------------------------
class TextEncoder:
    """HF ``tokenizers`` tokenizer.json; no BOS/EOS, special tokens in text are split (never control tokens)."""

    def __init__(self, tokenizer_json):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(tokenizer_json))
        self.tk.encode_special_tokens = True

    def __call__(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def id_to_token(self, i):
        return self.tk.id_to_token(int(i))

    def vocab_size(self):
        return self.tk.get_vocab_size(with_added_tokens=True)


def _blocks(start, stop):
    return [(s, min(s + BLOCK, stop)) for s in range(start, stop, BLOCK)]


class Rendered:
    """One rendered question: token ids, the verdict slots (one per option, canonical order), the
    attention blocks ([start, stop) pairs tiling [0, len(ids))) and the state prefix length."""
    __slots__ = ("ids", "prefix_len", "slots", "blocks", "names", "qtype", "catalogue_overflow")

    def __init__(self, ids, prefix_len, slots, blocks, names, qtype, catalogue_overflow):
        self.ids, self.prefix_len, self.slots, self.blocks = ids, prefix_len, slots, blocks
        self.names, self.qtype, self.catalogue_overflow = names, qtype, catalogue_overflow

    @property
    def state_blocks(self):
        return [b for b in self.blocks if b[1] <= self.prefix_len]

    @property
    def question_blocks(self):
        return [b for b in self.blocks if b[0] >= self.prefix_len]

    @property
    def input_tokens(self):
        return len(self.ids)

    @property
    def head_tokens(self):
        """question + options + readout tokens (everything after the state prefix)"""
        return len(self.ids) - self.prefix_len


class Renderer:
    def __init__(self, encode, readout_cfg, max_len=CONTEXT_LIMIT):
        want = {"format": READOUT_FORMAT, "template": TEMPLATE_VERSION, "layout": LAYOUT, "readout": "verdict",
                "block_size": BLOCK, "total_context_limit": CONTEXT_LIMIT}
        bad = {k: readout_cfg.get(k) for k, v in want.items() if readout_cfg.get(k) != v}
        if bad:
            raise ValueError(f"readout_config.json does not describe this runtime's protocol {want}; got {bad}")
        if not 0 < int(max_len) <= CONTEXT_LIMIT:
            raise ValueError(f"max_len must be in 1..{CONTEXT_LIMIT}")
        self.enc, self.max_len = encode, int(max_len)
        self.head_max = self.max_len    # no separate question/options cap (field kept for the 0.8B / jev-style API)
        st = readout_cfg["slot_tokens"]
        self.yes, self.no, arrow = int(st["yes"]["id"]), int(st["no"]["id"]), int(st["verdict_slot"]["id"])
        for text, want_id in ((" yes", self.yes), (" no", self.no), (" ->", arrow)):
            got = self.enc(text)
            if got != [want_id]:
                raise ValueError(f"tokenizer mismatch: {text!r} -> {got}, readout_config expects [{want_id}]")
        self.arrow = [arrow]
        self.newline = self.enc("\n")
        self.dash = self.enc("- ")
        self.rubric_head = self.enc("Judge each numbered option in the complete catalogue above:\n")
        self._numbered = {}

    def _option_label(self, pos, catalogue):
        key = (pos, catalogue)
        if key not in self._numbered:
            self._numbered[key] = self.enc(f"Option {pos + 1}: " if catalogue else f"Option {pos + 1}")
        return self._numbered[key]

    def prefix_ids(self, state):
        return self.enc("State:\n") + self.enc(serialize_state(state)) + self.enc("\n\n")

    def pieces(self, state, question, prefix=None):
        """Tokenised segments of one question (``prefix``: already tokenised state, to tokenise it once)."""
        names = option_names(question)
        return {"prefix": self.prefix_ids(state) if prefix is None else list(prefix),
                "head": self.enc(f"Question [{question['t']}]: {question['ins']}\nOptions:\n"),
                "opts": [self.enc(o) for o in render_options(question)], "names": names, "qtype": question["t"]}

    def assemble(self, pieces, max_len=None):
        maximum = self.max_len if max_len is None else min(self.max_len, int(max_len))
        prefix, head, opts = list(pieces["prefix"]), list(pieces["head"]), pieces["opts"]
        k = len(opts)
        if k < 1:
            raise QuestionError("at least one option is required")
        short, rel = list(head), []
        for o in opts:
            short += self.dash + list(o) + self.arrow
            rel.append(len(short) - 1)
            short += self.newline
        overflow = len(short) > BLOCK
        if not overflow:
            ids = prefix + short
            slots = [len(prefix) + s for s in rel]
            blocks = _blocks(0, len(prefix)) + [(len(prefix), len(ids))]
        else:
            catalogue = list(head)
            for pos, o in enumerate(opts):
                catalogue += self._option_label(pos, True) + list(o) + self.newline
            doc_end = len(prefix) + len(catalogue)
            rubric, rel = list(self.rubric_head), []
            for pos in range(k):
                rubric += self._option_label(pos, False) + self.arrow
                rel.append(len(rubric) - 1)
                rubric += self.newline
            ids = prefix + catalogue + rubric
            slots = [doc_end + s for s in rel]
            blocks = _blocks(0, len(prefix)) + _blocks(len(prefix), doc_end) + _blocks(doc_end, len(ids))
        if len(ids) > maximum:
            raise InputBudgetError(f"the complete input needs {len(ids)} tokens (state {len(prefix)} + question/"
                                   f"options/readout {len(ids) - len(prefix)}); the limit is {maximum} tokens (model "
                                   f"maximum {CONTEXT_LIMIT}). Nothing was truncated: shorten the state, the "
                                   f"question or the options.")
        return Rendered(ids, len(prefix), slots, blocks, list(pieces["names"]), pieces["qtype"], overflow)

    def render(self, state, question, max_len=None, prefix=None):
        return self.assemble(self.pieces(state, question, prefix), max_len)


# -- calibration ----------------------------------------------------------------------------------
def check_temperature(t):
    try:
        t = float(t)
    except (TypeError, ValueError):
        raise ValueError(f"temperature must be a number, got {t!r}") from None
    if not (math.isfinite(t) and t > 0):
        raise ValueError(f"temperature must be finite and > 0, got {t!r}")
    return t


def softmax_probabilities(scores, temperature):
    """softmax(scores / T) in float64 (the reference computation)."""
    z = np.asarray(scores, float) / float(temperature)
    z = np.exp(z - z.max())
    return z / z.sum()


def concentration(p):
    k = len(p)
    if k < 2:
        return 1.0
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


NOT_VERIFIED = ("README.md", "eval_results.json")          # documentation / records: in manifest.json, not checked
NOT_VERIFIED_DIRS = ("assets/", "figures/", "validation/")


def verify_manifest(model_dir, only=None):
    """Re-hash the files listed in manifest.json (all, or those whose path starts with one of ``only``).
    Documentation and evaluation records (README.md, eval_results.json, assets/, figures/, validation/) are
    recorded in the manifest but not checked here (not even when named in ``only``), so a card edit never
    makes the runtime refuse to load and a download without them still verifies."""
    model_dir = Path(model_dir)
    man = json.loads((model_dir / "manifest.json").read_text())
    bad, missing, checked = [], [], 0
    for name, rec in man["files"].items():
        if name in NOT_VERIFIED or name.startswith(NOT_VERIFIED_DIRS):
            continue
        if only and not any(name == o or name.startswith(o.rstrip("/") + "/") for o in only):
            continue
        p = model_dir / name
        if not p.exists():
            missing.append(name)
        elif _sha256(p) != rec["sha256"]:
            bad.append(name)
        checked += 1
    return {"ok": not bad and not missing, "checked": checked, "bad": bad, "missing": missing}


class DecisionBase:
    """Backend-independent part. A backend implements ``_scores_many(rendered) -> list of list[float]``:
    ``rendered`` is a non-empty list of Rendered that all share ONE state (identical prefix ids and
    state blocks); it returns the raw float32 scores of every slot of every Rendered (slot order =
    canonical option order), computing the state blocks once per call. Backends keep the most recent
    state for the next call (exact prefix match only), so consecutive calls about the same state (e.g.
    JSONL rows read from stdin) reuse it."""
    backend = "base"

    def _setup(self, model_dir, tokenizer_json, max_len=CONTEXT_LIMIT, temperature=None):
        self.model_dir = Path(model_dir)
        self.readout_config = json.loads((self.model_dir / "readout_config.json").read_text())
        by_qtype = self.readout_config["temperatures"]["by_qtype"]
        missing = [q for q in QTYPES if q not in by_qtype]
        if missing:
            raise ValueError(f"readout_config.json has no calibrated temperature for {missing}")
        self.calibrated_temperatures = {q: check_temperature(by_qtype[q]) for q in QTYPES}
        self.default_temperature = None if temperature is None else check_temperature(temperature)
        self.encode = TextEncoder(tokenizer_json)
        self.renderer = Renderer(self.encode, self.readout_config, max_len=max_len)

    def temperature_for(self, qtype):
        """The temperature used for questions of this type (the override given to the constructor, if any)."""
        if self.default_temperature is not None:
            return self.default_temperature
        return self.calibrated_temperatures[qtype]

    def _scores_many(self, rendered):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _score_all(self, rendered):
        """Raw scores (lists of floats, canonical order) for Rendered sharing one state; one backend call."""
        if not rendered:
            return []
        p = rendered[0].prefix_len
        prefix, sblocks = rendered[0].ids[:p], rendered[0].state_blocks
        for r in rendered[1:]:
            if r.prefix_len != p or r.ids[:p] != prefix or r.state_blocks != sblocks:
                raise ValueError("all questions of one backend call must share the same state")
        scores = self._scores_many(rendered)
        if len(scores) != len(rendered) or any(len(s) != len(r.slots) for s, r in zip(scores, rendered)):
            raise RuntimeError("backend returned a wrong number of scores")
        return [[float(x) for x in s] for s in scores]

    def score_pieces(self, pieces_list, max_len=None):
        """Raw scores for pre-tokenised questions sharing one state: dicts {"prefix", "head", "opts",
        "names", "qtype"} of token ids (see Renderer.pieces). For parity checks; no temperature."""
        return self._score_all([self.renderer.assemble(p, max_len) for p in pieces_list])

    def _result(self, r, scores, temperature):
        t = self.temperature_for(r.qtype) if temperature is None else check_temperature(temperature)
        bad = [n for n, x in zip(r.names, scores) if not math.isfinite(x)]
        if bad:
            raise NonFiniteScoreError(f"non-finite decision scores for option(s) {bad} ({len(r.ids)} input tokens); "
                                      f"refusing to return probabilities. Check the weights file / engine build.")
        p = softmax_probabilities(scores, t)
        if not np.all(np.isfinite(p)):
            raise NonFiniteScoreError("non-finite probabilities")
        i = int(p.argmax())
        return {"answer": r.names[i], "probabilities": dict(zip(r.names, p.tolist())),
                "scores": dict(zip(r.names, scores)), "temperature": t, "top_probability": float(p[i]),
                "entropy_concentration": concentration(p), "input_tokens": len(r.ids),
                "state_tokens": r.prefix_len, "head_tokens": r.head_tokens, "image_tokens": None,
                "blocks": len(r.blocks), "catalogue_overflow": r.catalogue_overflow, "model": MODEL_NAME,
                "backend": self.backend}

    def decide(self, state, question, options=None, qtype=None, category=None, temperature=None, head_max=None,
               max_len=None):
        """Score one question about ``state``.

        Returns {"answer", "probabilities" {option: p}, "scores" {option: logit(yes)-logit(no)},
        "temperature", "top_probability", "entropy_concentration", "input_tokens", "state_tokens",
        "head_tokens", "image_tokens" (always None: text only), "blocks", "catalogue_overflow", "model",
        "backend"}. ``temperature`` overrides the calibrated temperature of the question's type (1.0 =
        uncalibrated scores). ``category`` and ``head_max`` are accepted for compatibility with the 0.8B v3
        runtime and the jev-style package, and ignored: this model has one temperature per question type and
        no separate question/options budget. Raises InputBudgetError (never truncates), QuestionError or
        NonFiniteScoreError."""
        q = make_question(question, options, qtype)
        r = self.renderer.render(state, q, max_len)
        return self._result(r, self._score_all([r])[0], temperature)

    def score_many(self, state, questions, category=None, temperature=None, head_max=None, max_len=None):
        """Several questions about ONE state: the state is tokenised and computed once and reused for
        every question (one backend call). ``questions``: dicts {"t","ins","crit"} (or anything
        make_question accepts). Results in order, identical to calling decide() per question. All
        questions are rendered and budget-checked before any scoring. ``category`` / ``head_max``: see
        decide() (accepted and ignored)."""
        qs = [make_question(q) for q in questions]
        if not qs:
            return []
        prefix = self.renderer.prefix_ids(state)
        rs = [self.renderer.render(state, q, max_len, prefix=prefix) for q in qs]
        return [self._result(r, sc, temperature) for r, sc in zip(rs, self._score_all(rs))]

    decide_many = score_many


def base_arg_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model-dir", default=str(HERE), help="folder with the weights and readout_config.json")
    ap.add_argument("--state", help="state as plain text")
    ap.add_argument("--state-json", help="state as a JSON value")
    ap.add_argument("--question", help="question text (or a JSON question {'t','ins','crit'})")
    ap.add_argument("--options", help="JSON: {name: description} or [names] (choice); [levels] (score)")
    ap.add_argument("--qtype", choices=QTYPES)
    ap.add_argument("--temperature", type=float,
                    help="override the calibrated per-type temperatures of readout_config.json (1.0 = raw scores)")
    ap.add_argument("--max-len", type=int, default=CONTEXT_LIMIT,
                    help=f"total token budget (state + question + options + readout), at most {CONTEXT_LIMIT}")
    ap.add_argument("--jsonl", help="batch mode: input JSON lines {id?, state, question, options?, qtype?, "
                                    "temperature?} ('-' = stdin); one JSON result per line on stdout. Consecutive "
                                    "rows with an identical state share one state computation")
    ap.add_argument("--verify", action="store_true", help="check sha256 of the files in manifest.json first")
    return ap


def _jsonl_groups(src, streaming):
    """(line number, record or error text) grouped into runs of consecutive rows with one state. From a
    file up to JSONL_GROUP_MAX rows are grouped; from stdin every row is its own group (answered at once;
    the backend's kept state still makes consecutive identical states cheap)."""
    group, key = [], None
    for n, line in enumerate(src):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError("a JSONL row must be a JSON object")
            k = serialize_state(rec.get("state", ""))
        except ValueError as e:
            if group:
                yield group
            group, key = [], None
            yield [(n, f"{type(e).__name__}: {e}")]
            continue
        if group and (k != key or len(group) >= JSONL_GROUP_MAX):
            yield group
            group = []
        group.append((n, rec))
        key = k
        if streaming:
            yield group
            group, key = [], None
    if group:
        yield group


def _run_group(engine, group, args):
    """Score one group of JSONL rows sharing a state. Returns (output rows, non-finite count)."""
    out, todo = {}, []
    prefix = None
    for n, rec in group:
        if isinstance(rec, str):
            out[n] = {"id": n, "error": rec}
            continue
        rid = rec.get("id", n)
        try:
            if "question" not in rec:
                raise QuestionError("row has no 'question'")
            q = make_question(rec["question"], options=rec.get("options"), qtype=rec.get("qtype"))
            t = rec.get("temperature", args.temperature)
            t = None if t is None else check_temperature(t)
            if prefix is None:
                prefix = engine.renderer.prefix_ids(rec.get("state", ""))
            todo.append((n, rid, engine.renderer.render(rec.get("state", ""), q, prefix=prefix), t))
        except (InputBudgetError, QuestionError, ValueError, TypeError, AttributeError) as e:
            out[n] = {"id": rid, "error": f"{type(e).__name__}: {e}"}
    nonfinite = 0
    if todo:
        scores = engine._score_all([r for _, _, r, _ in todo])
        for (n, rid, r, t), sc in zip(todo, scores):
            try:
                out[n] = {"id": rid, **engine._result(r, sc, t)}
            except NonFiniteScoreError as e:
                nonfinite += 1
                print(f"ERROR row {rid}: NonFiniteScoreError: {e}", file=sys.stderr, flush=True)
                out[n] = {"id": rid, "error": f"NonFiniteScoreError: {e}"}
    return [out[n] for n, _ in group], nonfinite


def run_cli(args, engine):
    """Exit status: 0 = ok (JSONL rows with input errors carry an "error" field), 2 = input error
    (single question), 3 = at least one non-finite score (refused, see stderr)."""
    if args.jsonl:
        streaming = args.jsonl == "-"
        src = sys.stdin if streaming else open(args.jsonl, encoding="utf-8")
        nonfinite = 0
        try:
            for group in _jsonl_groups(src, streaming):
                rows, bad = _run_group(engine, group, args)
                nonfinite += bad
                for row in rows:
                    print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            if not streaming:
                src.close()
        return 3 if nonfinite else 0
    if args.question is None:
        raise SystemExit("--question (or --jsonl) is required")
    try:
        state = json.loads(args.state_json) if args.state_json is not None else (args.state or "")
        question = args.question
        if question.lstrip().startswith("{"):
            try:                                            # a JSON question {'t','ins','crit'}; else plain text
                parsed = json.loads(question)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                question = parsed
        options = json.loads(args.options) if args.options else None
        res = engine.decide(state, question, options=options, qtype=args.qtype, temperature=args.temperature)
    except (InputBudgetError, QuestionError, ValueError, TypeError) as e:  # NonFiniteScoreError is not a ValueError
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    except NonFiniteScoreError as e:
        print(f"ERROR: NonFiniteScoreError: {e}", file=sys.stderr)
        return 3
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0
# ---------------------------------------------------------------------------- end of shared core


# ---------------------------------------------------------------------------- llama.cpp backend
import os
import shutil
import subprocess
import tempfile
import threading
import time

GGUF_FILES = {q: f"{MODEL_NAME}-{q}.gguf" for q in ("F16", "Q8_0", "Q4_K_M")}
SCORER_PROTOCOL = "jev-score-v2"
TESTED_LLAMA_CPP = "441df11f65ea0b6d0c72965aaf70c8241070ddcb"
LLAMA_N_CTX = 32768        # >= the 32,768-token context (the parity-tested value)
N_UBATCH = BLOCK           # one llama_decode per renderer block; blocks are never re-split
N_OUTPUTS_MAX = 512        # > the slots of any <= 2,048-token block (every option/rubric line is >= 4 tokens)
FLASH_ATTN = ("default", "auto", "on", "off")
assert LLAMA_N_CTX >= CONTEXT_LIMIT
EXPECTED_GGUF = {"general.architecture": "qwen35", "qwen35.block_count": 24, "qwen35.embedding_length": 2048,
                 "general.name": "Jev-Style-2B-Decision-v4"}


def read_gguf_metadata(path, keys=tuple(EXPECTED_GGUF)):
    """Read a few scalar key/value pairs from a GGUF header (no dependency; stops once all are found)."""
    import struct
    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    want, out = set(keys), {}
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize(fmt)
            return struct.unpack(fmt, f.read(n))[0]

        def rstr():
            return f.read(rd("<Q")).decode("utf-8", "replace")

        def skip(t):
            if t in scalar:
                f.seek(struct.calcsize(scalar[t]), 1)
            elif t == 8:
                f.seek(rd("<Q"), 1)
            elif t == 9:
                et, n = rd("<I"), rd("<Q")
                if et in scalar:
                    f.seek(struct.calcsize(scalar[et]) * n, 1)
                else:
                    for _ in range(n):
                        skip(et)
            else:
                raise ValueError(f"unknown GGUF value type {t}")

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        rd("<I"), rd("<Q")
        for _ in range(rd("<Q")):
            key, t = rstr(), rd("<I")
            if key in want and (t in scalar or t == 8):
                out[key] = rstr() if t == 8 else rd(scalar[t])
                if len(out) == len(want):
                    break
            else:
                skip(t)
    return out


def find_scorer(binary=None):
    """jev-score-v2 binary: argument, $JEV_SCORE_V2_BIN, ./build/jev-score-v2 next to this script, or on PATH."""
    if binary and not Path(binary).is_file():
        raise FileNotFoundError(f"jev-score-v2 binary not found at {binary} (the path given with --jev-score / binary=)")
    for cand in (binary, os.environ.get("JEV_SCORE_V2_BIN"), HERE / "build" / "jev-score-v2",
                 shutil.which("jev-score-v2")):
        if cand and Path(cand).is_file():
            return Path(cand)
    raise FileNotFoundError("jev-score-v2 binary not found: build it with `sh build_jev_score.sh /path/to/llama.cpp` "
                            f"(tested llama.cpp commit {TESTED_LLAMA_CPP}), or pass --jev-score / set JEV_SCORE_V2_BIN")


class JevStyleDecisionGGUF(DecisionBase):
    """llama.cpp runtime. The GGUF is run by ``jev-score-v2`` (jev_score_v2.cpp, a ~350-line program on
    stock libllama shipped in this repo): one context with NON_CAUSAL attention and one llama_decode per
    renderer block (block-causal attention as in training), " yes" / " no" logits read only at the slots.
    This class renders the input with the shipped tokenizer and applies the readout + calibration.

    quant: "F16" (default), "Q8_0" or "Q4_K_M"; ``gguf`` overrides it with an explicit file.
    n_gpu_layers: 999 = everything on the GPU (CUDA or Metal; the parity checks ran on CUDA); 0 = CPU.
    flash_attn: None/"default" = llama.cpp "auto" when layers run on a GPU backend (the parity-tested
    setting) and "off" on the CPU fallback (CPU flash attention was measured to lose precision with the
    2B v3); or "auto" / "on" / "off" explicitly.
    temperature: None = the calibrated temperature of each question's type (readout_config.json).

    State reuse: ``score_many`` sends all questions about one state in ONE request; the state blocks are
    decoded once into sequence 0 and each question runs on a copy (seq_cp) of it. The last state stays
    cached in jev-score-v2, so a following request with the identical state skips it (exact match
    only). Results are identical to scoring every question alone (measured max |diff| 0.0, see
    release_config.json -> runtime_parity).
    """
    backend = "llama.cpp"

    def __init__(self, model_dir=HERE, quant="F16", gguf=None, binary=None, max_len=CONTEXT_LIMIT, temperature=None,
                 n_gpu_layers=999, threads=None, flash_attn=None, cache_type="f16", verify=False, stderr=None,
                 check_vocab=True, check_model=True):
        model_dir = Path(model_dir)
        if gguf:
            self.gguf = Path(gguf)
        else:
            if str(quant).upper() not in GGUF_FILES:
                raise ValueError(f"quant must be one of {sorted(GGUF_FILES)}, got {quant!r}")
            self.gguf = model_dir / GGUF_FILES[str(quant).upper()]
        if not self.gguf.is_file():
            raise FileNotFoundError(f"GGUF file not found: {self.gguf} (download it next to this script, or pass "
                                    f"--gguf / --model-dir)")
        meta = read_gguf_metadata(self.gguf)
        wrong = {k: meta.get(k) for k, v in EXPECTED_GGUF.items() if meta.get(k) != v}
        if wrong and check_model:      # e.g. the 2B v3 / 0.8B v3 GGUFs: same tokenizer, the vocab check alone would not catch it
            raise ValueError(f"{self.gguf} is not a {MODEL_NAME} GGUF (expected {EXPECTED_GGUF}, got {wrong}); "
                             f"other models' GGUFs give plausible-looking but wrong probabilities with this runtime")
        tokenizer_json = model_dir / "tokenizer" / "tokenizer.json"
        if not tokenizer_json.is_file():
            raise FileNotFoundError(f"{tokenizer_json} not found: download the repository's tokenizer/ folder")
        flash_attn = "default" if flash_attn is None else str(flash_attn)
        if flash_attn not in FLASH_ATTN:
            raise ValueError(f"flash_attn must be one of {FLASH_ATTN}")
        if cache_type not in ("f16", "f32"):
            raise ValueError("cache_type must be 'f16' or 'f32'")
        if verify:
            only = ["tokenizer", "readout_config.json"]
            if self.gguf.parent.resolve() == model_dir.resolve():      # a file of this repo folder
                only.append(self.gguf.name)
            res = verify_manifest(model_dir, only=only)
            if not res["ok"]:
                raise RuntimeError(f"integrity check failed: {res}")
            self.verified = res
        self._setup(model_dir, tokenizer_json, max_len, temperature)
        self.rows = [self.renderer.yes, self.renderer.no]
        self.binary = find_scorer(binary)
        self.cmd = [str(self.binary), "--model", str(self.gguf), "--n-ctx", str(LLAMA_N_CTX), "--n-ubatch",
                    str(N_UBATCH), "--ngl", str(int(n_gpu_layers)), "--flash-attn", flash_attn,
                    "--n-outputs-max", str(N_OUTPUTS_MAX), "--cache-type", cache_type]
        if threads:
            self.cmd += ["--threads", str(int(threads))]
        self._err = stderr if stderr is not None else tempfile.TemporaryFile(prefix="jev-score-v2-")
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                                     text=True, encoding="utf-8", bufsize=1)
        self._lock = threading.Lock()
        line = self.proc.stdout.readline()
        try:
            ready = json.loads(line) if line else {"ready": False, "error": "no output"}
        except json.JSONDecodeError:
            ready = {"ready": False, "error": f"unexpected output {line[:200]!r}"}
        problem = None
        if not ready.get("ready"):
            problem = f"jev-score-v2 failed to start: {ready}"
        elif ready.get("protocol") != SCORER_PROTOCOL:
            problem = (f"{self.binary} speaks protocol {ready.get('protocol')!r}, not {SCORER_PROTOCOL!r}: build "
                       f"jev_score_v2.cpp from this repository (the 0.8B jev-score binary is NOT valid for this model)")
        elif (ready.get("attention_type") != "non_causal" or int(ready.get("n_ubatch", 0)) != N_UBATCH
              or int(ready.get("n_batch", 0)) != N_UBATCH):
            problem = f"jev-score-v2 context is not the v2 block protocol: {ready}"
        if problem:
            tail = self._stderr_tail()
            self.close()
            raise RuntimeError(problem + (f"\n--- jev-score-v2 stderr ---\n{tail}" if tail else ""))
        self.info = ready
        self.last_timing = {}
        if check_vocab:
            self.vocab_check = self._check_vocab()

    def _stderr_tail(self, n=4000):
        try:
            if hasattr(self._err, "seek"):
                self._err.flush()
                self._err.seek(0)
                return self._err.read()[-n:].decode("utf-8", "replace")
        except Exception:
            pass
        return ""

    def _check_vocab(self):
        """The GGUF vocabulary must spell the readout / glue tokens exactly like the shipped tokenizer."""
        ids = sorted(set(self.rows + self.renderer.arrow + self.renderer.newline + self.renderer.dash
                         + self.renderer.rubric_head + self.encode("State:\n") + self.encode("Question [choice]")))
        resp = self._request({"cmd": "vocab", "ids": ids})
        for i, got in zip(ids, resp["tokens"]):
            want = self.encode.id_to_token(i)
            if got != want:
                raise ValueError(f"GGUF vocabulary id {i} is {got!r} but tokenizer/tokenizer.json has {want!r}: "
                                 f"the GGUF and the tokenizer do not belong together")
        if int(resp["n_vocab"]) < self.encode.vocab_size():
            raise ValueError(f"GGUF vocabulary ({resp['n_vocab']}) is smaller than the tokenizer's "
                             f"({self.encode.vocab_size()})")
        return {"checked": len(ids), "n_vocab_gguf": int(resp["n_vocab"])}

    def _request(self, req):
        with self._lock:
            if self.proc.poll() is not None:
                raise RuntimeError(f"jev-score-v2 exited ({self.proc.returncode}): {self._stderr_tail()}")
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"jev-score-v2 exited: {self._stderr_tail()}")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"jev-score-v2: {resp['error']}")
        return resp

    def _scores_many(self, rendered):
        """All questions about one state in ONE jev-score-v2 request (state blocks decoded once)."""
        r0 = rendered[0]
        p = r0.prefix_len
        req = {"prefix": r0.ids[:p], "prefix_blocks": [list(b) for b in r0.state_blocks], "keep_prefix": True,
               "questions": [{"ids": r.ids[p:], "blocks": [list(b) for b in r.question_blocks], "slots": list(r.slots),
                              "rows": self.rows} for r in rendered]}
        t0 = time.perf_counter()
        resp = self._request(req)
        self.last_timing = dict(resp.get("timing") or {}, wall_ms=(time.perf_counter() - t0) * 1000,
                                prefix_reused=resp.get("prefix_reused"), n_state_blocks=resp.get("n_state_blocks"))
        return [[float("nan") if v is None else float(v) for v in res["scores"]] for res in resp["results"]]

    def reset(self):
        """Forget the cached state (the next request recomputes it)."""
        self._request({"cmd": "reset"})

    def close(self):
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write('{"cmd":"quit"}\n')
                proc.stdin.flush()
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
        for f in (getattr(proc, "stdin", None), getattr(proc, "stdout", None)):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def main(argv=None):
    ap = base_arg_parser(f"{MODEL_NAME}: typed decisions with llama.cpp (GGUF)")
    ap.add_argument("--quant", default="F16", choices=sorted(GGUF_FILES))
    ap.add_argument("--gguf", help="explicit GGUF path (overrides --quant)")
    ap.add_argument("--jev-score", help="path to the jev-score-v2 binary (default: $JEV_SCORE_V2_BIN, "
                                        "./build/jev-score-v2, PATH)")
    ap.add_argument("--ngl", type=int, default=999, help="layers offloaded to the GPU (0 = CPU only)")
    ap.add_argument("--no-model-check", action="store_true", help=argparse.SUPPRESS)   # tests with tiny GGUFs only
    ap.add_argument("--threads", type=int, help="CPU threads of jev-score-v2 (default 8)")
    ap.add_argument("--flash-attn", default="default", choices=FLASH_ATTN,
                    help="default = auto on a GPU backend, off on the CPU fallback")
    args = ap.parse_args(argv)
    try:
        engine = JevStyleDecisionGGUF(args.model_dir, quant=args.quant, gguf=args.gguf, binary=args.jev_score,
                                      max_len=args.max_len, n_gpu_layers=args.ngl, threads=args.threads,
                                      flash_attn=args.flash_attn, verify=args.verify,
                                      check_model=not args.no_model_check)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise SystemExit(f"error: {e}")
    try:
        return run_cli(args, engine)
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
