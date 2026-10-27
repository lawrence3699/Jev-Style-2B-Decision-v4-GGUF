// Part of Jev-Style-2B-Decision-v4 (Apache-2.0). Standalone copy of the project's v2 scorer: the 2B v3 scorer,
// unchanged apart from these header comments (same protocol "jev-score-v2").
// Tested against llama.cpp commit 441df11f65ea0b6d0c72965aaf70c8241070ddcb (2026-09-23; macOS / Metal with the
// 2B v3, Linux / CUDA with v4);
// needs a llama.cpp with qwen35 support and llama_context_params.n_outputs_max.
// Build: sh build_jev_score.sh /path/to/llama.cpp   (-> ./build/jev-score-v2)
//
// jev-score-v2: score Jev-Style-2B-Decision-v4 typed decisions (block attention, render v2) with STOCK
// libllama, reading logits only at the slot positions.
//
// Protocol (must match training exactly):
//   * the renderer (jev_style_decision_gguf.py) gives [start, stop) blocks; in the full-attention layers
//     the queries of block [a, b) attend keys [0, b) with NO causal mask inside the block; the
//     Gated-DeltaNet layers stay causal/recurrent;
//   * readout: logit(" yes") - logit(" no") at every slot, i.e. h_slot . (W_yes - W_no) in FP32.
//
// How stock libllama gives exactly this:
//   * the context is created ONCE with attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL and is never
//     toggled: the KQ mask then lets a query see every cache cell of its sequence (no position test);
//   * n_batch = n_ubatch = 2048 and every llama_decode receives exactly ONE renderer block (never
//     re-split, never merged): one llama_decode per block. A non-causal ubatch sees only the cells
//     already in the cache (= all earlier blocks) plus itself => block-causal attention, the layout of
//     training. llama_decode asserts n_ubatch >= n_tokens for non-causal contexts, so a block longer
//     than n_ubatch is rejected here with an error before it can reach that assert;
//   * the recurrent (GDN) layers are sequential by construction and do not read causal_attn;
//   * the state blocks go into sequence 0 once; each question: seq_cp(0 -> 1), decode its
//     question-side block(s) into sequence 1 (logits requested ONLY at slot positions, read after
//     every decode because the next decode overwrites them), then seq_rm(1). One question per decode;
//   * exact-prefix reuse: with keep_prefix an identical prefix (same ids AND same blocks) in the next
//     request reuses sequence 0 (recurrent state cannot be rolled back => exact match only).
//
// Persistent JSON-lines server. Start:
//   jev-score-v2 --model model.gguf [--n-ctx 32768] [--n-ubatch 2048] [--ngl 999] [--threads 8]
//                [--flash-attn default|auto|on|off] [--n-outputs-max 512] [--cache-type f16|f32]
// --flash-attn default (the default) = llama.cpp "auto" when layers are offloaded to a GPU backend
// (ngl > 0 and the build supports GPU offload; the parity-tested setting) and "off" otherwise:
// on the CPU backend flash attention was measured to lose precision, so the CPU fallback keeps it off.
// (--cache-type f32 keeps K/V in FP32, used by exactness tests; f16 is the llama.cpp default.)
// It prints {"ready":true,"protocol":"jev-score-v2",...}. Each stdin line is one request:
//   {"prefix":[ids], "prefix_blocks":[[a,b],...],
//    "questions":[{"ids":[question-side ids], "blocks":[[a,b],...] (absolute), "slots":[abs],
//                  "rows":[yes_id, no_id]}],
//    "keep_prefix":true, "return_logits":false}
// prefix_blocks must tile [0, n_prefix) and each question's blocks must tile
// [n_prefix, n_prefix + len(ids)); every block <= n_ubatch; slots inside the question part.
// Response:
//   {"results":[{"scores":[yes-no per slot, null if non-finite], "finite":bool, ["logits":[[yes,no]...]]}],
//    "prefix_reused":bool, "n_prefix":N, "n_state_blocks":B,
//    "timing":{"prefix_ms":..,"question_ms":[..],"total_ms":..}}
// {"cmd":"reset"} clears the memory; {"cmd":"info"} reports the context; {"cmd":"vocab","ids":[..]}
// returns the GGUF vocabulary text of those token ids (tokenizer check); {"cmd":"quit"} exits.

#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;
using clk = std::chrono::steady_clock;
using Block = std::pair<int, int>;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static std::vector<Block> parse_blocks(const json & j) {
    std::vector<Block> out;
    for (const auto & b : j) {
        if (!b.is_array() || b.size() != 2) throw std::runtime_error("block must be [start, stop]");
        out.emplace_back(b[0].get<int>(), b[1].get<int>());
    }
    return out;
}

// blocks must tile [lo, hi) contiguously, each non-empty and <= max_len
static void check_tiling(const std::vector<Block> & blocks, int lo, int hi, int max_len, const char * what) {
    int at = lo;
    for (const auto & b : blocks) {
        if (b.first != at || b.second <= b.first)
            throw std::runtime_error(std::string(what) + " blocks do not tile the range contiguously");
        if (b.second - b.first > max_len)
            throw std::runtime_error(std::string(what) + " block of " + std::to_string(b.second - b.first) +
                                     " tokens exceeds n_ubatch=" + std::to_string(max_len) +
                                     " (blocks are never re-split: restart with a larger --n-ubatch)");
        at = b.second;
    }
    if (at != hi) throw std::runtime_error(std::string(what) + " blocks end at " + std::to_string(at) +
                                           ", expected " + std::to_string(hi));
}

struct ScorerV2 {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    int n_ctx = 0, n_batch = 0, n_ubatch = 0, n_vocab = 0, n_outputs_max = 512;
    std::vector<llama_token> cached_prefix;   // content of sequence 0
    std::vector<Block> cached_blocks;
    bool cached_valid = false;
    long n_decodes = 0;

    void clear() {
        llama_memory_clear(llama_get_memory(ctx), true);
        cached_prefix.clear();
        cached_blocks.clear();
        cached_valid = false;
    }

    // Decode exactly one block: tokens ids[off .. off+len) at positions pos0.. into seq. Logits are
    // requested at the absolute positions in `want` that fall inside the block; returns
    // (want index, batch index) pairs.
    std::vector<std::pair<int, int>> decode_block(const std::vector<llama_token> & ids, int off, int len, int pos0,
                                                  llama_seq_id seq, const std::vector<int> & want) {
        if (len <= 0 || len > n_ubatch) throw std::runtime_error("internal: block length out of range");
        llama_batch batch = llama_batch_init(len, 0, 1);
        batch.n_tokens = len;
        std::vector<std::pair<int, int>> got;
        for (int j = 0; j < len; ++j) {
            batch.token[j] = ids[off + j];
            batch.pos[j] = pos0 + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq;
            batch.logits[j] = 0;
        }
        for (size_t w = 0; w < want.size(); ++w) {
            const int rel = want[w] - pos0;
            if (rel >= 0 && rel < len) {
                batch.logits[rel] = 1;
                got.emplace_back((int) w, rel);
            }
        }
        if ((int) got.size() > n_outputs_max) {
            llama_batch_free(batch);
            throw std::runtime_error("block has " + std::to_string(got.size()) + " slots; n_outputs_max=" +
                                     std::to_string(n_outputs_max) + " (restart with a larger --n-outputs-max)");
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        ++n_decodes;
        if (rc != 0) throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
        return got;
    }

    json handle(const json & req) {
        const auto t_total = clk::now();
        const std::vector<llama_token> prefix = req.value("prefix", std::vector<llama_token>{});
        const int n_prefix = (int) prefix.size();
        const std::vector<Block> pblocks = parse_blocks(req.value("prefix_blocks", json::array()));
        const bool keep = req.value("keep_prefix", true);
        const bool ret_logits = req.value("return_logits", false);
        const json & qs = req.at("questions");
        llama_memory_t mem = llama_get_memory(ctx);

        check_tiling(pblocks, 0, n_prefix, n_ubatch, "prefix");
        struct Q { std::vector<llama_token> ids; std::vector<Block> blocks; std::vector<int> slots; std::vector<llama_token> rows; };
        std::vector<Q> Qs;
        for (const auto & q : qs) {
            Q x;
            x.ids = q.at("ids").get<std::vector<llama_token>>();
            x.blocks = parse_blocks(q.at("blocks"));
            x.slots = q.at("slots").get<std::vector<int>>();
            x.rows = q.value("rows", std::vector<llama_token>{});
            const int end = n_prefix + (int) x.ids.size();
            if (x.ids.empty()) throw std::runtime_error("empty question");
            if (end > n_ctx) throw std::runtime_error("input of " + std::to_string(end) + " tokens exceeds n_ctx=" +
                                                      std::to_string(n_ctx) + "; nothing truncated");
            check_tiling(x.blocks, n_prefix, end, n_ubatch, "question");
            if (x.rows.size() != 2) throw std::runtime_error("rows must be [yes_id, no_id]");
            for (llama_token t : x.rows)
                if (t < 0 || t >= n_vocab) throw std::runtime_error("row token id out of range");
            for (int s : x.slots)
                if (s < n_prefix || s >= end) throw std::runtime_error("slot position outside the question part");
            Qs.push_back(std::move(x));
        }

        bool reused = keep && cached_valid && n_prefix > 0 && cached_prefix == prefix && cached_blocks == pblocks;
        double prefix_ms = 0.0;
        if (!reused) {
            const auto t0 = clk::now();
            clear();
            for (const auto & b : pblocks) decode_block(prefix, b.first, b.second - b.first, b.first, 0, {});
            llama_synchronize(ctx);
            prefix_ms = ms_since(t0);
            if (n_prefix > 0) {
                cached_prefix = prefix;
                cached_blocks = pblocks;
                cached_valid = true;
            }
        }

        json results = json::array();
        json q_ms = json::array();
        for (const auto & x : Qs) {
            const auto t0 = clk::now();
            llama_memory_seq_rm(mem, 1, -1, -1);
            if (n_prefix > 0) llama_memory_seq_cp(mem, 0, 1, -1, -1);
            std::vector<float> scores(x.slots.size(), NAN);
            json logits = json::array();
            std::vector<std::pair<float, float>> raw(x.slots.size(), {NAN, NAN});
            std::vector<char> seen(x.slots.size(), 0);
            for (const auto & b : x.blocks) {
                const int off = b.first - n_prefix;
                auto got = decode_block(x.ids, off, b.second - b.first, b.first, 1, x.slots);
                for (const auto & wg : got) {
                    const float * lg = llama_get_logits_ith(ctx, wg.second);
                    if (!lg) throw std::runtime_error("no logits at requested slot");
                    const float y = lg[x.rows[0]], n = lg[x.rows[1]];
                    raw[wg.first] = {y, n};
                    scores[wg.first] = y - n;           // FP32 difference of FP32 logits
                    seen[wg.first] = 1;
                }
            }
            llama_memory_seq_rm(mem, 1, -1, -1);
            bool finite = true;
            json sj = json::array();
            for (size_t k = 0; k < scores.size(); ++k) {
                if (!seen[k]) throw std::runtime_error("internal: slot not covered by any block");
                if (!std::isfinite(scores[k])) finite = false;
                sj.push_back(std::isfinite(scores[k]) ? json(scores[k]) : json(nullptr));
                if (ret_logits) logits.push_back({raw[k].first, raw[k].second});
            }
            json r = {{"scores", sj}, {"finite", finite}, {"n_blocks", x.blocks.size()}};
            if (ret_logits) r["logits"] = logits;
            results.push_back(r);
            q_ms.push_back(ms_since(t0));
        }
        if (!keep) clear();
        json out;
        out["results"] = results;
        out["prefix_reused"] = reused;
        out["n_prefix"] = n_prefix;
        out["n_state_blocks"] = pblocks.size();
        out["timing"] = {{"prefix_ms", prefix_ms}, {"question_ms", q_ms}, {"total_ms", ms_since(t_total)}};
        return out;
    }
};

static void usage() {
    fprintf(stderr, "usage: jev-score-v2 --model PATH [--n-ctx N] [--n-ubatch N] [--ngl N] [--threads N] "
                    "[--flash-attn default|auto|on|off] [--n-outputs-max N] [--cache-type f16|f32]\n");
}

int main(int argc, char ** argv) {
    std::string model_path, fa = "default", cache_type = "f16";
    int n_ctx = 32768, n_ubatch = 2048, ngl = 999, threads = 8, n_out = 512;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(2); }
            return argv[++i];
        };
        if (a == "--model") model_path = next();
        else if (a == "--n-ctx") n_ctx = std::stoi(next());
        else if (a == "--n-ubatch") n_ubatch = std::stoi(next());
        else if (a == "--ngl") ngl = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--flash-attn") fa = next();
        else if (a == "--n-outputs-max") n_out = std::stoi(next());
        else if (a == "--cache-type") cache_type = next();
        else { usage(); return 2; }
    }
    if (model_path.empty() || (cache_type != "f16" && cache_type != "f32") ||
        (fa != "default" && fa != "auto" && fa != "on" && fa != "off")) { usage(); return 2; }

    const auto t0 = clk::now();
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
    }, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    ScorerV2 S;
    S.model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!S.model) { std::cout << json{{"ready", false}, {"error", "model load failed"}}.dump() << std::endl; return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ubatch;           // n_batch = n_ubatch: one decode = one block = one ubatch
    cp.n_ubatch = n_ubatch;
    cp.n_seq_max = 2;                // seq 0 = state, seq 1 = current question
    cp.n_outputs_max = std::max(2, n_out);
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.kv_unified = true;
    cp.no_perf = true;
    cp.embeddings = false;
    cp.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;   // set once, never toggled
    cp.type_k = cp.type_v = cache_type == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
    const bool gpu_offload = llama_supports_gpu_offload() && ngl > 0;
    if (fa == "default") fa = gpu_offload ? "auto" : "off";   // CPU fallback: flash attention off (precision)
    cp.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    S.ctx = llama_init_from_model(S.model, cp);
    if (!S.ctx) { std::cout << json{{"ready", false}, {"error", "context init failed"}}.dump() << std::endl; return 1; }
    S.n_ctx = (int) llama_n_ctx(S.ctx);
    S.n_batch = (int) llama_n_batch(S.ctx);
    S.n_ubatch = (int) llama_n_ubatch(S.ctx);
    S.n_outputs_max = (int) cp.n_outputs_max;
    S.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(S.model));
    if (S.n_ubatch != S.n_batch) {
        std::cout << json{{"ready", false}, {"error", "n_batch != n_ubatch after context init"}}.dump() << std::endl;
        return 1;
    }
    char desc[256];
    llama_model_desc(S.model, desc, sizeof(desc));
    const json info = {{"ready", true}, {"protocol", "jev-score-v2"}, {"attention_type", "non_causal"},
                       {"flash_attn", fa}, {"gpu_offload", gpu_offload}, {"ngl", ngl}, {"n_threads", threads},
                       {"n_ctx", S.n_ctx}, {"n_batch", S.n_batch}, {"cache_type", cache_type}, {"n_ubatch", S.n_ubatch},
                       {"n_seq_max", (int) llama_n_seq_max(S.ctx)}, {"n_outputs_max", S.n_outputs_max},
                       {"n_vocab", S.n_vocab}, {"desc", desc}, {"load_ms", ms_since(t0)},
                       {"system_info", llama_print_system_info()}};
    std::cout << info.dump() << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json resp;
        try {
            json req = json::parse(line);
            const std::string cmd = req.value("cmd", "");
            if (cmd == "quit") break;
            if (cmd == "reset") { S.clear(); resp = {{"ok", true}}; }
            else if (cmd == "info") { resp = info; resp["n_decodes"] = S.n_decodes; resp["cached_prefix"] = S.cached_prefix.size(); }
            else if (cmd == "vocab") {
                const llama_vocab * vocab = llama_model_get_vocab(S.model);
                json toks = json::array();
                for (const auto & t : req.at("ids")) {
                    const int id = t.get<int>();
                    if (id < 0 || id >= S.n_vocab) throw std::runtime_error("token id out of range");
                    toks.push_back(std::string(llama_vocab_get_text(vocab, id)));
                }
                resp = {{"tokens", toks}, {"n_vocab", S.n_vocab}};
            }
            else resp = S.handle(req);
        } catch (const std::exception & e) {
            try { S.clear(); } catch (...) {}
            resp = {{"error", e.what()}};
        }
        std::cout << resp.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
    }
    llama_free(S.ctx);
    llama_model_free(S.model);
    llama_backend_free();
    return 0;
}
