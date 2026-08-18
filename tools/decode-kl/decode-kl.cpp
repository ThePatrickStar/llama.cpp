// decode-kl: measure decode-time drift of a model vs a full-model reference.
//
// Motivation: llama-perplexity --kl-divergence evaluates a corpus in large
// prefill batches, so it exercises the batched path, not the single-token
// autoregressive decode path where the UMA give-back expert streaming actually
// operates. This tool runs a real batch=1 decode loop so the captured logits
// reflect whatever the loaded model does at decode time (give-back is enabled
// purely via LLAMA_UMA_STREAM_* env vars at model load; this tool is agnostic).
//
// Two passes over the same held-out prompts:
//   --teacher <file>  full model greedily decodes each prompt and writes its
//                     per-step next-token distribution (perplexity-compatible
//                     compressed logprobs) plus the token it produced.
//   --student <file>  a (typically shrunk) model is teacher-forced on the
//                     teacher's tokens; at each step it computes KL(teacher ||
//                     student), same-top agreement and dp vs the reference.
//
// The reference file is written once (full model) and reused across many
// student residency levels. Alignment is exact by construction (teacher
// forcing), so KL is well defined even when greedy paths diverge.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ---- compressed-logprob helpers (format shared with tools/perplexity) ----

static int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// encode the full-vocab distribution into perplexity's uint16 logprob record.
// layout: 2 floats (scale, min_log_prob) in the first 4 uint16 slots, then one
// quantized logprob per vocab entry. nv = 2*((n_vocab+1)/2) + 4 uint16 values.
static void encode_logprobs(int n_vocab, const float * logits, uint16_t * log_prob) {
    float max_logit = logits[0];
    float min_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    min_logit = std::max(min_logit, max_logit - 16);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    const float min_log_prob = min_logit - max_logit - log_sum_exp;
    const float scale = (max_logit - min_logit)/65535.f;
    float * d = (float *)log_prob;
    d[0] = scale;
    d[1] = min_log_prob;
    log_prob += 4;
    if (scale) {
        const float inv_scale = 1/scale;
        for (int i = 0; i < n_vocab; ++i) {
            log_prob[i] = logits[i] > min_logit ? nearest_int(inv_scale*(logits[i] - min_logit)) : 0;
        }
    } else {
        memset(log_prob, 0, n_vocab*sizeof(uint16_t));
    }
}

// KL(base || cur) plus same-top and dp for the target token, from cur logits
// and the base compressed record. Mirrors tools/perplexity log_softmax().
struct kl_step_out {
    double kl;
    double p_diff;
    bool   same_top;
};

static kl_step_out kl_step(int n_vocab, const float * logits, const uint16_t * base_log_prob, int tok) {
    float max_logit = logits[0];
    int imax = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            imax = i;
        }
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    const float * d = (const float *)base_log_prob;
    const float scale = d[0];
    const float min_log_prob = d[1];
    base_log_prob += 4;

    const float nll = max_logit + log_sum_exp - logits[tok];
    const float nll_base = -(scale*base_log_prob[tok] + min_log_prob);

    max_logit += log_sum_exp;
    double sum = 0;
    int imax_base = -1;
    float p_log_base_max = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float p_log_base = scale*base_log_prob[i] + min_log_prob;
        if (i == 0 || p_log_base > p_log_base_max) {
            p_log_base_max = p_log_base;
            imax_base = i;
        }
        if (p_log_base > -16.f) {
            const float p_base = expf(p_log_base);
            sum += p_base * (p_log_base - logits[i] + max_logit);
        }
    }
    const float p_base = expf(-nll_base);
    const float p = expf(-nll);

    kl_step_out out;
    out.kl = sum;
    out.p_diff = p - p_base;
    out.same_top = (imax == imax_base);
    return out;
}

// ---- reference file format ----
// magic[8]="DECKLK01", int32 n_vocab, int32 n_prompts, then per prompt:
//   int32 n_prompt_tokens, int32 n_steps, llama_token prompt_tokens[n_prompt_tokens],
//   per step: llama_token teacher_token, uint16 logprob[nv]

static const char DECKL_MAGIC[8] = {'D','E','C','K','L','K','0','1'};

struct args_t {
    std::string model;
    std::string prompts;
    std::string ref;           // teacher output / student input
    std::string csv;           // optional per-step dump (student)
    std::string tag = "NA";    // echoed into RESULT (e.g. the residency level)
    bool teacher = false;
    bool student = false;
    int  n_gen   = 128;
    int  n_prompts = 0;        // 0 = all
    int  n_ctx   = 8192;
    int  ngl     = 99;
};

static void usage(const char * a0) {
    fprintf(stderr,
        "usage: %s -m MODEL --prompts FILE (--teacher REF | --student REF) [opts]\n"
        "  --teacher REF     full model: write reference logprobs to REF\n"
        "  --student REF     shrunk model: read REF, teacher-force, print KL\n"
        "  --prompts FILE    one prompt per line (single-line prompts)\n"
        "  --n-gen N         tokens generated/compared per prompt (default 128)\n"
        "  --n-prompts N     cap number of prompts (default all)\n"
        "  --n-ctx N         context size (default 8192)\n"
        "  --ngl N           gpu layers (default 99)\n"
        "  --tag STR         label echoed into the RESULT line\n"
        "  --csv FILE        optional per-step CSV (student mode)\n",
        a0);
}

static bool parse_args(int argc, char ** argv, args_t & a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](std::string & dst) { if (i+1 < argc) { dst = argv[++i]; return true; } return false; };
        auto nexti = [&](int & dst) { if (i+1 < argc) { dst = std::stoi(argv[++i]); return true; } return false; };
        if      (k == "-m")           { if (!next(a.model)) return false; }
        else if (k == "--prompts")    { if (!next(a.prompts)) return false; }
        else if (k == "--teacher")    { a.teacher = true; if (!next(a.ref)) return false; }
        else if (k == "--student")    { a.student = true; if (!next(a.ref)) return false; }
        else if (k == "--csv")        { if (!next(a.csv)) return false; }
        else if (k == "--tag")        { if (!next(a.tag)) return false; }
        else if (k == "--n-gen")      { if (!nexti(a.n_gen)) return false; }
        else if (k == "--n-prompts")  { if (!nexti(a.n_prompts)) return false; }
        else if (k == "--n-ctx")      { if (!nexti(a.n_ctx)) return false; }
        else if (k == "--ngl")        { if (!nexti(a.ngl)) return false; }
        else { fprintf(stderr, "unknown arg: %s\n", k.c_str()); return false; }
    }
    if (a.model.empty() || a.prompts.empty() || a.teacher == a.student) return false;
    return true;
}

static std::vector<std::string> read_prompts(const std::string & path, int cap) {
    std::vector<std::string> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        out.push_back(line);
        if (cap > 0 && (int) out.size() >= cap) break;
    }
    return out;
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & p) {
    int n = -llama_tokenize(vocab, p.c_str(), p.size(), NULL, 0, true, true);
    std::vector<llama_token> toks(n);
    if (llama_tokenize(vocab, p.c_str(), p.size(), toks.data(), toks.size(), true, true) < 0) {
        toks.clear();
    }
    return toks;
}

static bool moe_dump_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    const char * name = t->name;
    // DECKL_DUMP_MOE_ALL: dump every ffn_moe_* internal tensor to localize where a
    // compressed-slot run diverges from full. Default: just the final ffn_moe_out.
    static const bool all = getenv("DECKL_DUMP_MOE_ALL") != nullptr;
    const bool match = all ? (strncmp(name, "ffn_moe_", 8) == 0)
                           : (strncmp(name, "ffn_moe_out", 11) == 0);
    if (ask) return match;
    if (match && t->type == GGML_TYPE_F32) {
        static int nlog = 0;
        if (nlog < (all ? 80 : 16)) {
            const int64_t n = ggml_nelements(t);
            std::vector<float> buf(n);
            ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
            double s = 0.0, a2 = 0.0;
            for (int64_t i = 0; i < n; i++) { s += buf[i]; a2 += (double) buf[i] * buf[i]; }
            fprintf(stderr, "MOEDUMP %s ne=[%lld,%lld] sum=%.6f l2=%.6f\n", name,
                    (long long) t->ne[0], (long long) t->ne[1], s, a2);
            nlog++;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    args_t a;
    if (!parse_args(argc, argv, a)) { usage(argv[0]); return 1; }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = a.ngl;
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) { fprintf(stderr, "error: cannot load model %s\n", a.model.c_str()); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int nv = 2*((n_vocab + 1)/2) + 4;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = a.n_ctx;
    cp.n_batch = a.n_ctx;
    if (getenv("DECKL_UBATCH1")) { cp.n_ubatch = 1; }   // TEST: isolate multi-token prefill service
    if (getenv("DECKL_DUMP_MOE")) { cp.cb_eval = moe_dump_cb; cp.cb_eval_user_data = nullptr; }
    cp.no_perf = false;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "error: cannot create context\n"); return 1; }

    std::vector<std::string> prompts = read_prompts(a.prompts, a.n_prompts);
    fprintf(stderr, "decode-kl: %s mode, %zu prompts, n_gen=%d, n_vocab=%d\n",
            a.teacher ? "teacher" : "student", prompts.size(), a.n_gen, n_vocab);

    if (a.teacher) {
        std::ofstream out(a.ref, std::ios::binary);
        if (!out) { fprintf(stderr, "error: cannot open ref for write\n"); return 1; }
        out.write(DECKL_MAGIC, 8);
        out.write((const char *)&n_vocab, sizeof(int32_t));
        int32_t n_written = 0;
        // placeholder for prompt count; patched after the loop
        std::streampos count_pos = out.tellp();
        out.write((const char *)&n_written, sizeof(int32_t));

        std::vector<uint16_t> enc(nv);
        for (size_t pi = 0; pi < prompts.size(); ++pi) {
            std::vector<llama_token> ptoks = tokenize_prompt(vocab, prompts[pi]);
            if (ptoks.empty() || (int) ptoks.size() + a.n_gen > a.n_ctx) {
                fprintf(stderr, "  skip prompt %zu (empty or too long: %zu toks)\n", pi, ptoks.size());
                continue;
            }
            llama_memory_clear(llama_get_memory(ctx), true);

            std::vector<llama_token> gen_tokens;
            std::vector<uint16_t> gen_logprobs; // nv per step, appended

            // feed persists across iterations so the single-token batch pointer stays valid
            llama_token feed = 0;
            bool have_feed = false;
            llama_batch batch = llama_batch_get_one(ptoks.data(), ptoks.size());
            for (int s = 0; s < a.n_gen; ++s) {
                if (have_feed) batch = llama_batch_get_one(&feed, 1);
                if (llama_decode(ctx, batch)) { fprintf(stderr, "  decode failed\n"); return 1; }
                const float * logits = llama_get_logits_ith(ctx, -1);
                llama_token tok = 0;
                float best = logits[0];
                for (int i = 1; i < n_vocab; ++i) if (logits[i] > best) { best = logits[i]; tok = i; }
                if (llama_vocab_is_eog(vocab, tok)) break;
                encode_logprobs(n_vocab, logits, enc.data());
                gen_tokens.push_back(tok);
                gen_logprobs.insert(gen_logprobs.end(), enc.begin(), enc.end());
                feed = tok;
                have_feed = true;
            }

            int32_t n_ptok = (int32_t) ptoks.size();
            int32_t n_steps = (int32_t) gen_tokens.size();
            out.write((const char *)&n_ptok, sizeof(int32_t));
            out.write((const char *)&n_steps, sizeof(int32_t));
            out.write((const char *)ptoks.data(), n_ptok*sizeof(llama_token));
            for (int s = 0; s < n_steps; ++s) {
                out.write((const char *)&gen_tokens[s], sizeof(llama_token));
                out.write((const char *)(gen_logprobs.data() + (size_t) s*nv), nv*sizeof(uint16_t));
            }
            ++n_written;
            if ((pi+1) % 8 == 0) fprintf(stderr, "  teacher: %zu/%zu prompts\n", pi+1, prompts.size());
        }
        out.seekp(count_pos);
        out.write((const char *)&n_written, sizeof(int32_t));
        out.flush();
        if (!out) { fprintf(stderr, "error: failed to finalize ref prompt count\n"); return 1; }
        out.close();
        fprintf(stderr, "decode-kl: wrote %d prompt records to %s\n", n_written, a.ref.c_str());
    } else {
        std::ifstream in(a.ref, std::ios::binary);
        if (!in) { fprintf(stderr, "error: cannot open ref for read\n"); return 1; }
        char magic[8];
        in.read(magic, 8);
        if (memcmp(magic, DECKL_MAGIC, 8) != 0) { fprintf(stderr, "error: bad ref magic\n"); return 1; }
        int32_t ref_vocab = 0, ref_prompts = 0;
        in.read((char *)&ref_vocab, sizeof(int32_t));
        in.read((char *)&ref_prompts, sizeof(int32_t));
        if (!in) { fprintf(stderr, "error: truncated ref header\n"); return 1; }
        if (ref_vocab != n_vocab) { fprintf(stderr, "error: vocab mismatch ref=%d model=%d\n", ref_vocab, n_vocab); return 1; }

        std::ofstream csv;
        if (!a.csv.empty()) { csv.open(a.csv); csv << "prompt,step,kl,p_diff,same_top\n"; }

        std::vector<uint16_t> base(nv);
        std::vector<double> kl_vals;
        std::vector<double> pdiff2_vals;
        size_t n_same = 0, n_tot = 0;

        for (int pi = 0; pi < ref_prompts; ++pi) {
            int32_t n_ptok = 0, n_steps = 0;
            in.read((char *)&n_ptok, sizeof(int32_t));
            in.read((char *)&n_steps, sizeof(int32_t));
            if (!in || n_ptok < 0 || n_steps < 0 || n_ptok > a.n_ctx) {
                fprintf(stderr, "error: truncated/invalid ref at prompt %d (n_ptok=%d n_steps=%d)\n", pi, n_ptok, n_steps);
                return 1;
            }
            std::vector<llama_token> ptoks(n_ptok);
            in.read((char *)ptoks.data(), n_ptok*sizeof(llama_token));
            if (!in) { fprintf(stderr, "error: truncated ref prompt tokens at %d\n", pi); return 1; }

            llama_memory_clear(llama_get_memory(ctx), true);
            llama_batch batch = llama_batch_get_one(ptoks.data(), ptoks.size());
            llama_token feed = 0;   // persists so the single-token batch pointer stays valid
            bool have_feed = false;

            for (int s = 0; s < n_steps; ++s) {
                llama_token tok = 0;
                in.read((char *)&tok, sizeof(llama_token));
                in.read((char *)base.data(), nv*sizeof(uint16_t));
                if (!in) { fprintf(stderr, "error: truncated ref step (prompt %d step %d)\n", pi, s); return 1; }
                if (tok < 0 || tok >= n_vocab) { fprintf(stderr, "error: ref token %d out of range at prompt %d step %d\n", tok, pi, s); return 1; }

                if (have_feed) batch = llama_batch_get_one(&feed, 1);   // teacher forcing
                if (llama_decode(ctx, batch)) { fprintf(stderr, "  decode failed\n"); return 1; }
                const float * logits = llama_get_logits_ith(ctx, -1);
                kl_step_out r = kl_step(n_vocab, logits, base.data(), tok);
                kl_vals.push_back(r.kl);
                pdiff2_vals.push_back(r.p_diff*r.p_diff);
                if (r.same_top) ++n_same;
                ++n_tot;
                if (csv.is_open()) csv << pi << "," << s << "," << r.kl << "," << r.p_diff << "," << (r.same_top?1:0) << "\n";

                feed = tok;
                have_feed = true;
            }
            if ((pi+1) % 8 == 0) fprintf(stderr, "  student: %d/%d prompts\n", pi+1, ref_prompts);
        }

        double kl_sum = 0, pd2_sum = 0;
        for (double v : kl_vals) kl_sum += v;
        for (double v : pdiff2_vals) pd2_sum += v;
        double kl_mean = n_tot ? kl_sum/n_tot : 0.0;
        double rms_pdiff = n_tot ? std::sqrt(pd2_sum/n_tot) : 0.0;
        double same_top_pct = n_tot ? 100.0*n_same/n_tot : 0.0;
        std::vector<double> sorted = kl_vals;
        std::sort(sorted.begin(), sorted.end());
        double kl_median = sorted.empty() ? 0.0 : sorted[sorted.size()/2];
        double kl_p99 = sorted.empty() ? 0.0 : sorted[std::min(sorted.size()-1, (size_t)(0.99*sorted.size()))];

        printf("RESULT tag=%s kl_mean=%.6f kl_median=%.6f kl_p99=%.6f rms_pdiff=%.6f same_top_pct=%.4f count=%zu n_prompts=%d\n",
               a.tag.c_str(), kl_mean, kl_median, kl_p99, rms_pdiff, same_top_pct, n_tot, ref_prompts);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
