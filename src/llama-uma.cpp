#include "llama-uma.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <sys/stat.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

bool llama_uma_stream_static_full_enabled() {
    const char * value = getenv("LLAMA_UMA_STREAM_STATIC_FULL");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool llama_uma_stream_device_slots_enabled() {
    const char * value = getenv("LLAMA_UMA_STREAM_DEVICE_SLOTS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

llama_uma_router::llama_uma_router(llama_uma_policy policy, uint32_t n_cpu_layers, uint32_t n_layer, uint32_t n_expert, uint32_t n_expert_used) :
        policy(policy), n_cpu_layers(n_cpu_layers), n_layer(n_layer), n_expert(n_expert), n_expert_used(n_expert_used) {
    const uint32_t words = (n_expert + 63)/64;
    placement.assign((size_t) n_layer * words, 0);
    placement_prev = placement;
    topk_tensors.assign(n_layer, nullptr);
    expert_freq.assign((size_t) n_layer * n_expert, 0);
    expert_cur.assign((size_t) n_layer * words, 0);
    expert_prev = expert_cur;
    // measurement evidence, deliberately NOT via LLAMA_LOG: llama-bench
    // installs a null log callback by default and a record run must still
    // prove the router was engaged
    char policy_str[64];
    if (policy == LLAMA_UMA_POLICY_CPU_STATIC) {
        snprintf(policy_str, sizeof(policy_str), "cpu-static:%u", n_cpu_layers);
    } else if (policy == LLAMA_UMA_POLICY_AUTO) {
        snprintf(policy_str, sizeof(policy_str), "auto(k=%u)", n_cpu_layers);
    } else {
        snprintf(policy_str, sizeof(policy_str), "gpu-only");
    }
    fprintf(stderr, "uma: placement router active, policy=%s (n_layer=%u, n_expert=%u, n_expert_used=%u)\n",
            policy_str, n_layer, n_expert, n_expert_used);
}

bool llama_uma_router::parse_policy(const char * s, llama_uma_policy & policy, uint32_t & n_cpu_layers) {
    if (strcmp(s, "gpu-only") == 0) {
        policy       = LLAMA_UMA_POLICY_GPU_ONLY;
        n_cpu_layers = 0;
        return true;
    }
    if (strcmp(s, "auto") == 0) {
        policy       = LLAMA_UMA_POLICY_AUTO;
        n_cpu_layers = 0;
        return true;
    }
    const char * prefix = "cpu-static:";
    if (strncmp(s, prefix, strlen(prefix)) == 0) {
        const char * num = s + strlen(prefix);
        char * end = nullptr;
        const unsigned long long v = strtoull(num, &end, 10);
        if (end == num || *end != '\0' || v == 0 || v > UINT32_MAX) {
            return false;
        }
        policy       = LLAMA_UMA_POLICY_CPU_STATIC;
        n_cpu_layers = (uint32_t) v;
        return true;
    }
    return false;
}

bool llama_uma_router::parse_layout(const char * s, llama_uma_layout & layout) {
    if (strcmp(s, "std") == 0) {
        layout = LLAMA_UMA_LAYOUT_STD;
        return true;
    }
    if (strcmp(s, "repack") == 0) {
        layout = LLAMA_UMA_LAYOUT_REPACK;
        return true;
    }
    return false;
}

// strict flat "key value" parser: '#' comments and blank lines only; every
// other line must be a known key; every required key must appear exactly
// once. A typo in a profile must never silently plan something else.
bool llama_uma_profile::load(const char * path, llama_uma_profile & out, std::string & err) {
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        err = std::string("cannot open profile '") + path + "'";
        return false;
    }
    std::map<std::string, std::string> kv;
    char line[4096];
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char * s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (*s == '#' || *s == '\n' || *s == '\0') {
            continue;
        }
        char * nl = strchr(s, '\n');
        if (nl) *nl = '\0';
        char * sp = strchr(s, ' ');
        if (sp == nullptr) {
            err = "line " + std::to_string(lineno) + ": expected 'key value'";
            fclose(f);
            return false;
        }
        *sp = '\0';
        if (!kv.emplace(s, sp + 1).second) {
            err = "line " + std::to_string(lineno) + ": duplicate key '" + s + "'";
            fclose(f);
            return false;
        }
    }
    fclose(f);

    static const char * informational[] = { "produced_by", "date", "device_name", "backend_kind", "model_file_name" };
    auto take = [&](const char * key, std::string & val) -> bool {
        auto it = kv.find(key);
        if (it == kv.end()) {
            err = std::string("missing required key '") + key + "'";
            return false;
        }
        val = it->second;
        kv.erase(it);
        return true;
    };
    std::string v;
    auto take_i64 = [&](const char * key, int64_t & dst) -> bool {
        if (!take(key, v)) return false;
        char * end = nullptr;
        dst = strtoll(v.c_str(), &end, 10);
        if (end == v.c_str() || *end != '\0') {
            err = std::string("key '") + key + "': not an integer: '" + v + "'";
            return false;
        }
        return true;
    };
    auto take_f64 = [&](const char * key, double & dst) -> bool {
        if (!take(key, v)) return false;
        char * end = nullptr;
        dst = strtod(v.c_str(), &end);
        if (end == v.c_str() || *end != '\0') {
            err = std::string("key '") + key + "': not a number: '" + v + "'";
            return false;
        }
        return true;
    };

    int64_t version = 0;
    if (!take_i64("uma_profile_version", version)) return false;
    if (version != 1) {
        err = "unsupported uma_profile_version " + std::to_string(version);
        return false;
    }
    for (const char * key : informational) {
        kv.erase(key);
    }
    int64_t n_layer = 0, n_expert = 0, n_expert_used = 0;
    if (!take_i64("model_file_bytes", out.model_file_bytes)) return false;
    if (!take_i64("n_layer", n_layer) || !take_i64("n_expert", n_expert) || !take_i64("n_expert_used", n_expert_used)) return false;
    out.n_layer       = (uint32_t) n_layer;
    out.n_expert      = (uint32_t) n_expert;
    out.n_expert_used = (uint32_t) n_expert_used;
    if (!take("expert_bytes_layers", v)) return false;
    {
        const char * p = v.c_str();
        while (*p) {
            char * end = nullptr;
            const int64_t b = strtoll(p, &end, 10);
            if (end == p) {
                err = "expert_bytes_layers: malformed list";
                return false;
            }
            out.expert_bytes_layers.push_back(b);
            p = end;
            if (*p == ',') p++;
        }
        if (out.expert_bytes_layers.size() != out.n_layer) {
            err = "expert_bytes_layers: " + std::to_string(out.expert_bytes_layers.size()) + " entries for n_layer " + std::to_string(out.n_layer);
            return false;
        }
    }
    if (!take_f64("t_tok_gpu_tg_us",       out.t_tok_gpu_tg_us))       return false;
    if (!take_f64("t_pass_gpu_pp_us",      out.t_pass_gpu_pp_us))      return false;
    if (!take_i64("calib_pp_tokens",       out.calib_pp_tokens))       return false;
    if (!take_i64("calib_ubatch",          out.calib_ubatch))          return false;
    if (!take_f64("dt_layer_std_tg_us",    out.dt_layer_std_tg_us))    return false;
    if (!take_f64("dt_layer_repack_tg_us", out.dt_layer_repack_tg_us)) return false;
    if (!take_f64("dt_layer_std_pp_us",    out.dt_layer_std_pp_us))    return false;
    if (!take_f64("dt_layer_repack_pp_us", out.dt_layer_repack_pp_us)) return false;
    // OPTIONAL pair (schema v1 + A2, 2026-07-31): none-tier decode marginals.
    // All-or-nothing - a lone key means a truncated/mis-merged profile.
    {
        const bool has_std    = kv.count("dt_layer_std_tg_none_us")    != 0;
        const bool has_repack = kv.count("dt_layer_repack_tg_none_us") != 0;
        if (has_std != has_repack) {
            err = "dt_layer_std_tg_none_us and dt_layer_repack_tg_none_us must appear together";
            return false;
        }
        if (has_std) {
            if (!take_f64("dt_layer_std_tg_none_us",    out.dt_layer_std_tg_none_us))    return false;
            if (!take_f64("dt_layer_repack_tg_none_us", out.dt_layer_repack_tg_none_us)) return false;
        }
    }
    if (!take_f64("tax_hostres_pp_frac_per_layer", out.tax_hostres_pp_frac_per_layer)) return false;
    if (!take_f64("replan_cost_ms",        out.replan_cost_ms))        return false;
    if (!take_f64("decide_cost_us",        out.decide_cost_us))        return false;
    if (!take_f64("wire_margin_frac",      out.wire_margin_frac))      return false;
    // OPTIONAL (schema v1 + A1, 2026-07-31): absent = 0 = pure-frac margin.
    // Old profiles round-trip unchanged; a profile carrying this key on a
    // pre-A1 binary fails closed via the unknown-key check, as intended.
    if (kv.count("wire_margin_floor_bytes") != 0) {
        if (!take_i64("wire_margin_floor_bytes", out.wire_margin_floor_bytes)) return false;
    }
    if (!take_i64("gpu_working_set_bytes", out.gpu_working_set_bytes)) return false;
    if (!take_i64("wired_transient_ok",    out.wired_transient_ok))    return false;
    if (!take_i64("lambda_pp_tokens",      out.lambda_pp_tokens))      return false;
    if (!take_i64("lambda_tg_tokens",      out.lambda_tg_tokens))      return false;
    if (!kv.empty()) {
        err = "unknown key '" + kv.begin()->first + "' (schema v1 is strict)";
        return false;
    }
    if (out.t_tok_gpu_tg_us <= 0 || out.t_pass_gpu_pp_us <= 0 || out.calib_ubatch <= 0 || out.calib_pp_tokens <= 0 || out.gpu_working_set_bytes <= 0) {
        err = "non-positive calibration baseline";
        return false;
    }
    // a negative marginal means calibration noise swamped the effect - the
    // planner would treat exclusion as free-or-better and overreach
    if (out.dt_layer_std_tg_us < 0 || out.dt_layer_repack_tg_us < 0 || out.dt_layer_std_pp_us < 0 || out.dt_layer_repack_pp_us < 0) {
        err = "negative per-layer marginal (noisy calibration - recalibrate)";
        return false;
    }
    // -1 exactly = absent sentinel; anything else negative is a bad profile
    if ((out.dt_layer_std_tg_none_us    != -1.0 && out.dt_layer_std_tg_none_us    < 0) ||
        (out.dt_layer_repack_tg_none_us != -1.0 && out.dt_layer_repack_tg_none_us < 0)) {
        err = "negative per-layer marginal (noisy calibration - recalibrate)";
        return false;
    }
    if (out.lambda_pp_tokens < 0 || out.lambda_tg_tokens < 0 || out.lambda_pp_tokens + out.lambda_tg_tokens == 0) {
        err = "invalid workload mix (lambda)";
        return false;
    }
    if (out.wire_margin_frac < 0.0 || out.wire_margin_frac >= 1.0) {
        err = "wire_margin_frac outside [0,1)";
        return false;
    }
    if (out.wire_margin_floor_bytes < 0) {
        err = "negative wire_margin_floor_bytes";
        return false;
    }
    for (const int64_t b : out.expert_bytes_layers) {
        if (b <= 0) {
            err = "non-positive expert layer bytes";
            return false;
        }
    }
    return true;
}

llama_uma_plan llama_uma_plan_compute(const llama_uma_profile & prof, int64_t wire_budget_bytes) {
    llama_uma_plan best;
    best.wire_budget_bytes = wire_budget_bytes;
    // margin = max(absolute floor, fractional): the measured slack need is
    // absolute (llama transients + shared-pool clients; +256 MiB headroom
    // failed bench pp, +512 passed - results/e5-cq1-c16.md), so a bare
    // fraction under-margins small capacity budgets
    const double margin = std::max((double) prof.wire_margin_floor_bytes, (double) wire_budget_bytes * prof.wire_margin_frac);
    const double budget = (double) wire_budget_bytes - margin;
    const double passes = (double) ((prof.lambda_pp_tokens + prof.calib_ubatch - 1) / prof.calib_ubatch);

    double best_j = -1.0;
    int64_t excluded_bytes = 0;
    for (uint32_t k = 0; k <= prof.n_layer; k++) {
        if ((double) prof.gpu_working_set_bytes - (double) excluded_bytes <= budget) {
            if (k == 0) {
                const double t_pp = prof.t_pass_gpu_pp_us;
                const double t_tg = prof.t_tok_gpu_tg_us;
                best_j = passes * t_pp + (double) prof.lambda_tg_tokens * t_tg;
                best.k = 0;
                best.layout = LLAMA_UMA_LAYOUT_DEFAULT;
                best.pred_pp_tps = 1e6 * (double) prof.calib_pp_tokens / t_pp;
                best.pred_tg_tps = 1e6 / t_tg;
            } else {
                // k>0 = capacity plan = -lm none at load (G1b/WO-B): price
                // decode with the none-tier marginals when calibrated. pp
                // marginals and baselines are tier-flat (WO-A2).
                const double dt_tg_std    = prof.dt_layer_std_tg_none_us    >= 0.0 ? prof.dt_layer_std_tg_none_us    : prof.dt_layer_std_tg_us;
                const double dt_tg_repack = prof.dt_layer_repack_tg_none_us >= 0.0 ? prof.dt_layer_repack_tg_none_us : prof.dt_layer_repack_tg_us;
                const struct { llama_uma_layout layout; double dt_pp, dt_tg; } cands[] = {
                    { LLAMA_UMA_LAYOUT_STD,    prof.dt_layer_std_pp_us,    dt_tg_std    },
                    { LLAMA_UMA_LAYOUT_REPACK, prof.dt_layer_repack_pp_us, dt_tg_repack },
                };
                for (const auto & c : cands) {
                    const double t_pp = prof.t_pass_gpu_pp_us + k * c.dt_pp;
                    const double t_tg = prof.t_tok_gpu_tg_us  + k * c.dt_tg;
                    const double j    = passes * t_pp + (double) prof.lambda_tg_tokens * t_tg;
                    if (best_j < 0.0 || j < best_j) {
                        best_j = j;
                        best.k = k;
                        best.layout = c.layout;
                        best.pred_pp_tps = 1e6 * (double) prof.calib_pp_tokens / t_pp;
                        best.pred_tg_tps = 1e6 / t_tg;
                    }
                }
            }
            // marginals are non-negative in every measured regime, so J is
            // non-decreasing in k past the first feasible point
            break;
        }
        excluded_bytes += prof.expert_bytes_layers[k];
    }
    if (best_j < 0.0) {
        best.k = prof.n_layer + 1;   // sentinel: infeasible even fully excluded
    }
    return best;
}

bool llama_uma_auto_plan(const char * path_model, llama_uma_plan & plan, std::string & err, llama_uma_profile * out_prof) {
    const char * prof_path = getenv("LLAMA_UMA_PROFILE");
    if (prof_path == nullptr || prof_path[0] == '\0') {
        err = "LLAMA_UMA_POLICY=auto requires LLAMA_UMA_PROFILE=<file>";
        return false;
    }
    llama_uma_profile prof;
    if (!llama_uma_profile::load(prof_path, prof, err)) {
        return false;
    }
    const char * force = getenv("LLAMA_UMA_PROFILE_FORCE");
    const bool forced = force != nullptr && strcmp(force, "1") == 0;
    if (path_model != nullptr) {
        struct stat st;
        if (stat(path_model, &st) != 0) {
            err = std::string("cannot stat model file '") + path_model + "' for the profile identity check";
            return false;
        }
        if ((int64_t) st.st_size != prof.model_file_bytes) {
            if (!forced) {
                err = "profile is for a model of " + std::to_string(prof.model_file_bytes) + " bytes, this file is " + std::to_string((int64_t) st.st_size) + " (set LLAMA_UMA_PROFILE_FORCE=1 only for the wrong-profile ablation)";
                return false;
            }
            fprintf(stderr, "uma: WARNING: profile/model identity mismatch FORCED - ablation mode, numbers are not placement results\n");
        }
    }
    // wire budget = the first GPU device's reported total. On Metal this is
    // recommendedMaxWorkingSetSize (tracks iogpu.wired_limit_mb, the E5
    // knob); on CUDA/GB10 it is the device memory total.
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(d);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu = d;
            break;
        }
    }
    if (gpu == nullptr) {
        err = "policy=auto needs a GPU device to plan against";
        return false;
    }
    size_t mem_free = 0, mem_total = 0;
    ggml_backend_dev_memory(gpu, &mem_free, &mem_total);
    if (mem_total == 0) {
        err = "GPU device reports no memory total";
        return false;
    }
    plan = llama_uma_plan_compute(prof, (int64_t) mem_total);
    if (plan.k > prof.n_layer) {
        err = "no feasible plan: model exceeds the wire budget even with all expert layers excluded";
        return false;
    }
    const char * lay_str = plan.layout == LLAMA_UMA_LAYOUT_STD ? "std" : plan.layout == LLAMA_UMA_LAYOUT_REPACK ? "repack" : "gpu-std";
    fprintf(stderr, "uma: auto plan: wire_budget=%lld MiB k=%u layout=%s predicted pp=%.1f t/s tg=%.1f t/s\n",
            (long long) (plan.wire_budget_bytes / (1024*1024)), plan.k, lay_str, plan.pred_pp_tps, plan.pred_tg_tps);
    if (out_prof != nullptr) {
        *out_prof = prof;
    }
    return true;
}

bool llama_uma_inject_load_overrides(const char * path_model, llama_model_params & params, std::vector<std::string> & patterns, std::vector<llama_model_tensor_buft_override> & overrides) {
    // M5 S1.1.3 streaming footprint give-back: route the front-K layers' expert
    // weights off the pinned Metal weights buffer into a dedicated CPU buffer, and
    // force no-mmap (a Metal mmap buffer's get_mapping_range span would wire the
    // excluded pages -> the E5 freeze). After the context builds its resident slot
    // pool, uma_stream_free_excluded() frees this buffer, so only the S slots stay
    // resident and the cold experts stream from the fd on demand.
    const char * env_stream = getenv("LLAMA_UMA_STREAM_K");
    const bool device_slots = llama_uma_stream_device_slots_enabled();
    if (device_slots) {
        const char * lazy = getenv("LLAMA_UMA_STREAM_LAZYLOAD");
        if (env_stream == nullptr || env_stream[0] == '\0') {
            fprintf(stderr, "uma: LLAMA_UMA_STREAM_DEVICE_SLOTS requires LLAMA_UMA_STREAM_K\n");
            return false;
        }
        if (llama_uma_stream_static_full_enabled()) {
            fprintf(stderr, "uma: LLAMA_UMA_STREAM_DEVICE_SLOTS and LLAMA_UMA_STREAM_STATIC_FULL are mutually exclusive\n");
            return false;
        }
        if (lazy == nullptr || lazy[0] == '\0' || lazy[0] == '0') {
            fprintf(stderr, "uma: LLAMA_UMA_STREAM_DEVICE_SLOTS requires LLAMA_UMA_STREAM_LAZYLOAD=1 (refusing full-expert + device-pool load transient)\n");
            return false;
        }
    }
    if (env_stream != nullptr && env_stream[0] != '\0' && !params.vocab_only) {
        char * end = nullptr;
        const long k = strtol(env_stream, &end, 10);
        if (end == env_stream || *end != '\0' || k <= 0 || k > 4096) {
            fprintf(stderr, "uma: invalid LLAMA_UMA_STREAM_K '%s' (want 1..4096)\n", env_stream);
            return false;
        }
        if (getenv("LLAMA_UMA_POLICY") != nullptr) {
            fprintf(stderr, "uma: LLAMA_UMA_STREAM_K and LLAMA_UMA_POLICY are mutually exclusive\n");
            return false;
        }
        if (params.tensor_buft_overrides != nullptr && params.tensor_buft_overrides[0].pattern != nullptr) {
            fprintf(stderr, "uma: refusing to mix tensor overrides with LLAMA_UMA_STREAM_K\n");
            return false;
        }
        if (llama_uma_stream_static_full_enabled()) {
            // The loader does not have model hparams yet. Preserve the exact stock
            // placement here; uma_stream_build_manifest validates S==SMAX==n_expert
            // once hparams are known, before a context can be created.
            fprintf(stderr, "uma: stream static-full requested: preserving stock contiguous load path (pending n_expert validation)\n");
            return true;
        }
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        patterns.reserve((size_t) k);
        overrides.reserve((size_t) k + 1);
        for (long il = 0; il < k; il++) {
            char pat[64];
            snprintf(pat, sizeof(pat), "blk\\.%ld\\.ffn_(up|down|gate|gate_up)_(ch|)exps", il);
            patterns.push_back(pat);
            overrides.push_back({ patterns.back().c_str(), cpu_buft });
        }
        overrides.push_back({ nullptr, nullptr });
        params.tensor_buft_overrides = overrides.data();
        params.use_extra_bufts = false; // plain CPU std layout (not repacked)
        // no_host: keep the excluded experts on PLAIN CPU, NOT the CUDA pinned-host buffer type.
        // The loader's CPU override "considers extra buffer types" (make_cpu_buft_list) and would
        // otherwise upgrade MXFP4 experts to CUDA_Host (cudaHostAlloc), which COMMITS pinned memory
        // at allocation (unlike lazy malloc) - so even under LAZYLOAD the full ~58 GB expert buffer
        // stays resident alongside the pinned slot pool = the double-allocation. Plain CPU pages are
        // lazy (unread -> 0 resident) and madvise-freeable. The GPU-read copy is the slot pool, so
        // these excluded expert tensors are never computed here. (Slot pool buft is set separately in
        // the context via uma_stream_cuda_host_buft and is unaffected by no_host.)
        params.no_host = true;
        if (params.load_mode == LLAMA_LOAD_MODE_MMAP || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK) {
            params.load_mode = params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE;
        }
        const char * lz = getenv("LLAMA_UMA_STREAM_LAZYLOAD");
        const bool lazy = lz != nullptr && lz[0] != '\0' && lz[0] != '0';
        fprintf(stderr, "uma: stream K=%ld: front-layer experts -> CPU buffer (freeable) + load-mode %s%s\n",
                k, llama_load_mode_name(params.load_mode),
                lazy ? " + LAZYLOAD (experts not read into RAM; slot pool is the sole copy - load transient ~= slot-pool size)" : "");
        return true;
    }

    const char * env = getenv("LLAMA_UMA_POLICY");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    llama_uma_policy policy       = LLAMA_UMA_POLICY_NONE;
    uint32_t         n_cpu_layers = 0;
    if (!llama_uma_router::parse_policy(env, policy, n_cpu_layers)) {
        fprintf(stderr, "uma: unrecognized LLAMA_UMA_POLICY value '%s' (known: gpu-only, cpu-static:N, auto)\n", env);
        return false;
    }
    llama_uma_layout layout = LLAMA_UMA_LAYOUT_DEFAULT;
    const char * env_layout = getenv("LLAMA_UMA_LAYOUT");
    if (env_layout != nullptr && env_layout[0] != '\0') {
        if (!llama_uma_router::parse_layout(env_layout, layout)) {
            fprintf(stderr, "uma: unrecognized LLAMA_UMA_LAYOUT value '%s' (known: std, repack)\n", env_layout);
            return false;
        }
        if (policy != LLAMA_UMA_POLICY_CPU_STATIC) {
            fprintf(stderr, "uma: LLAMA_UMA_LAYOUT requires LLAMA_UMA_POLICY=cpu-static:N (auto plans its own layout)\n");
            return false;
        }
    }
    if (policy == LLAMA_UMA_POLICY_AUTO && !params.vocab_only) {
        llama_uma_plan plan;
        std::string err;
        if (!llama_uma_auto_plan(path_model, plan, err)) {
            fprintf(stderr, "uma: auto plan failed: %s\n", err.c_str());
            return false;
        }
        if (plan.k == 0) {
            return true;
        }
        policy       = LLAMA_UMA_POLICY_CPU_STATIC;
        n_cpu_layers = plan.k;
        layout       = plan.layout;
    }
    if (policy != LLAMA_UMA_POLICY_CPU_STATIC || params.vocab_only) {
        return true;
    }

    // CUDA device discovery. On Metal the default route keeps the weights in
    // their device buffers (host-visible, context-side allowlist); any other
    // backend fails closed there too. Explicit name check, not a property
    // heuristic: a denylist here failed open for CUDA VRAM once already.
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(d);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
        if (reg != nullptr && strcmp(ggml_backend_reg_name(reg), "CUDA") == 0) {
            dev = d;
            break;
        }
    }

    // route per (layout, platform):
    //   default: CUDA -> pinned host; Metal -> no injection (M2 route)
    //   std:     CUDA -> pinned host (same bytes, host-resident either way);
    //            Metal -> plain CPU + extra bufts off (standard layout,
    //            staging-eligible; load-mode forced off mmap below)
    //   repack:  plain CPU + no_host on any platform (re-resolution then
    //            lands on the CPU repack tier; verified: the loader honors
    //            non-CPU override bufts verbatim and re-resolves only the
    //            plain CPU buft through the buft list)
    const bool cpu_route = layout == LLAMA_UMA_LAYOUT_REPACK || (layout == LLAMA_UMA_LAYOUT_STD && dev == nullptr);
    if (dev == nullptr && !cpu_route) {
        return true;
    }

    if (params.tensor_buft_overrides != nullptr && params.tensor_buft_overrides[0].pattern != nullptr) {
        fprintf(stderr, "uma: refusing to mix tensor overrides with LLAMA_UMA_POLICY=cpu-static (from -ot/--n-cpu-moe, or from automatic -fit overflow placement - pass -fit off)\n");
        return false;
    }
    // sanity bound only - the context ctor enforces the strict n_layer check;
    // this stops an absurd N from turning into a bad_alloc during reserve
    if (n_cpu_layers > 4096) {
        fprintf(stderr, "uma: cpu-static:%u is not a plausible layer count\n", n_cpu_layers);
        return false;
    }

    ggml_backend_buffer_type_t inject_buft = nullptr;
    if (cpu_route) {
        inject_buft = ggml_backend_cpu_buffer_type();
        if (layout == LLAMA_UMA_LAYOUT_REPACK) {
            params.no_host = true;
        } else {
            params.use_extra_bufts = false;
        }
        // fail-closed (design rule #7): a k>0 capacity plan must never
        // mmap-load - on Metal the weight buffer spans get_mapping_range()
        // over the Metal-assigned tensors, interleaved exclusion makes that
        // span ~the whole file, and the residency set wires it eagerly at
        // creation (results/e5-cq1-20260731-freeze.md: machine-fatal, not
        // slow). Keyed on cpu_route, not a backend-name test: load-mode
        // none is safe everywhere, a Metal allowlist could fail open.
        if (n_cpu_layers > 0 && (params.load_mode == LLAMA_LOAD_MODE_MMAP || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK)) {
            params.load_mode = params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE;
            fprintf(stderr, "uma: forcing load-mode %s for capacity plan (mmap span would wire the full file - see design rule #7)\n",
                    llama_load_mode_name(params.load_mode));
        }
    } else {
        inject_buft = ggml_backend_dev_host_buffer_type(dev);
        if (inject_buft == nullptr) {
            fprintf(stderr, "uma: cpu-static needs a pinned host buffer type, none on %s\n", ggml_backend_dev_name(dev));
            return false;
        }
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            fprintf(stderr, "uma: WARNING: %s is not an integrated device - host-resident expert weights are functional-validation-only, wrong physics for placement numbers\n",
                    ggml_backend_dev_name(dev));
        }
    }

    // same per-layer pattern --n-cpu-moe uses; reserve first, the c_str()s
    // must stay valid while the loader iterates the override array
    patterns.reserve(n_cpu_layers);
    overrides.reserve(n_cpu_layers + 1);
    for (uint32_t il = 0; il < n_cpu_layers; il++) {
        char pat[64];
        snprintf(pat, sizeof(pat), "blk\\.%u\\.ffn_(up|down|gate|gate_up)_(ch|)exps", il);
        patterns.push_back(pat);
        overrides.push_back({ patterns.back().c_str(), inject_buft });
    }
    overrides.push_back({ nullptr, nullptr });
    params.tensor_buft_overrides = overrides.data();

    if (cpu_route) {
        if (layout == LLAMA_UMA_LAYOUT_REPACK) {
            fprintf(stderr, "uma: layout repack: expert weights of layers [0,%u) -> CPU repack tier (no_host)\n", n_cpu_layers);
        } else {
            fprintf(stderr, "uma: layout std: expert weights of layers [0,%u) -> CPU standard layout (extra bufts off)\n", n_cpu_layers);
        }
    } else {
        // the loader downgrades host-buft overrides to plain CPU when mmap is
        // on (pinned host memory cannot be mmap zero-copy)
        if (params.load_mode == LLAMA_LOAD_MODE_MMAP || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK) {
            params.load_mode = params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE;
            fprintf(stderr, "uma: mmap disabled for this load (host-resident expert weights)\n");
        }
        fprintf(stderr, "uma: cpu-static:%u load-time placement: expert weights of layers [0,%u) -> %s\n",
                n_cpu_layers, n_cpu_layers, ggml_backend_buft_name(inject_buft));
    }
    return true;
}

void llama_uma_router::observe_pass(uint32_t n_tokens, int64_t t_us) {
    if (n_tokens == 1) {
        t_tg_us += t_us;
        n_tg_tokens++;
    } else {
        t_pp_us += t_us;
        n_pp_tokens += n_tokens;
        n_pp_passes++;
    }
}

void llama_uma_router::observe_rebuild(int64_t t_us) {
    t_rebuild_us += t_us;
    n_rebuild++;
}

void llama_uma_router::observe_experts_cache(int il, ggml_tensor * topk) {
    if (il >= 0 && (uint32_t) il < n_layer) {
        topk_tensors[il] = topk;
    }
}

void llama_uma_router::observe_experts_read() {
    const uint32_t words = (n_expert + 63)/64;
    std::vector<int32_t> ids(n_expert_used);
    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor * t = topk_tensors[il];
        if (t == nullptr) {
            continue;
        }
        // garbage here corrupts every downstream locality stat - abort loud
        if (t->type != GGML_TYPE_I32 || t->ne[0] < (int64_t) n_expert_used) {
            GGML_ABORT("uma: observe experts: topk tensor %s has type %s ne0 %" PRId64 ", expected i32 x %u", t->name, ggml_type_name(t->type), t->ne[0], n_expert_used);
        }
        ggml_backend_tensor_get(t, ids.data(), 0, n_expert_used * sizeof(int32_t));
        uint64_t * cur  = expert_cur.data()  + (size_t) il * words;
        uint64_t * prev = expert_prev.data() + (size_t) il * words;
        memset(cur, 0, words * sizeof(uint64_t));
        for (uint32_t e = 0; e < n_expert_used; e++) {
            const int32_t id = ids[e];
            if (id < 0 || (uint32_t) id >= n_expert) {
                GGML_ABORT("uma: observe experts: layer %u expert id %d out of range [0,%u)", il, id, n_expert);
            }
            cur[id / 64] |= 1ull << (id % 64);
            expert_freq[(size_t) il * n_expert + id]++;
        }
        if (n_expert_obs > 0) {
            for (uint32_t w = 0; w < words; w++) {
                reuse_num += __builtin_popcountll(cur[w] & prev[w]);
                reuse_den += __builtin_popcountll(cur[w]);
            }
        }
        for (uint32_t w = 0; w < words; w++) {
            prev[w] = cur[w];
        }
    }
    n_expert_obs++;
}

bool llama_uma_router::layer_on_cpu(int il, uint32_t n_tokens) const {
    // single-token decode only: batches (prefill) stay all-GPU, which is the
    // per-pass freedom a load-time split cannot express
    return placement_active() && n_tokens == 1 && il >= 0 && (uint32_t) il < n_cpu_layers;
}

bool llama_uma_router::decide(uint32_t n_tokens) {
    const int64_t t0 = ggml_time_us();

    const uint32_t words = (n_expert + 63)/64;
    const uint32_t tail  = n_expert % 64;
    for (uint32_t il = 0; il < n_layer; il++) {
        const uint64_t fill = layer_on_cpu((int) il, n_tokens) ? ~0ull : 0ull;
        for (uint32_t w = 0; w < words; w++) {
            uint64_t mask = ~0ull;
            if (w == words - 1 && tail != 0) {
                mask = (1ull << tail) - 1;
            }
            placement[(size_t) il * words + w] = fill & mask;
        }
    }

    // exact compare, no fingerprint: a missed replan on a hash collision
    // would silently decode with a stale placement in later milestones
    const bool changed = n_decide > 0 && placement != placement_prev;

    placement_prev = placement;
    n_decide++;
    if (changed) {
        n_replan++;
    }
    t_decide_us += ggml_time_us() - t0;

    return changed;
}
