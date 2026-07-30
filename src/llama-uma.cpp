#include "llama-uma.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

bool llama_uma_inject_load_overrides(llama_model_params & params, std::vector<std::string> & patterns, std::vector<llama_model_tensor_buft_override> & overrides) {
    const char * env = getenv("LLAMA_UMA_POLICY");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    llama_uma_policy policy       = LLAMA_UMA_POLICY_NONE;
    uint32_t         n_cpu_layers = 0;
    if (!llama_uma_router::parse_policy(env, policy, n_cpu_layers)) {
        fprintf(stderr, "uma: unrecognized LLAMA_UMA_POLICY value '%s' (known: gpu-only, cpu-static:N)\n", env);
        return false;
    }
    if (policy != LLAMA_UMA_POLICY_CPU_STATIC || params.vocab_only) {
        return true;
    }

    // CUDA route only. On Metal the weights are already host-visible in their
    // device buffers and the context-side allowlist handles them; any other
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
    if (dev == nullptr) {
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
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (host_buft == nullptr) {
        fprintf(stderr, "uma: cpu-static needs a pinned host buffer type, none on %s\n", ggml_backend_dev_name(dev));
        return false;
    }
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        fprintf(stderr, "uma: WARNING: %s is not an integrated device - host-resident expert weights are functional-validation-only, wrong physics for placement numbers\n",
                ggml_backend_dev_name(dev));
    }

    // same per-layer pattern --n-cpu-moe uses; reserve first, the c_str()s
    // must stay valid while the loader iterates the override array
    patterns.reserve(n_cpu_layers);
    overrides.reserve(n_cpu_layers + 1);
    for (uint32_t il = 0; il < n_cpu_layers; il++) {
        char pat[64];
        snprintf(pat, sizeof(pat), "blk\\.%u\\.ffn_(up|down|gate|gate_up)_(ch|)exps", il);
        patterns.push_back(pat);
        overrides.push_back({ patterns.back().c_str(), host_buft });
    }
    overrides.push_back({ nullptr, nullptr });
    params.tensor_buft_overrides = overrides.data();

    // the loader downgrades host-buft overrides to plain CPU when mmap is on
    // (pinned host memory cannot be mmap zero-copy)
    if (params.load_mode == LLAMA_LOAD_MODE_MMAP || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK) {
        params.load_mode = params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE;
        fprintf(stderr, "uma: mmap disabled for this load (host-resident expert weights)\n");
    }
    fprintf(stderr, "uma: cpu-static:%u load-time placement: expert weights of layers [0,%u) -> %s\n",
            n_cpu_layers, n_cpu_layers, ggml_backend_buft_name(host_buft));
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
    return policy == LLAMA_UMA_POLICY_CPU_STATIC && n_tokens == 1 && il >= 0 && (uint32_t) il < n_cpu_layers;
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
