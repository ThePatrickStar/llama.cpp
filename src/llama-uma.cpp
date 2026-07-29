#include "llama-uma.h"

#include "llama-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

llama_uma_router::llama_uma_router(llama_uma_policy policy, uint32_t n_cpu_layers, uint32_t n_layer, uint32_t n_expert, uint32_t n_expert_used) :
        policy(policy), n_cpu_layers(n_cpu_layers), n_layer(n_layer), n_expert(n_expert), n_expert_used(n_expert_used) {
    const uint32_t words = (n_expert + 63)/64;
    placement.assign((size_t) n_layer * words, 0);
    placement_prev = placement;
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
