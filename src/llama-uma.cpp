#include "llama-uma.h"

#include "llama-impl.h"

#include <algorithm>

llama_uma_router::llama_uma_router(llama_uma_policy policy, uint32_t n_layer, uint32_t n_expert, uint32_t n_expert_used) :
        policy(policy), n_layer(n_layer), n_expert(n_expert), n_expert_used(n_expert_used) {
    const uint32_t words = (n_expert + 63)/64;
    placement.assign((size_t) n_layer * words, 0);
    LLAMA_LOG_INFO("%s: placement router active, policy=gpu-only (n_layer=%u, n_expert=%u, n_expert_used=%u)\n",
            __func__, n_layer, n_expert, n_expert_used);
}

bool llama_uma_router::decide(uint32_t n_tokens) {
    GGML_UNUSED(n_tokens);

    const int64_t t0 = ggml_time_us();

    // gpu-only: every expert of every layer stays on the GPU
    std::fill(placement.begin(), placement.end(), 0);

    // FNV-1a over the placement map; a change forces a graph re-split
    uint64_t fp = 1469598103934665603ULL;
    for (uint64_t w : placement) {
        fp ^= w;
        fp *= 1099511628211ULL;
    }

    const bool changed = n_decide > 0 && fp != fp_prev;

    fp_prev = fp;
    n_decide++;
    t_decide_us += ggml_time_us() - t0;

    return changed;
}
