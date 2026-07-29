#include "llama-uma.h"

#include "llama-impl.h"

#include <algorithm>
#include <cstdio>

llama_uma_router::llama_uma_router(llama_uma_policy policy, uint32_t n_layer, uint32_t n_expert, uint32_t n_expert_used) :
        policy(policy), n_layer(n_layer), n_expert(n_expert), n_expert_used(n_expert_used) {
    const uint32_t words = (n_expert + 63)/64;
    placement.assign((size_t) n_layer * words, 0);
    placement_prev = placement;
    // measurement evidence, deliberately NOT via LLAMA_LOG: llama-bench
    // installs a null log callback by default and a record run must still
    // prove the router was engaged
    fprintf(stderr, "uma: placement router active, policy=gpu-only (n_layer=%u, n_expert=%u, n_expert_used=%u)\n",
            n_layer, n_expert, n_expert_used);
}

bool llama_uma_router::decide(uint32_t n_tokens) {
    GGML_UNUSED(n_tokens);

    const int64_t t0 = ggml_time_us();

    // gpu-only: every expert of every layer stays on the GPU
    std::fill(placement.begin(), placement.end(), 0);

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
