#pragma once

// uma-moe fork: per-token CPU/GPU expert placement router (research skeleton).
// Milestone 1 interposes the router on the decode path with the constant
// "all experts on GPU" policy: the decision runs every token, and only a
// placement CHANGE invalidates the cached graph schedule, so graph reuse is
// preserved and the overhead budget is the decision bookkeeping alone.

#include <cstdint>
#include <vector>

enum llama_uma_policy {
    LLAMA_UMA_POLICY_NONE = 0,
    LLAMA_UMA_POLICY_GPU_ONLY,
};

struct llama_uma_router {
    llama_uma_router(llama_uma_policy policy, uint32_t n_layer, uint32_t n_expert, uint32_t n_expert_used);

    // per-token placement decision; returns true when the placement changed
    // vs the previous token (caller must then rebuild/re-split the graph)
    bool decide(uint32_t n_tokens);

    llama_uma_policy policy;

    uint32_t n_layer;
    uint32_t n_expert;
    uint32_t n_expert_used;

    // per-layer expert placement bitmap, 1 bit per expert (0 = GPU, 1 = CPU)
    std::vector<uint64_t> placement;

    uint64_t fp_prev = 0;

    int64_t n_decide    = 0;
    int64_t t_decide_us = 0;
};
