#pragma once

// uma-moe fork: per-token CPU/GPU expert placement router (research skeleton).
// Milestone 1 interposes the router on the decode path with the constant
// "all experts on GPU" policy: the decision runs every token, and only a
// placement CHANGE invalidates the cached graph schedule, so graph reuse is
// preserved and the overhead budget is the decision bookkeeping alone.
// Milestone 2 adds cpu-static:N - the expert tensors of layers [0, N) run on
// the CPU backend during single-token decode, read in place from the Metal
// shared/mapped weight buffers (zero-copy); batches (prefill) stay all-GPU.

#include <cstdint>
#include <vector>

enum llama_uma_policy {
    LLAMA_UMA_POLICY_NONE = 0,
    LLAMA_UMA_POLICY_GPU_ONLY,
    LLAMA_UMA_POLICY_CPU_STATIC,
};

struct llama_uma_router {
    llama_uma_router(llama_uma_policy policy, uint32_t n_cpu_layers, uint32_t n_layer, uint32_t n_expert, uint32_t n_expert_used);

    // parses "gpu-only" or "cpu-static:N"; returns false on anything else
    static bool parse_policy(const char * s, llama_uma_policy & policy, uint32_t & n_cpu_layers);

    // per-token placement decision; returns true when the placement changed
    // vs the previous token (caller must then rebuild/re-split the graph)
    bool decide(uint32_t n_tokens);

    // pure function of (policy, il, n_tokens): true when layer il's expert
    // matmuls run on the CPU backend for a batch of n_tokens. Also consulted
    // by the graph callback, including during the context reserve builds, so
    // the tg reserve graph sizes the CPU compute buffer with the pins active.
    bool layer_on_cpu(int il, uint32_t n_tokens) const;

    llama_uma_policy policy;

    uint32_t n_cpu_layers;

    uint32_t n_layer;
    uint32_t n_expert;
    uint32_t n_expert_used;

    // per-layer expert placement bitmap, 1 bit per expert (0 = GPU, 1 = CPU)
    std::vector<uint64_t> placement;
    std::vector<uint64_t> placement_prev;

    int64_t n_decide    = 0;
    int64_t n_replan    = 0;
    int64_t t_decide_us = 0;
};
