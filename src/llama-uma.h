#pragma once

// uma-moe fork: per-token CPU/GPU expert placement router (research skeleton).
// Milestone 1 interposes the router on the decode path with the constant
// "all experts on GPU" policy: the decision runs every token, and only a
// placement CHANGE invalidates the cached graph schedule, so graph reuse is
// preserved and the overhead budget is the decision bookkeeping alone.
// Milestone 2 adds cpu-static:N - the expert tensors of layers [0, N) run on
// the CPU backend during single-token decode; batches (prefill) stay all-GPU.
// The zero-copy route is per platform: on Metal the weights stay in the GPU
// shared/mapped buffers and the CPU reads them in place; on CUDA the loader
// places them in the device's pinned host buffer type (injected at model
// load, see llama_uma_inject_load_overrides) and the GPU reads them in place.

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

enum llama_uma_policy {
    LLAMA_UMA_POLICY_NONE = 0,
    LLAMA_UMA_POLICY_GPU_ONLY,
    LLAMA_UMA_POLICY_CPU_STATIC,
};

// layout of the policy-designated expert layers, LLAMA_UMA_LAYOUT env (M3:
// layout is a priced action of the cost model, uniform over the designated
// set). default: Metal keeps weights GPU-resident (M2 route), CUDA pins them
// in host memory (Task C route). std: CPU-resident standard layout - decode
// runs the generic vec_dot tier, batches stay staging-eligible (plain CPU
// buft is_host). repack: CPU-resident repacked - the fast CPU decode tier,
// CPU-only readable, prefill collapses to the gemv tier (the priced cost).
enum llama_uma_layout {
    LLAMA_UMA_LAYOUT_DEFAULT = 0,
    LLAMA_UMA_LAYOUT_STD,
    LLAMA_UMA_LAYOUT_REPACK,
};

struct llama_uma_router {
    llama_uma_router(llama_uma_policy policy, uint32_t n_cpu_layers, uint32_t n_layer, uint32_t n_expert, uint32_t n_expert_used);

    // parses "gpu-only" or "cpu-static:N"; returns false on anything else
    static bool parse_policy(const char * s, llama_uma_policy & policy, uint32_t & n_cpu_layers);

    // parses "std" or "repack"; returns false on anything else
    static bool parse_layout(const char * s, llama_uma_layout & layout);

    // per-token placement decision; returns true when the placement changed
    // vs the previous token (caller must then rebuild/re-split the graph)
    bool decide(uint32_t n_tokens);

    // observe() feedback (M3): fed only at existing sync points and on the
    // existing graph-rebuild path - observation must never add a sync
    void observe_pass(uint32_t n_tokens, int64_t t_us);
    void observe_rebuild(int64_t t_us);

    // expert-id channel, env-gated (LLAMA_UMA_OBSERVE=experts), default off.
    // The topk tensors are read back AFTER the graph is synchronized (on UMA
    // this is a small memcpy, on CUDA a small D2H) - decode passes only.
    // The graph callback caches the per-layer topk tensor pointers at every
    // (re)build, so reads never scan the graph by name.
    void observe_experts_cache(int il, ggml_tensor * topk);
    void observe_experts_read();

    // pure function of (policy, il, n_tokens): true when layer il's expert
    // matmuls run on the CPU backend for a batch of n_tokens. Also consulted
    // by the graph callback, including during the context reserve builds, so
    // the tg reserve graph sizes the CPU compute buffer with the pins active.
    bool layer_on_cpu(int il, uint32_t n_tokens) const;

    llama_uma_policy policy;
    llama_uma_layout layout = LLAMA_UMA_LAYOUT_DEFAULT;

    uint32_t n_cpu_layers;

    uint32_t n_layer;
    uint32_t n_expert;
    uint32_t n_expert_used;

    // CUDA and std-layout routes (set at weights-buft registration): expert
    // matmuls of layers [0, n_cpu_layers) are pinned HERE for n_tokens > 1.
    // With host/CPU-resident weights the default assignment is a heuristic
    // (batch-size gated op_offload, and on Metal defeated entirely by the
    // BLAS backend claiming plain-CPU weight buffers); the pin keeps
    // placement policy-owned and per-pass at every batch size. nullptr on
    // the Metal default route (weights device-resident, default already
    // all-GPU) and for repack layout (CPU-only readable by design).
    ggml_backend_t gpu_pin_backend = nullptr;

    // per-layer expert placement bitmap, 1 bit per expert (0 = GPU, 1 = CPU)
    std::vector<uint64_t> placement;
    std::vector<uint64_t> placement_prev;

    int64_t n_decide    = 0;
    int64_t n_replan    = 0;
    int64_t t_decide_us = 0;

    // observe() aggregates
    int64_t t_pp_us      = 0;
    int64_t n_pp_tokens  = 0;
    int64_t n_pp_passes  = 0;
    int64_t t_tg_us      = 0;
    int64_t n_tg_tokens  = 0;
    // times the whole rebuild path (build + alloc), not only placement-caused
    // replans - this is the C-Q4 replan-cost quantity
    int64_t t_rebuild_us = 0;
    int64_t n_rebuild    = 0;

    bool observe_experts = false;

    std::vector<ggml_tensor *> topk_tensors;   // per layer, refreshed by the cb
    std::vector<uint32_t>      expert_freq;    // n_layer x n_expert counts
    std::vector<uint64_t>      expert_cur;     // active-set bitmaps, layer-major
    std::vector<uint64_t>      expert_prev;
    int64_t n_expert_obs = 0;                  // decode tokens observed
    int64_t reuse_num    = 0;                  // sum of |cur & prev| per layer
    int64_t reuse_den    = 0;                  // sum of |cur| per layer
};

// cpu-static:N on a CUDA device needs load-time placement: expert weights of
// layers [0, N) go to the device's pinned host buffer type so both engines
// address the same host-resident bytes (measured on GB10: hostmapped == device
// read speed for the GPU; managed is -25% and never used). Injects
// tensor_buft_overrides into params before llama_model_load; patterns and
// overrides are caller-owned storage that must outlive the load. Disables
// mmap for the load (the loader downgrades host-buft overrides to plain CPU
// under mmap). Returns false on a fatal condition (unparseable policy, user
// tensor overrides alongside the policy, no host buffer type) - the caller
// must fail the load; a typoed policy must never silently measure stock.
bool llama_uma_inject_load_overrides(llama_model_params & params, std::vector<std::string> & patterns, std::vector<llama_model_tensor_buft_override> & overrides);
