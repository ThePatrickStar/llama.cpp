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
    LLAMA_UMA_POLICY_AUTO,
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

    // policies that designate CPU layers (cpu-static, and auto once planned)
    bool placement_active() const {
        return policy == LLAMA_UMA_POLICY_CPU_STATIC || policy == LLAMA_UMA_POLICY_AUTO;
    }

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

// device+model profile artifact (M3): flat ASCII "key value" lines, produced
// only by the instruments (scripts/uma-profile.sh merging harness CSVs and
// engine self-calibration). The auto policy reads THIS file plus runtime
// probes and nothing else - no device-name branches anywhere (identity
// fields are checked for provenance, never branched on for decisions).
// dt_* keys are SYNC-AND-COPY-INCLUSIVE per-layer marginals vs the all-GPU
// baseline; t_tok_*/t_pass_* are absolute baselines at the calib sizes.
struct llama_uma_profile {
    // identity (checked; mismatch aborts unless LLAMA_UMA_PROFILE_FORCE=1)
    int64_t  model_file_bytes = 0;
    uint32_t n_layer = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    std::vector<int64_t> expert_bytes_layers;
    // engine calibration
    double  t_tok_gpu_tg_us  = 0.0;
    double  t_pass_gpu_pp_us = 0.0;
    int64_t calib_pp_tokens  = 0;
    int64_t calib_ubatch     = 0;
    double  dt_layer_std_tg_us    = 0.0;
    double  dt_layer_repack_tg_us = 0.0;
    double  dt_layer_std_pp_us    = 0.0;
    double  dt_layer_repack_pp_us = 0.0;
    // load-path tier (WO-A2, 2026-07-31): capacity plans execute -lm none
    // (G1b/WO-B force it), where the DECODE marginal is ~10% cheaper than
    // mmap; pp marginals and gpu baselines are tier-flat
    // (results/wo-a2-loadpath-tier-20260731.md). OPTIONAL schema-v1 pair,
    // all-or-nothing; -1 = absent = price k>0 tg with the mmap marginals.
    double  dt_layer_std_tg_none_us    = -1.0;
    double  dt_layer_repack_tg_none_us = -1.0;
    // carried for reporting/regime checks, not consumed by the v1 planner
    double  tax_hostres_pp_frac_per_layer = 0.0;
    double  replan_cost_ms = 0.0;
    double  decide_cost_us = 0.0;
    // capacity. margin = max(wire_margin_floor_bytes, frac * budget): the
    // measured slack drivers are absolute costs (llama transients ~150 MiB +
    // the shared iogpu pool's other clients), so small budgets need an
    // absolute floor while big budgets keep the fractional margin.
    // wire_margin_floor_bytes is an OPTIONAL schema-v1 key (2026-07-31,
    // WO-A1); absent = 0 = pre-A1 pure-frac behavior.
    double  wire_margin_frac        = 0.0;
    int64_t wire_margin_floor_bytes = 0;
    int64_t gpu_working_set_bytes   = 0;
    int64_t wired_transient_ok      = 0;
    // workload mix defaults
    int64_t lambda_pp_tokens = 512;
    int64_t lambda_tg_tokens = 128;

    static bool load(const char * path, llama_uma_profile & out, std::string & err);
};

struct llama_uma_plan {
    uint32_t         k = 0;
    llama_uma_layout layout = LLAMA_UMA_LAYOUT_DEFAULT;
    int64_t          wire_budget_bytes = 0;
    double           pred_pp_tps = 0.0;
    double           pred_tg_tps = 0.0;
};

// the v1 planner: enumerate k x {std, repack} under the wired-budget
// constraint, argmin J = passes x T_pp_pass + lambda_tg x T_tg_token.
// Pure function of (profile, wire budget) - deterministic, us-scale.
llama_uma_plan llama_uma_plan_compute(const llama_uma_profile & prof, int64_t wire_budget_bytes);

// policy=auto entry: loads LLAMA_UMA_PROFILE, checks identity vs the model
// file (skipped when path_model is null - the context ctor re-plans and
// checks hparams instead), probes the GPU device wire budget (on Metal this
// is recommendedMaxWorkingSetSize, which tracks iogpu.wired_limit_mb - the
// E5 knob), computes the plan and prints the plan evidence line. Fail-closed.
bool llama_uma_auto_plan(const char * path_model, llama_uma_plan & plan, std::string & err, llama_uma_profile * out_prof = nullptr);

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
// policy=auto plans first (llama_uma_auto_plan) and then injects as if
// cpu-static:k with the planned layout.
bool llama_uma_inject_load_overrides(const char * path_model, llama_model_params & params, std::vector<std::string> & patterns, std::vector<llama_model_tensor_buft_override> & overrides);
