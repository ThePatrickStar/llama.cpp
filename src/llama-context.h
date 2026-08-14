#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-uma.h"
#include "llama-uma-stream.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);
    bool uma_stream_prepare_parked_decode();

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // env: LLAMA_UMA_POLICY (uma-moe fork)
    std::unique_ptr<llama_uma_router> uma_router;

    // uma-moe fork M5 (residency give-back): env LLAMA_UMA_GIVEBACK_K keeps the
    // K hottest experts per layer resident and MADV_DONTNEED-evicts the cold
    // experts' slabs so they re-fault from the file-backed GGUF mmap. -1 = off.
    // Freeze-safety: only engaged with GGML_METAL_NO_RESIDENCY set (ctor gate).
    int32_t uma_giveback_k      = -1;
    int32_t uma_giveback_period = 256;  // decode tokens between give-back sweeps
    int32_t uma_giveback_tick   = 0;    // decode-token counter for the rate limit

    // uma-moe fork M5: evict the cold experts' page ranges (per-expert slab,
    // not per-layer) off the hot-K set read from uma_router->expert_freq.
    // Rate-limited from the decode-only observe point; a no-op unless
    // LLAMA_UMA_GIVEBACK_K is set. Never touches non-host (Metal-resident)
    // buffers, so it stays off the eager-residency-set freeze class.
    void uma_apply_residency();

    // uma-moe fork: register the designated expert weight bufts for CPU
    // in-place reads; must run after every sched (re)creation
    void uma_allow_weights_bufts();

    // uma-moe fork: zero-copy staged reads (C1) - wrap the CPU buffers
    // holding the designated std-layout expert weights in no-rset Metal
    // mapped views so the GPU reads them in place (no split copies); falls
    // back to the staged pin when the GPU backend has no wrap entry point
    void uma_wrap_std_buffers(const std::map<ggml_backend_buffer_t, std::pair<std::vector<ggml_tensor *>, size_t>> & targets);

    // uma-moe fork M5 S1.1.1: expert residency streaming (env LLAMA_UMA_STREAM_K,
    // manifest built in the model). Builds the resident slot pool ONCE (Metal
    // StorageModeShared, no rset) and registers the slot bufts on the current
    // sched; called from uma_allow_weights_bufts after every sched (re)creation.
    // No-op unless model.uma_stream_k() > 0. build_moe_ffn wraps the front-K
    // layers' expert matmuls with a CPU fill op that preads into these slots.
    void uma_stream_setup();
    std::unique_ptr<llama_uma_stream_state> uma_stream;

    // uma-moe fork M6 (give-back controller): elastic runtime resize of the
    // stream slot window n_slots_active in [smin, n_slots ceiling], driven decode-only;
    // PARK alone may set active=n_expert_used below smin.
    // from the post-sync GPU-idle window. Shed = clear the LRU/table entries of
    // the shed slots then MADV_FREE_REUSABLE their pages (drops phys_footprint);
    // grow = re-warm the re-activated slots from the freq ranking. smin is the
    // knee: a hard SLO floor, never crossed while serving (below it => distress).
    int32_t uma_resize_smin    = -1;   // knee floor; -1 = controller off
    int32_t uma_resize_period  = 32;   // decode tokens between controller ticks
    int32_t uma_resize_tick    = 0;    // decode-token counter for the rate limit
    int64_t uma_resize_dtoken  = 0;    // monotonic decode-token counter (SCHED clock)
    // commanded schedule: (decode_token -> target S), sorted ascending by token.
    std::vector<std::pair<int64_t, int32_t>> uma_resize_sched;
    size_t  uma_resize_sched_i = 0;
    // closed-loop (CTRL) watermarks in MiB + step; low <= 0 => CTRL off.
    int32_t uma_resize_lowmib  = 0;    // avail below this => shed one step
    int32_t uma_resize_highmib = 0;    // avail above this => grow one step
    int32_t uma_resize_step    = 16;   // slots per shed/grow step (CTRL)
    // uma-moe fork M7.1 (arbiter actuator): external control-input + telemetry over
    // per-context files. When uma_control_path is set the cross-tenant coordinator drives
    // target S (the local CTRL watermark is bypassed); uma_telemetry_path exports the
    // per-tenant state the arbiter reads (S, miss rate, distress). Empty => today's behavior.
    std::string uma_control_path;
    std::string uma_telemetry_path;
    // resize reallocates the slot buffers, so it needs the GPU device + no-rset wrap
    // entry (resolved once in setup) and forces one graph rebuild afterwards (the
    // reused graph would reference the freed slot tensors).
    ggml_backend_dev_t uma_stream_gpu_dev = nullptr;
    void *  uma_stream_wrap_fn        = nullptr; // buffer_mapped_norset_t, cast in the .cpp
    bool    uma_stream_force_rebuild  = false;   // set by resize; consumed in process_ubatch
    // M7.0 (CUDA/Spark port): the slot pool is Metal (no-rset wrap over mmap) OR CUDA (the
    // pinned host buffer type the GPU reads in place - Task C precedent). Device slots select
    // the CUDA device buft and stage misses through the CUDA host buft. Picked once in setup.
    bool    uma_stream_use_cuda_host  = false;
    ggml_backend_buffer_type_t uma_stream_cuda_host_buft = nullptr;
    bool    uma_stream_use_cuda_device = false;
    ggml_backend_buffer_type_t uma_stream_cuda_device_buft = nullptr;
    ggml_backend_t uma_stream_cuda_backend = nullptr; // async H2D stream; backend not owned here
    void *  uma_stream_vmm_alloc_fn  = nullptr;
    void *  uma_stream_vmm_resize_fn = nullptr;
    void *  uma_stream_vmm_info_fn   = nullptr;
    void *  uma_stream_vmm_stats_fn  = nullptr;
    // Allocate/free ONE GPU-readable slot tensor. out_alloc = mmap length (Metal,
    // munmap on free) or 0 (CUDA, the buffer owns the pinned-host/device allocation).
    // Both CUDA paths retain MATRIX_ROW_PADDING; device buffers use the backend's
    // tensor-aware alloc size + init hook.
    bool uma_stream_alloc_slot_buf(ggml_tensor * slot, size_t data_bytes, ggml_backend_buffer_t * out_buf, void ** out_host, size_t * out_alloc);
    void uma_stream_free_slot_buf(size_t i);
    void uma_stream_resize(uint32_t s_new, bool park = false); // free+realloc the slot buffers to s_new (clamped to the knee unless park: then to n_expert_used, KV-only, no distress)
    bool uma_stream_try_alloc_slots(uint32_t s_new); // (re)allocate all slot buffers at s_new; rolls back its partial allocations and returns false on OOM (precondition: slots freed)
    void uma_stream_reseed_resident(uint32_t s_new); // reseed [0,s_new) from the freq ranking
    void uma_stream_controller_tick();        // rate-limited decode-only driver
    int32_t uma_read_control();               // M7.1: read a target S from the control file (-1 hold, -2 park)
    void    uma_write_telemetry();            // M7.1: atomic write of the per-tenant state

    std::set<ggml_backend_buffer_type_t> uma_bufts_logged;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
