#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-uma.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

//
// llama_context
//

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

struct llama_uma_stream_s_config {
    uint32_t initial;
    uint32_t ceiling;
};

static long llama_uma_stream_env_long(const char * name, long fallback) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(format("invalid %s '%s' (want an integer)", name, value));
    }
    return parsed;
}

static llama_uma_stream_s_config llama_uma_stream_parse_s(uint32_t n_expert_used, uint32_t n_expert) {
    const bool have_initial = []() {
        const char * v = getenv("LLAMA_UMA_STREAM_S");
        return v != nullptr && v[0] != '\0';
    }();
    const bool have_ceiling = []() {
        const char * v = getenv("LLAMA_UMA_STREAM_SMAX");
        return v != nullptr && v[0] != '\0';
    }();
    const long ceiling_env = llama_uma_stream_env_long("LLAMA_UMA_STREAM_SMAX", n_expert);
    const long initial = have_initial ? llama_uma_stream_env_long("LLAMA_UMA_STREAM_S", n_expert)
                                      : (have_ceiling ? ceiling_env : (long) n_expert);
    const long ceiling = have_ceiling ? ceiling_env : initial;
    if (initial < (long) n_expert_used || initial > ceiling || ceiling > (long) n_expert) {
        throw std::runtime_error(format("invalid UMA stream S range: require %u <= initial %ld <= ceiling %ld <= %u",
                                        n_expert_used, initial, ceiling, n_expert));
    }
    return { (uint32_t) initial, (uint32_t) ceiling };
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model arch does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    cparams.embeddings_layer_inp.resize(hparams.n_layer(), false);
    embd_layer_inp.resize(hparams.n_layer());

    cparams.ctx_type     = params.ctx_type;
    cparams.pooling_type = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = true;

    cparams.fused_lid    = true;
    cparams.auto_flid    = true;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    // uma-moe give-back (B): the ordinary synchronous admit path still caps prefill so one ubatch
    // cannot select more distinct experts than the resident floor. DEVICE_SLOTS use the static
    // expert->slot table instead; their CUDA MUL_MAT_ID nodes are marked for duplicate-safe internal
    // MMVQ tiling, so both compressed and all-resident pools retain the requested full prefill ubatch.
    // S_floor = SMIN (M6 give-back floor) else initial S. Decode is single-token and unchanged.
    if (const char * env_k = getenv("LLAMA_UMA_STREAM_K");
            env_k != nullptr && env_k[0] != '\0' && !llama_uma_stream_static_full_enabled()) {
        const uint32_t nu = model.hparams.n_expert_used ? model.hparams.n_expert_used : 1;
        const llama_uma_stream_s_config s_cfg = llama_uma_stream_parse_s(nu, model.hparams.n_expert);
        long s_floor = llama_uma_stream_env_long("LLAMA_UMA_STREAM_SMIN", s_cfg.initial);
        if (s_floor < (long) nu || s_floor > (long) s_cfg.initial) {
            throw std::runtime_error(format("invalid LLAMA_UMA_STREAM_SMIN %ld (want %u..initial S=%u)",
                                            s_floor, nu, s_cfg.initial));
        }
        const bool device_slots = llama_uma_stream_device_slots_enabled();
        if (s_floor > 0 && !device_slots) {
            const uint32_t ub_cap = std::max<uint32_t>(1, (uint32_t) (s_floor / (long) nu));
            if (cparams.n_ubatch > ub_cap) {
                fprintf(stderr, "uma: give-back capping n_ubatch %u -> %u (S_floor=%ld / n_expert_used=%u) "
                        "so a prefill ubatch never overflows the resident slots (mmq-scatter overflow guard)\n",
                        cparams.n_ubatch, ub_cap, s_floor, nu);
                cparams.n_ubatch = ub_cap;
            }
        }
    }

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    {
        const char * LLAMA_UMA_POLICY = getenv("LLAMA_UMA_POLICY");
        if (LLAMA_UMA_POLICY && LLAMA_UMA_POLICY[0] != '\0') {
            llama_uma_policy uma_policy   = LLAMA_UMA_POLICY_NONE;
            uint32_t         n_cpu_layers = 0;
            if (!llama_uma_router::parse_policy(LLAMA_UMA_POLICY, uma_policy, n_cpu_layers)) {
                // abort, not warn: llama-bench swallows llama logs by default,
                // and a typoed policy silently measuring stock is the one
                // failure a record run cannot be allowed to hide
                throw std::runtime_error(format("unrecognized LLAMA_UMA_POLICY value '%s' (known: gpu-only, cpu-static:N)", LLAMA_UMA_POLICY));
            }
            const auto & hparams = model.hparams;
            if (uma_policy == LLAMA_UMA_POLICY_CPU_STATIC && n_cpu_layers > hparams.n_layer()) {
                throw std::runtime_error(format("LLAMA_UMA_POLICY cpu-static:%u exceeds n_layer=%u", n_cpu_layers, hparams.n_layer()));
            }
            llama_uma_layout uma_auto_layout = LLAMA_UMA_LAYOUT_DEFAULT;
            if (uma_policy == LLAMA_UMA_POLICY_AUTO && !hparams.vocab_only) {
                // re-plan deterministically (same profile, same probes as the
                // load-time injection) and check the profile identity against
                // the loaded hparams - the file-size check ran at injection
                llama_uma_plan    plan;
                llama_uma_profile prof;
                std::string       err;
                if (!llama_uma_auto_plan(nullptr, plan, err, &prof)) {
                    throw std::runtime_error(format("uma auto plan failed: %s", err.c_str()));
                }
                const char * force = getenv("LLAMA_UMA_PROFILE_FORCE");
                const bool forced = force != nullptr && strcmp(force, "1") == 0;
                if (!forced && (prof.n_layer != hparams.n_layer() || prof.n_expert != hparams.n_expert || prof.n_expert_used != hparams.n_expert_used)) {
                    throw std::runtime_error(format("uma profile identity mismatch: profile %u/%u/%u vs model %u/%u/%u (layers/experts/used)",
                            prof.n_layer, prof.n_expert, prof.n_expert_used, hparams.n_layer(), hparams.n_expert, hparams.n_expert_used));
                }
                n_cpu_layers    = plan.k;
                uma_auto_layout = plan.layout;
            }
            if (hparams.n_expert > 0) {
                uma_router = std::make_unique<llama_uma_router>(uma_policy, n_cpu_layers, hparams.n_layer(), hparams.n_expert, hparams.n_expert_used);
                uma_router->layout = uma_auto_layout;
            } else {
                fprintf(stderr, "uma: LLAMA_UMA_POLICY set but model has no experts, router inactive\n");
            }
        }
        const char * LLAMA_UMA_OBSERVE = getenv("LLAMA_UMA_OBSERVE");
        if (LLAMA_UMA_OBSERVE && LLAMA_UMA_OBSERVE[0] != '\0') {
            if (strcmp(LLAMA_UMA_OBSERVE, "experts") != 0) {
                throw std::runtime_error(format("unrecognized LLAMA_UMA_OBSERVE value '%s' (known: experts)", LLAMA_UMA_OBSERVE));
            }
            if (!uma_router) {
                throw std::runtime_error("LLAMA_UMA_OBSERVE requires an active LLAMA_UMA_POLICY router");
            }
            uma_router->observe_experts = true;
            fprintf(stderr, "uma: observe experts channel on\n");
        }
        const char * LLAMA_UMA_LAYOUT = getenv("LLAMA_UMA_LAYOUT");
        if (LLAMA_UMA_LAYOUT && LLAMA_UMA_LAYOUT[0] != '\0') {
            llama_uma_layout uma_layout = LLAMA_UMA_LAYOUT_DEFAULT;
            if (!llama_uma_router::parse_layout(LLAMA_UMA_LAYOUT, uma_layout)) {
                throw std::runtime_error(format("unrecognized LLAMA_UMA_LAYOUT value '%s' (known: std, repack)", LLAMA_UMA_LAYOUT));
            }
            if (!uma_router || uma_router->policy != LLAMA_UMA_POLICY_CPU_STATIC) {
                throw std::runtime_error("LLAMA_UMA_LAYOUT requires LLAMA_UMA_POLICY=cpu-static:N");
            }
            uma_router->layout = uma_layout;
        }
        // M5 residency give-back (uma-moe fork): keep the K hottest experts per
        // layer resident and MADV_DONTNEED-evict the cold experts' slabs so they
        // re-fault from the file-backed GGUF mmap. Env-gated so a default run is
        // untouched. Drives off the expert-hotness histogram, so it force-enables
        // the observe channel. The core new mechanism for co-location give-back.
        const char * LLAMA_UMA_GIVEBACK_K = getenv("LLAMA_UMA_GIVEBACK_K");
        if (LLAMA_UMA_GIVEBACK_K && LLAMA_UMA_GIVEBACK_K[0] != '\0') {
            if (!uma_router) {
                throw std::runtime_error("LLAMA_UMA_GIVEBACK_K requires an active LLAMA_UMA_POLICY router");
            }
            // FREEZE-SAFETY (non-negotiable, D-M5.8): MADV_DONTNEED on a page a
            // live Metal residency set holds gets it re-requested on the ~5ms
            // heartbeat and wedges the GPU pipeline OS-wide (the E5 freeze
            // class). Give-back is only safe with residency sets OFF, so the
            // weights stay reclaimable file-backed page cache. Refuse otherwise.
            if (getenv("GGML_METAL_NO_RESIDENCY") == nullptr) {
                throw std::runtime_error("LLAMA_UMA_GIVEBACK_K requires GGML_METAL_NO_RESIDENCY set: MADV_DONTNEED on pages held by a live Metal residency set re-wires them and can wedge the machine");
            }
            char * end = nullptr;
            const long k = strtol(LLAMA_UMA_GIVEBACK_K, &end, 10);
            if (end == LLAMA_UMA_GIVEBACK_K || *end != '\0' || k < 0 || k > (long) model.hparams.n_expert) {
                throw std::runtime_error(format("invalid LLAMA_UMA_GIVEBACK_K '%s' (want 0..n_expert=%u)", LLAMA_UMA_GIVEBACK_K, model.hparams.n_expert));
            }
            uma_giveback_k = (int32_t) k;
            // give-back ranks by expert_freq; turn the observe channel on so the
            // histogram populates and the topk-cb caches fire (same as OBSERVE).
            uma_router->observe_experts = true;
            const char * LLAMA_UMA_GIVEBACK_PERIOD = getenv("LLAMA_UMA_GIVEBACK_PERIOD");
            if (LLAMA_UMA_GIVEBACK_PERIOD && LLAMA_UMA_GIVEBACK_PERIOD[0] != '\0') {
                char * pend = nullptr;
                const long p = strtol(LLAMA_UMA_GIVEBACK_PERIOD, &pend, 10);
                if (pend == LLAMA_UMA_GIVEBACK_PERIOD || *pend != '\0' || p <= 0) {
                    throw std::runtime_error(format("invalid LLAMA_UMA_GIVEBACK_PERIOD '%s' (want a positive integer)", LLAMA_UMA_GIVEBACK_PERIOD));
                }
                uma_giveback_period = (int32_t) p;
            }
            fprintf(stderr, "uma: residency give-back ON, keep top-%d experts/layer, sweep every %d decode tokens (GGML_METAL_NO_RESIDENCY set)\n",
                    uma_giveback_k, uma_giveback_period);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq      = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max = %u\n",   __func__, cparams.n_outputs_max);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    // M5 finish: supply-curve telemetry - steady-state phys_footprint (the load-time
    // free-excluded value reflects lazy fill) + the realized slot miss rate.
    if (uma_stream) {
        const double miss_pct = uma_stream->n_read > 0 ? 100.0 * (double) uma_stream->n_miss / (double) uma_stream->n_read : 0.0;
        fprintf(stderr, "uma: stream teardown: phys_footprint %zu MiB (steady-state), misses %llu / %llu reads (%.2f%%), S=%u/%u, ceiling=%u\n",
                llama_uma_phys_footprint_mib(),
                (unsigned long long) uma_stream->n_miss, (unsigned long long) uma_stream->n_read,
                miss_pct, uma_stream->n_slots_active, uma_stream->n_expert, uma_stream->n_slots);
        if (uma_stream->device_slots) {
            fprintf(stderr, "uma: device-slot miss H2D: %llu experts, %.1f MiB uploaded\n",
                    (unsigned long long) uma_stream->n_h2d_miss,
                    uma_stream->n_h2d_bytes / (1024.0 * 1024.0));
            fprintf(stderr,
                    "uma: device-slot prefill service: tiles %llu, cold misses %llu, H2D %llu experts, %.1f MiB uploaded, failures %llu, substitutions %llu, max distinct %u\n",
                    (unsigned long long) uma_stream->n_prefill_tiles,
                    (unsigned long long) uma_stream->n_prefill_cold_miss,
                    (unsigned long long) uma_stream->n_prefill_h2d_miss,
                    uma_stream->n_prefill_h2d_bytes / (1024.0 * 1024.0),
                    (unsigned long long) uma_stream->n_prefill_service_fail,
                    (unsigned long long) uma_stream->n_prefill_substitute,
                    uma_stream->prefill_max_distinct);
            fprintf(stderr,
                    "uma: device-slot decode service: calls %llu, cold misses %llu, H2D %llu experts, %.1f MiB uploaded, failures %llu, substitutions %llu\n",
                    (unsigned long long) uma_stream->n_decode_service,
                    (unsigned long long) uma_stream->n_decode_cold_miss,
                    (unsigned long long) uma_stream->n_decode_h2d_miss,
                    uma_stream->n_decode_h2d_bytes / (1024.0 * 1024.0),
                    (unsigned long long) uma_stream->n_decode_service_fail,
                    (unsigned long long) uma_stream->n_decode_substitute);
            fprintf(stderr,
                    "uma: device-slot fixed-S decode maintenance: calls %llu, late misses %llu, H2D %llu experts, %.1f MiB uploaded, failures %llu, exact-after-resize %u\n",
                    (unsigned long long) uma_stream->n_decode_late_calls,
                    (unsigned long long) uma_stream->n_decode_late_cold_miss,
                    (unsigned long long) uma_stream->n_decode_late_h2d_miss,
                    uma_stream->n_decode_late_h2d_bytes / (1024.0 * 1024.0),
                    (unsigned long long) uma_stream->n_decode_late_fail,
                    (unsigned) uma_stream->decode_exact_after_resize);
            fprintf(stderr, "uma: stream substitution overflow: %llu\n",
                    (unsigned long long) uma_stream->n_overflow);
        }
        if (uma_stream->n_resizes > 0 || uma_stream->n_distress > 0) {
            fprintf(stderr, "uma: M6 controller teardown: %llu resizes, %llu distress clamps, S reached [%u,%u] of max %u\n",
                    (unsigned long long) uma_stream->n_resizes, (unsigned long long) uma_stream->n_distress,
                    uma_stream->s_min_active, uma_stream->s_max_active, uma_stream->n_slots);
        }
    } else if (getenv("LLAMA_UMA_FOOTPRINT")) {
        // stock baseline footprint (no streaming), for the supply-curve comparison
        fprintf(stderr, "uma: teardown: phys_footprint %zu MiB (steady-state, stock)\n", llama_uma_phys_footprint_mib());
    }
    if (uma_router && uma_router->n_decide > 0) {
        // stderr on purpose, see the activation line in llama-uma.cpp
        fprintf(stderr, "uma: %" PRId64 " decisions, %" PRId64 " replans, %.3f us avg, %d graphs reused, %d splits last graph\n",
                uma_router->n_decide, uma_router->n_replan, (double) uma_router->t_decide_us/uma_router->n_decide, n_reused,
                sched ? ggml_backend_sched_get_n_splits(sched.get()) : -1);
        fprintf(stderr, "uma: observe pp %" PRId64 " tokens / %" PRId64 " passes / %.1f ms, tg %" PRId64 " tokens / %.3f ms avg, %" PRId64 " rebuilds / %.3f ms avg\n",
                uma_router->n_pp_tokens, uma_router->n_pp_passes, 1e-3 * uma_router->t_pp_us,
                uma_router->n_tg_tokens, uma_router->n_tg_tokens > 0 ? 1e-3 * uma_router->t_tg_us/uma_router->n_tg_tokens : 0.0,
                uma_router->n_rebuild, uma_router->n_rebuild > 0 ? 1e-3 * uma_router->t_rebuild_us/uma_router->n_rebuild : 0.0);
        if (uma_router->observe_experts && uma_router->reuse_den > 0) {
            fprintf(stderr, "uma: observe experts: reuse %.1f%% over %" PRId64 " decode tokens\n",
                    100.0 * (double) uma_router->reuse_num/uma_router->reuse_den, uma_router->n_expert_obs);
        }
        // M5 coverage(S): dump the per-(layer,expert) hotness histogram so the
        // supply-curve analyzer (scripts/coverage_curve.py) can place K_hot. Env-gated
        // (LLAMA_UMA_DUMP_FREQ=path); requires the observe-experts channel populated.
        const char * dump_path = getenv("LLAMA_UMA_DUMP_FREQ");
        if (uma_router->observe_experts && dump_path && dump_path[0] != '\0') {
            FILE * df = fopen(dump_path, "w");
            if (df) {
                const uint32_t nl = uma_router->n_layer, ne = uma_router->n_expert;
                fprintf(df, "# n_layer=%u n_expert=%u n_expert_used=%u decode_tokens=%lld\n",
                        nl, ne, uma_router->n_expert_used, (long long) uma_router->n_expert_obs);
                fprintf(df, "layer,expert,count\n");
                for (uint32_t il = 0; il < nl; il++) {
                    const uint32_t * freq = uma_router->expert_freq.data() + (size_t) il * ne;
                    for (uint32_t e = 0; e < ne; e++) {
                        fprintf(df, "%u,%u,%u\n", il, e, freq[e]);
                    }
                }
                fclose(df);
                fprintf(stderr, "uma: dumped expert_freq (%u layers x %u experts, %lld decode tokens) to %s\n",
                        nl, ne, (long long) uma_router->n_expert_obs, dump_path);
            } else {
                fprintf(stderr, "uma: WARNING could not open LLAMA_UMA_DUMP_FREQ path '%s'\n", dump_path);
            }
        }
    }

    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    gf_res_reserve.reset(new llm_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    uma_allow_weights_bufts();

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                uma_allow_weights_bufts();
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
        // observe() reads the same clock as the perf stats but is gated on
        // the router, not no_perf - llama-bench leaves no_perf on and the
        // policy needs feedback during record runs. The t_compute guard
        // skips mid-decode reserve/realloc syncs that would book a bogus
        // elapsed (same hole the upstream perf stats have)
        if (uma_router && t_compute_start_us != 0) {
            uma_router->observe_pass(1, ggml_time_us() - t_compute_start_us);
            if (uma_router->observe_experts) {
                uma_router->observe_experts_read();
                // M5 residency give-back: rate-limited cold-expert eviction.
                // Fires only every uma_giveback_period decode tokens so the
                // hotness histogram stabilizes and a shed amortizes (M4 trigger
                // discipline: never per token). No-op unless LLAMA_UMA_GIVEBACK_K
                // is set (uma_giveback_k >= 0). Post-sync, decode-only path.
                if (uma_giveback_k >= 0 && ++uma_giveback_tick >= uma_giveback_period) {
                    uma_giveback_tick = 0;
                    uma_apply_residency();
                }
            }
        }
        // decouple maintenance (Part 2): post-sync, per selected expert - count residency and
        // (Step 3, LLAMA_UMA_STREAM_ADAPT) online-admit misses so the resident set adapts to
        // the live workload (no matched-freq oracle). Runs in the GPU-idle window (after
        // synchronize), so slot + table writes are torn-write-safe with NO background thread;
        // the next graph_compute sees the updated StorageModeShared state. One D2H per layer;
        // no per-layer graph op, so the decouple stays sync-free.
        // Multi-token prefill is always serviced exactly in its MUL_MAT_ID ops.
        // Fixed-S decode retains the Task-21 post-compute maintenance fast path;
        // after a live resize, decode_exact_after_resize switches the rebuilt
        // graph to exact pre-consume service and suppresses this late path.
        if (uma_stream && uma_stream->decouple && t_compute_start_us != 0 &&
            !(uma_stream->device_slots && uma_stream->n_slots_active < uma_stream->n_expert &&
              uma_stream->decode_exact_after_resize)) {
            const uint32_t n_used = model.hparams.n_expert_used;
            const uint32_t Sn     = uma_stream->n_slots_active; // M6: online admits stay in the active window
            const bool     adapt  = uma_stream->adapt;
            if (Sn == uma_stream->n_expert) {
                // Every expert is resident: misses and admissions are impossible, and
                // recency will be rebuilt if a later resize compresses the window. Avoid
                // one synchronous tiny D2H tensor_get per streaming layer per token while
                // retaining exact read/miss telemetry.
                for (uint32_t il = 0; il < uma_stream->topk.size(); il++) {
                    if (uma_stream->topk[il] != nullptr && uma_stream->streams_layer((int) il)) {
                        uma_stream->n_read += n_used;
                    }
                }
            } else {
              std::vector<int32_t> ids(n_used);
              for (uint32_t il = 0; il < uma_stream->topk.size(); il++) {
                ggml_tensor * t = uma_stream->topk[il];
                if (t == nullptr || !uma_stream->streams_layer((int) il)) {
                    continue;
                }
                if (t->type != GGML_TYPE_I32 || t->ne[0] < (int64_t) n_used) {
                    continue;
                }
                if (uma_stream->device_slots) {
                    uma_stream->n_decode_late_calls++;
                }
                ggml_backend_tensor_get(t, ids.data(), 0, n_used * sizeof(int32_t));
                llama_uma_stream_layer_lru & L = uma_stream->lru[il];
                int32_t * tbl = uma_stream->expert_table((int) il) ?
                    (int32_t *) uma_stream->expert_table((int) il)->data : nullptr;
                L.pass++;
                for (uint32_t j = 0; j < n_used; j++) {
                    const int32_t e = ids[j];
                    if (e < 0 || (uint32_t) e >= uma_stream->n_expert) {
                        continue;
                    }
                    uma_stream->n_read++;
                    int32_t slot = L.slot_of_expert[e];
                    if (slot >= 0 && (uint32_t) slot < Sn) {
                        // hit: refresh recency (adapt LRU keeps the live working set resident)
                        L.pinned[slot]    = L.pass;
                        L.last_used[slot] = L.tick++;
                        continue;
                    }
                    if (slot >= 0) {
                        // A resize must clear inactive mappings. Fail closed if stale metadata
                        // survives: unpublish it before treating this access as a miss.
                        L.slot_of_expert[e] = -1;
                        if (tbl) { tbl[e] = 0; }
                    }
                    uma_stream->n_miss++; // selected expert not resident = decouple miss
                    if (uma_stream->device_slots) {
                        uma_stream->n_decode_late_cold_miss++;
                    }
                    if (!adapt) {
                        continue; // static mode (Step 2): count only
                    }
                    // online admit (pure LRU): evict the LRU-cold slot not pinned this token
                    uint64_t best_lu = UINT64_MAX;
                    slot = -1;
                    for (uint32_t s = 0; s < Sn; s++) {
                        if (L.expert_in_slot[s] < 0) { slot = (int32_t) s; break; }
                        if (L.pinned[s] != L.pass && L.last_used[s] < best_lu) {
                            best_lu = L.last_used[s];
                            slot    = (int32_t) s;
                        }
                    }
                    if (slot < 0) {
                        if (uma_stream->device_slots) {
                            uma_stream->n_decode_late_fail++;
                        }
                        continue; // all S slots needed this token (>= S misses at once); next token
                    }
                    // Fill every streaming kind, THEN publish the table entry (so the next
                    // pass never routes to a half-filled slot). CUDA device slots pread into
                    // reusable pinned-host staging slabs, enqueue all H2D copies, and complete
                    // that backend stream before publication. The current graph is already
                    // synchronized, so overwriting the victim cannot race a kernel read.
                    bool ok = true;
                    if (uma_stream->device_slots) {
                        for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
                            if (!uma_stream->streams((int) il, kind)) { continue; }
                            const size_t slab = model.uma_stream_slab_bytes((int) il, kind);
                            if ((size_t) kind >= uma_stream->device_stage_host.size() ||
                                uma_stream->device_stage_host[kind] == nullptr ||
                                uma_stream->device_stage_bytes[kind] < slab ||
                                !model.uma_stream_pread_expert((int) il, kind, e,
                                                               uma_stream->device_stage_host[kind])) {
                                ok = false;
                                break;
                            }
                        }
                        if (ok) {
                            uint64_t uploaded = 0;
                            for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
                                if (!uma_stream->streams((int) il, kind)) { continue; }
                                const size_t slab = model.uma_stream_slab_bytes((int) il, kind);
                                ggml_backend_tensor_set_async(
                                    uma_stream_cuda_backend, uma_stream->slot((int) il, kind),
                                    uma_stream->device_stage_host[kind], (size_t) slot * slab, slab);
                                uploaded += slab;
                            }
                            ggml_backend_synchronize(uma_stream_cuda_backend);
                            uma_stream->n_h2d_miss++;
                            uma_stream->n_h2d_bytes += uploaded;
                            uma_stream->n_decode_late_h2d_miss++;
                            uma_stream->n_decode_late_h2d_bytes += uploaded;
                        }
                    } else {
                        for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
                            if (!uma_stream->streams((int) il, kind)) { continue; }
                            const size_t slab = model.uma_stream_slab_bytes((int) il, kind);
                            char * base = (char *) uma_stream->slot((int) il, kind)->data;
                            if (!model.uma_stream_pread_expert((int) il, kind, e, base + (size_t) slot * slab)) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if (!ok) {
                        if (uma_stream->device_slots) {
                            uma_stream->n_decode_late_fail++;
                        }
                        // partial pread (some kinds filled, one failed): the slot bytes are now
                        // mixed old/new = corrupt, so the OLD expert must not stay routable.
                        // Invalidate the slot (empty); a later admit refills it cleanly. e stays
                        // a miss. Adapt runs online, so a transient I/O error must not abort.
                        const int32_t bad = L.expert_in_slot[slot];
                        if (bad >= 0) {
                            L.slot_of_expert[bad] = -1;
                            if (tbl) { tbl[bad] = 0; }
                        }
                        L.expert_in_slot[slot] = -1;
                        continue;
                    }
                    const int32_t old = L.expert_in_slot[slot];
                    if (old >= 0) {
                        L.slot_of_expert[old] = -1;
                        if (tbl) { tbl[old] = 0; } // evicted -> sentinel slot 0
                    }
                    L.expert_in_slot[slot] = e;
                    L.slot_of_expert[e]    = slot;
                    L.pinned[slot]         = L.pass;
                    L.last_used[slot]      = L.tick++;
                    if (tbl) { tbl[e] = slot; } // publish AFTER the pread
                }
              }
            }
        }

        // M6 give-back controller: rate-limited runtime S-resize in the same
        // post-sync GPU-idle window. Reads the M4 budget signal (CTRL) or a
        // commanded schedule (SCHED); decode-only, never per token.
        if (uma_stream && uma_stream->decouple && uma_resize_smin >= 0 && t_compute_start_us != 0) {
            uma_stream_controller_tick();
        }
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
        if (uma_router && t_compute_start_us != 0) {
            uma_router->observe_pass(n_queued_tokens, ggml_time_us() - t_compute_start_us);
        }
    }

    // M7.0 continuous-batching: the give-back controller tick above fires only for single-token
    // decode (the n_queued_tokens == 1 branch). Under CONTINUOUS BATCHING a decode step is N
    // sequences x 1 token -> n_queued_tokens == N > 1 with every token an output
    // (n_outputs == n_queued_tokens), whereas prefill outputs fewer. Fire the tick for a BATCHED
    // decode step too, so runtime resize + the M7.1 control channel work under continuous batching
    // (the Spark throughput-bound regime; llama-server -np N). GUARDED to PURE-decode batches so a
    // commanded shed never lands mid-prefill (which would violate "a prefill ubatch needs
    // S >= its distinct experts"). The tick reseeds the slot table on resize, so no per-decode
    // decouple maintenance is required here.
    if (uma_stream && uma_stream->decouple && uma_resize_smin >= 0 && t_compute_start_us != 0
            && n_queued_tokens > 1 && (int64_t) n_outputs == n_queued_tokens) {
        uma_stream_controller_tick();
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid < model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // uma-moe: per-token placement decision; only a placement CHANGE
    // invalidates the cached schedule (gpu-only never changes)
    bool uma_replan = false;
    if (uma_router) {
        uma_replan = uma_router->decide(ubatch.n_tokens);
    }
    // M6: a runtime slot resize reallocated the slot tensors; the cached graph
    // references the freed ones (can_reuse only inspects ubatch topology, not
    // weight tensors), so force one rebuild to re-wrap the new slot buffers.
    if (uma_stream_force_rebuild) {
        uma_replan = true;
        uma_stream_force_rebuild = false;
    }

    if (!graph_reuse_disable && !uma_replan && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        const int64_t t_rebuild_start_us = uma_router ? ggml_time_us() : 0;

        // stale-pointer guard: reserve builds also fire the cb, so cached
        // topk entries must only ever point into the graph built HERE
        if (uma_router && uma_router->observe_experts) {
            uma_router->topk_tensors.assign(uma_router->topk_tensors.size(), nullptr);
        }
        // same guard for the decouple miss-detection topk cache (stream path)
        if (uma_stream && uma_stream->decouple) {
            uma_stream->topk.assign(uma_stream->topk.size(), nullptr);
        }

        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        if (uma_router) {
            uma_router->observe_rebuild(ggml_time_us() - t_rebuild_start_us);
        }
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_embd = hparams.n_embd_inp_enc();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & sampled,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<float> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LLAMA_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                            __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
            }
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers && (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched.get());

            copy_tensor_async_floats    (res->t_sampled_logits, sampling.logits,     stride, sampling.logits_count,     seq_to_output_row, sched.get());
            copy_tensor_async_floats    (res->t_sampled_probs,  sampling.probs,      stride, sampling.probs_count,      seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates,     sampling.candidates, stride, sampling.candidates_count, seq_to_output_row, sched.get());
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

void llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
    }
}

void llama_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd.data[i0*n_embd + k], embd.data[i1*n_embd + k]);
            }
        }

        if (embd_nextn.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd_nextn.data[i0*n_embd + k], embd_nextn.data[i1*n_embd + k]);
            }
        }

        if (embd_layer_inp.size() > 0) {
            for (int lid = 0; lid < (int) embd_layer_inp.size(); ++lid) {
                if (embd_layer_inp[lid].size > 0) {
                    for (uint64_t k = 0; k < n_embd; ++k) {
                        std::swap(embd_layer_inp[lid].data[i0*n_embd + k], embd_layer_inp[lid].data[i1*n_embd + k]);
                    }
                }
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_M3) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    uint32_t res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i] = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &seq_ids[i];
        ubatch.output[i] = true;
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
        /*.uma_stream  =*/ uma_stream.get(),
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

// allowlist, not denylist: only buffer types positively known to hold
// host-dereferenceable weights may be registered - anything else (Metal
// private, CUDA device VRAM, future backends) must abort rather than fail
// open. Metal shared = "MTL<i>", Metal mapped = "MTL<i>_Mapped" (mmap).
// The CUDA route does not come through here: its weights sit in the device's
// pinned host buft (is_host=true), placed at load by the injected overrides.
static bool uma_buft_host_visible(const char * name) {
    if (strncmp(name, "MTL", 3) != 0) {
        return false;
    }
    const char * p = name + 3;
    if (*p < '0' || *p > '9') {
        return false;
    }
    while (*p >= '0' && *p <= '9') {
        p++;
    }
    return *p == '\0' || strcmp(p, "_Mapped") == 0;
}

void llama_context::uma_apply_residency() {
    // M5 residency give-back: evict each layer's COLD experts (per-expert slab,
    // NOT per-layer - every layer runs every token, but within a layer the hot
    // set is small: 40-60% expert carryover). Keep the K hottest per layer
    // resident by expert_freq; MADV_DONTNEED the rest so they re-fault from the
    // file-backed GGUF mmap on the next route-miss. Env-gated (uma_giveback_k
    // >= 0) and only reached when GGML_METAL_NO_RESIDENCY is set (ctor gate),
    // so it never advises a page a live Metal residency set re-wires.
    if (!uma_router || uma_giveback_k < 0) {
        return;
    }
    const uint32_t n_layer  = uma_router->n_layer;
    const uint32_t n_expert = uma_router->n_expert;
    if (n_expert == 0) {
        return;
    }
    const uint32_t keep = std::min<uint32_t>((uint32_t) uma_giveback_k, n_expert);
    if (keep >= n_expert) {
        return;  // keeping every expert => nothing cold to shed
    }

    size_t   evicted_bytes = 0;
    uint32_t evicted_slabs = 0;
    uint32_t skipped_nonhost = 0;
    uint32_t skipped_anon    = 0;
    std::vector<uint32_t> order(n_expert);  // expert-index scratch, per layer

    for (uint32_t il = 0; il < n_layer; il++) {
        const uint32_t * freq = uma_router->expert_freq.data() + (size_t) il * n_expert;
        // partition experts so [0,keep) are the `keep` hottest (unordered) and
        // [keep,n_expert) are the cold experts to evict. nth_element is O(n).
        for (uint32_t e = 0; e < n_expert; e++) {
            order[e] = e;
        }
        std::nth_element(order.begin(), order.begin() + keep, order.end(),
                [freq](uint32_t a, uint32_t b) { return freq[a] > freq[b]; });

        const auto & layer = model.layers[il];
        for (ggml_tensor * t : { layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps, layer.ffn_gate_up_exps }) {
            if (t == nullptr || t->buffer == nullptr || t->data == nullptr) {
                continue;
            }
            // FREEZE-SAFETY: only advise host (CPU / file-backed mmap) buffers.
            // A Metal device buffer's pages may be held by a residency set whose
            // 5ms heartbeat re-wires anything we drop -> the E5 OS-wide wedge.
            // Skip anything non-host: give-back is a page-cache/mmap mechanism.
            if (!ggml_backend_buffer_is_host(t->buffer)) {
                skipped_nonhost++;
                continue;
            }
            // experts live in one 3D tensor [n_embd, n_ff_exp, n_expert];
            // expert e's contiguous slab is t->data + e*t->nb[2], size t->nb[2].
            if (t->ne[2] != (int64_t) n_expert) {
                continue;
            }
            // CORRECTNESS GATE: only evict pages that are FILE-BACKED by the
            // GGUF mmap. Dropping those re-faults identical immutable bytes from
            // disk; dropping ANONYMOUS host pages (e.g. -lm none copies) would
            // zero-fill on refault and corrupt the weights. A whole tensor is
            // one contiguous span, so testing it once covers all its slabs.
            if (!model.uma_addr_in_mmap(t->data, ggml_nbytes(t))) {
                skipped_anon++;
                continue;
            }
            for (uint32_t j = keep; j < n_expert; j++) {
                const uint32_t e = order[j];
                void * slab = (char *) t->data + (size_t) e * t->nb[2];
                const size_t adv = llama_uma_madvise_dontneed(slab, (size_t) t->nb[2]);
                if (adv > 0) {
                    evicted_bytes += adv;
                    evicted_slabs++;
                }
            }
        }
    }

    // measurement evidence, deliberately via fprintf not LLAMA_LOG (llama-bench
    // installs a null log callback): a give-back run must prove it evicted.
    if (evicted_slabs > 0 || skipped_nonhost > 0 || skipped_anon > 0) {
        fprintf(stderr, "uma: give-back sweep @ %lld tok: kept top-%u/%u experts/layer, evicted %u cold slabs = %.1f MiB%s%s\n",
                (long long) uma_router->n_expert_obs, keep, n_expert, evicted_slabs, evicted_bytes / (1024.0 * 1024.0),
                skipped_nonhost > 0 ? " (some experts non-host: not sheddable)" : "",
                skipped_anon    > 0 ? " (some host experts NOT mmap-backed: skipped to avoid zero-fill)" : "");
    }
}

// uma-moe fork M5 S1.1.2: admit op (forced onto CPU by graph_get_cb). Reads
// selected_experts (dst->src[0]), assigns each selected expert a slot via the
// per-layer LRU (admitting misses into an empty or LRU-cold unpinned slot,
// recording them in lru[il].newly_admitted), and writes slot_ids into dst. In-
// batch experts are pinned so a batch never evicts an expert it still needs.
void llama_uma_stream_admit(ggml_tensor * dst, int ith, int /*nth*/, void * userdata) {
    if (ith != 0) {
        return; // n_tasks == 1
    }
    const auto * ud = (const llama_uma_stream_admit_ud *) userdata;
    llama_uma_stream_state * S = ud->state;
    if (S->device_slots) {
        GGML_ABORT("uma stream admit: device slots must use the decoupled table path");
    }
    llama_uma_stream_layer_lru & L = S->lru[ud->il];
    const ggml_tensor * sel = dst->src[0]; // selected_experts, I32 [n_expert_used, n_tokens]
    if (sel == nullptr || sel->type != GGML_TYPE_I32 || sel->data == nullptr) {
        GGML_ABORT("uma stream admit: (il=%d) selected_experts not CPU-resident I32", ud->il);
    }
    if (dst->type != GGML_TYPE_I32 || dst->data == nullptr) {
        GGML_ABORT("uma stream admit: (il=%d) slot_ids not I32/allocated", ud->il);
    }
    // selected_experts is a STRIDED VIEW of the argsort (ggml_argsort_top_k:
    // nb[1] = n_expert*4, the first n_expert_used columns), so it MUST be read
    // via its strides - a flat read pulls in unselected experts for n_tokens > 1
    // (prefill). slot_ids (dst) is a fresh contiguous [n_used, n_tok] tensor.
    const int64_t  n_used = sel->ne[0]; // n_expert_used
    const int64_t  n_tok  = sel->ne[1]; // n_tokens
    int32_t      * out    = (int32_t *) dst->data;
    const uint32_t Sn     = S->n_slots_active;
    L.pass++;
    L.n_newly = 0;
    // GIVE-BACK ROUTER INVARIANT (ZEDA, arXiv:2605.18643, verified on Qwen3-30B-A3B): the give-back
    // is a pure DATA-PLANE indirection (expert id -> resident slot); it NEVER touches the router's
    // gating softmax and NEVER renormalizes it, so the pretrained top-K weight-sum magnitude is
    // preserved. ZEDA shows renormalizing surviving router weights after dropping an expert HURTS
    // accuracy (73.3 -> 71.6) by inflating the MoE residual scale -> leaving routing untouched is
    // the correct choice, and the coverage knee must be MEASURED on this un-renormalized path
    // (scripts/coverage_curve.py). A non-resident expert falls back to sentinel slot 0 (an
    // approximation carried by its ORIGINAL gating weight); this only fires BELOW the coverage knee
    // (above it coverage ~100% -> misses ~0%). If a below-knee graceful-miss (the dropped idea (3))
    // is ever added, it must SKIP (zero the expert's contribution) and MUST NOT renormalize.
    for (int64_t t = 0; t < n_tok; t++) {
        const int32_t * sp = (const int32_t *)((const char *) sel->data + t * sel->nb[1]);
        for (int64_t j = 0; j < n_used; j++) {
            const int32_t e  = sp[j];
            const int64_t oi = t * n_used + j;
            if (e < 0 || (uint32_t) e >= S->n_expert) {
                out[oi] = 0; // padding id: routes to slot 0 but its gating weight is 0
                continue;
            }
            S->n_read++; // valid expert-read (supply-curve miss-rate denominator)
            int32_t slot = L.slot_of_expert[e];
            if (slot < 0 || (uint32_t) slot >= Sn) {
                if (slot >= 0) {
                    L.slot_of_expert[e] = -1; // stale mapping outside the active window
                }
                // miss: admit into an empty slot, else evict the LRU-cold slot
                // that is not pinned by an expert this batch still needs
                uint64_t best_lu = UINT64_MAX;
                // pass 1: prefer an empty slot, else the LRU-cold slot that is neither
                // in-batch-pinned (needed this pass) nor hot-pinned (warm-start set)
                for (uint32_t s = 0; s < Sn; s++) {
                    if (L.expert_in_slot[s] < 0) { slot = (int32_t) s; break; } // empty wins
                    if (!L.pin_protected[s] && L.pinned[s] != L.pass && L.last_used[s] < best_lu) {
                        best_lu = L.last_used[s];
                        slot    = (int32_t) s;
                    }
                }
                // pass 2: no unpinned victim (e.g. a large prefill ubatch that needs more
                // churn than S-H). The hot pin is a SOFT bias, so yield it - still never
                // evicting an in-batch expert. The victim no longer holds a hot expert, so
                // drop its protection.
                if (slot < 0) {
                    best_lu = UINT64_MAX;
                    for (uint32_t s = 0; s < Sn; s++) {
                        if (L.pinned[s] != L.pass && L.last_used[s] < best_lu) {
                            best_lu = L.last_used[s];
                            slot    = (int32_t) s;
                        }
                    }
                    if (slot >= 0) {
                        L.pin_protected[slot] = 0;
                    }
                }
                if (slot < 0) {
                    // OVERFLOW: this batch (a large / prefill ubatch) needs > S distinct experts, and
                    // every slot is already pinned by an expert it still needs this pass. Rather than
                    // ABORT, route the overflow expert to sentinel slot 0 (a graceful MISS / approximation)
                    // so the give-back runs at S < n_expert through prefill. Coherence-bounded: at S >= the
                    // coverage knee the overflow is the cold tail (a small perturbation, small gating
                    // weight); below the knee it degrades but never crashes. Not admitted, not preaded;
                    // counted as a miss + overflow (the arbiter uses n_overflow as a raise-S signal).
                    out[oi] = 0;
                    S->n_miss++;
                    S->n_overflow++;
                    continue;
                }
                const int32_t old = L.expert_in_slot[slot];
                if (old >= 0) {
                    L.slot_of_expert[old] = -1; // evict
                }
                L.expert_in_slot[slot]        = e;
                L.slot_of_expert[e]           = slot;
                L.newly_admitted[L.n_newly++] = e;
            }
            L.pinned[slot]    = L.pass;
            L.last_used[slot] = L.tick++;
            out[oi]           = slot;
        }
    }
    S->n_miss += L.n_newly; // supply-curve telemetry: misses this pass (n_read counted per valid read above)
}

// uma-moe fork M5 S1.1.2: fill op (forced onto CPU). Preads the experts admitted
// THIS pass (lru[il].newly_admitted, set by the admit op) into their slots at
// dst->data + slot*slab. src[1] is slot_ids (a dependency anchor so this runs
// after admit). Output aliases the slot tensor so the matmul weight depends on
// the fill = the CPU-fill -> GPU-read sync. Hits (already resident) are not
// re-pread; their slot bytes are immutable weights written on a previous pass.
void llama_uma_stream_fill(ggml_tensor * dst, int ith, int /*nth*/, void * userdata) {
    if (ith != 0) {
        return; // n_tasks == 1
    }
    const auto * ud = (const llama_uma_stream_fill_ud *) userdata;
    llama_uma_stream_state * S = ud->state;
    if (S->device_slots) {
        GGML_ABORT("uma stream fill: refusing host pread into a CUDA device slot");
    }
    const llama_uma_stream_layer_lru & L = S->lru[ud->il];
    const size_t slab = S->model->uma_stream_slab_bytes(ud->il, ud->kind);
    if (slab == 0 || dst->data == nullptr) {
        GGML_ABORT("uma stream fill: (il=%d kind=%d) slot not streaming or unallocated", ud->il, ud->kind);
    }
    char * base = (char *) dst->data;
    for (uint32_t j = 0; j < L.n_newly; j++) {
        const int32_t e    = L.newly_admitted[j];
        const int32_t slot = L.slot_of_expert[e];
        if (slot < 0 || (uint32_t) slot >= S->n_slots_active) {
            GGML_ABORT("uma stream fill: (il=%d kind=%d) bad slot %d for expert %d", ud->il, ud->kind, slot, e);
        }
        if (!S->model->uma_stream_pread_expert(ud->il, ud->kind, (int) e, base + (size_t) slot * slab)) {
            GGML_ABORT("uma stream fill: pread failed (il=%d kind=%d e=%d)", ud->il, ud->kind, e);
        }
    }
}

int llama_uma_stream_service_prefill_tile(
        void * userdata, const int32_t * expert_ids, int32_t * slot_ids, int64_t n_ids) {
    const auto * ud = (const llama_uma_stream_admit_ud *) userdata;
    llama_uma_stream_state * S = ud == nullptr ? nullptr : ud->state;
    if (S == nullptr || !S->device_slots || S->device_backend == nullptr || expert_ids == nullptr ||
        slot_ids == nullptr || n_ids <= 0 || ud->il < 0 || (size_t) ud->il >= S->lru.size()) {
        if (S != nullptr) {
            if (ud != nullptr && !ud->prefill) { S->n_decode_service_fail++; }
            else                               { S->n_prefill_service_fail++; }
        }
        return -1;
    }

    llama_uma_stream_layer_lru & L = S->lru[ud->il];
    const bool prefill = ud->prefill;
    const uint32_t Sn = S->n_slots_active;
    std::vector<uint8_t> seen(S->n_expert, 0);
    std::vector<int32_t> distinct;
    distinct.reserve((size_t) n_ids);
    for (int64_t i = 0; i < n_ids; i++) {
        const int32_t e = expert_ids[i];
        if (e < 0 || (uint32_t) e >= S->n_expert) {
            continue;
        }
        if (!seen[e]) {
            seen[e] = 1;
            distinct.push_back(e);
        }
    }
    if (distinct.size() > Sn) {
        return 0;
    }

    if (prefill) {
        S->n_prefill_tiles++;
    } else {
        S->n_decode_service++;
    }
    for (int64_t i = 0; i < n_ids; i++) {
        if (expert_ids[i] >= 0 && (uint32_t) expert_ids[i] < S->n_expert) {
            S->n_read++;
        }
    }
    if (prefill) {
        S->prefill_max_distinct = std::max(S->prefill_max_distinct, (uint32_t) distinct.size());
    }
    L.pass++;
    int32_t * tbl = S->expert_table(ud->il) ? (int32_t *) S->expert_table(ud->il)->data : nullptr;

    for (const int32_t e : distinct) {
        int32_t slot = L.slot_of_expert[e];
        if (slot >= 0 && (uint32_t) slot < Sn) {
            L.pinned[slot] = L.pass;
            L.last_used[slot] = L.tick++;
        } else if (slot >= 0) {
            L.slot_of_expert[e] = -1;
            if (tbl) {
                tbl[e] = 0;
            }
        }
    }

    for (const int32_t e : distinct) {
        if (L.slot_of_expert[e] >= 0) {
            continue;
        }

        S->n_miss++;
        if (prefill) { S->n_prefill_cold_miss++; }
        else         { S->n_decode_cold_miss++; }
        uint64_t best_lu = UINT64_MAX;
        int32_t slot = -1;
        for (uint32_t s = 0; s < Sn; s++) {
            if (L.expert_in_slot[s] < 0) {
                slot = (int32_t) s;
                break;
            }
            if (L.pinned[s] != L.pass && L.last_used[s] < best_lu) {
                best_lu = L.last_used[s];
                slot = (int32_t) s;
            }
        }
        if (slot < 0) {
            if (prefill) { S->n_prefill_service_fail++; }
            else         { S->n_decode_service_fail++; }
            return -1;
        }

        for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
            if (!S->streams(ud->il, kind)) {
                continue;
            }
            const size_t slab = S->model->uma_stream_slab_bytes(ud->il, kind);
            if ((size_t) kind >= S->device_stage_host.size() ||
                S->device_stage_host[kind] == nullptr || S->device_stage_bytes[kind] < slab ||
                !S->model->uma_stream_pread_expert(ud->il, kind, e, S->device_stage_host[kind])) {
                if (prefill) { S->n_prefill_service_fail++; }
                else         { S->n_decode_service_fail++; }
                return -1;
            }
        }

        uint64_t uploaded = 0;
        for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
            if (!S->streams(ud->il, kind)) {
                continue;
            }
            const size_t slab = S->model->uma_stream_slab_bytes(ud->il, kind);
            ggml_backend_tensor_set_async(
                S->device_backend, S->slot(ud->il, kind), S->device_stage_host[kind],
                (size_t) slot * slab, slab);
            uploaded += slab;
        }
        ggml_backend_synchronize(S->device_backend);

        const int32_t old = L.expert_in_slot[slot];
        if (old >= 0) {
            L.slot_of_expert[old] = -1;
            if (tbl) {
                tbl[old] = 0;
            }
        }
        L.expert_in_slot[slot] = e;
        L.slot_of_expert[e] = slot;
        L.pinned[slot] = L.pass;
        L.last_used[slot] = L.tick++;
        if (tbl) {
            tbl[e] = slot;
        }
        S->n_h2d_miss++;
        S->n_h2d_bytes += uploaded;
        if (prefill) {
            S->n_prefill_h2d_miss++;
            S->n_prefill_h2d_bytes += uploaded;
        } else {
            S->n_decode_h2d_miss++;
            S->n_decode_h2d_bytes += uploaded;
        }
    }

    for (int64_t i = 0; i < n_ids; i++) {
        const int32_t e = expert_ids[i];
        if (e < 0 || (uint32_t) e >= S->n_expert) {
            slot_ids[i] = 0;
            continue;
        }
        const int32_t slot = L.slot_of_expert[e];
        if (slot < 0 || (uint32_t) slot >= Sn || L.expert_in_slot[slot] != e) {
            if (prefill) {
                S->n_prefill_substitute++;
                S->n_prefill_service_fail++;
            } else {
                S->n_decode_substitute++;
                S->n_decode_service_fail++;
            }
            return -1;
        }
        slot_ids[i] = slot;
    }
    return 1;
}

// uma-moe fork M5 finish: hot-K warm start + LFU pin. Seed each streaming layer's slots
// with its top-S hottest experts (from an LLAMA_UMA_STREAM_HOTFREQ dump = the
// m5-capture-freq CSV) and pin the top-H against LRU eviction, so the resident set starts
// LFU-optimal instead of cold. No env -> cold start (current behavior). Called once after
// the slot pool is built; preads work from the dup'd fd regardless of free_excluded order.
static void uma_stream_warm_start(llama_uma_stream_state * S, const llama_model & model) {
    const char * freq_path = getenv("LLAMA_UMA_STREAM_HOTFREQ");
    if (freq_path == nullptr || freq_path[0] == '\0') {
        return; // cold start
    }
    const uint32_t n_expert      = S->n_expert;
    const uint32_t n_slots       = S->n_slots_active;
    const uint32_t n_expert_used = model.hparams.n_expert_used;
    const uint32_t k             = model.uma_stream_k();

    // pin count H: hottest experts/layer immune to eviction. Default leaves >= n_used
    // unpinned churn slots so a cold miss always has a victim (else the first miss aborts).
    const uint32_t pin_cap = n_slots > n_expert_used ? n_slots - n_expert_used : 0;
    uint32_t H = pin_cap;
    if (const char * env_h = getenv("LLAMA_UMA_STREAM_PIN")) {
        char * end = nullptr;
        const long h = strtol(env_h, &end, 10);
        if (end == env_h || *end != '\0' || h < 0) {
            throw std::runtime_error(format("invalid LLAMA_UMA_STREAM_PIN '%s' (want 0..%u)", env_h, pin_cap));
        }
        H = (uint32_t) h;
        if (H > pin_cap) {
            fprintf(stderr, "uma: STREAM_PIN=%ld clamped to %u (need >= %u unpinned churn slots)\n", h, pin_cap, n_expert_used);
            H = pin_cap;
        }
    }
    S->pin_h = H; // M6: reseed after a resize reuses the hot-pin count (clamped to the new S)

    // parse the freq dump: "# n_layer=.. n_expert=.." header, then "layer,expert,count".
    FILE * f = fopen(freq_path, "r");
    if (f == nullptr) {
        throw std::runtime_error(format("uma stream: cannot open LLAMA_UMA_STREAM_HOTFREQ '%s'", freq_path));
    }
    std::vector<uint64_t> freq((size_t) k * n_expert, 0); // freq[il*n_expert + e]
    char line[256];
    uint32_t hdr_layer = 0, hdr_expert = 0;
    bool have_hdr = false;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            if (sscanf(line, "# n_layer=%u n_expert=%u", &hdr_layer, &hdr_expert) == 2) {
                have_hdr = true;
            }
            continue;
        }
        if (line[0] == 'l') { continue; } // "layer,expert,count" header
        unsigned il = 0, e = 0; unsigned long long c = 0;
        if (sscanf(line, "%u,%u,%llu", &il, &e, &c) != 3) { continue; }
        if (il < k && e < n_expert) {
            freq[(size_t) il * n_expert + e] = c;
        }
    }
    fclose(f);
    if (!have_hdr || hdr_expert != n_expert) {
        throw std::runtime_error(format("uma stream: HOTFREQ header n_expert=%u != model n_expert=%u (wrong dump?)", hdr_expert, n_expert));
    }
    if (hdr_layer < k) {
        fprintf(stderr, "uma: WARNING HOTFREQ n_layer=%u < streaming K=%u; missing layers seed by id\n", hdr_layer, k);
    }

    // per streaming layer: rank experts by count desc (tie-break by id), seed the top-S.
    // A cudaMalloc-backed slot cannot be a pread destination: read into the reusable
    // pinned-host staging slab and upload it with the tensor backend API.
    uint32_t seeded = 0;
    for (uint32_t il = 0; il < k; il++) {
        if (!S->streams_layer((int) il)) {
            continue;
        }
        llama_uma_stream_layer_lru & L = S->lru[il];
        const uint64_t * fq = freq.data() + (size_t) il * n_expert;
        int32_t * tbl = (S->decouple && S->expert_table((int) il)) ? (int32_t *) S->expert_table((int) il)->data : nullptr;
        std::vector<uint32_t> order(n_expert);
        for (uint32_t e = 0; e < n_expert; e++) { order[e] = e; }
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (fq[a] != fq[b]) { return fq[a] > fq[b]; }
            return a < b; // deterministic tie-break
        });
        S->ranked[il].assign(order.begin(), order.end()); // M6: keep the ranking for eager grow
        for (uint32_t r = 0; r < n_slots; r++) {
            const uint32_t e = order[r];
            L.expert_in_slot[r] = (int32_t) e;
            L.slot_of_expert[e] = (int32_t) r;
            if (tbl) { tbl[e] = (int32_t) r; } // decouple: expert e resident in slot r (GPU table)
            L.last_used[r]      = (uint64_t) (n_slots - r); // hotter => more-recently-used
            L.pin_protected[r]  = r < H ? 1 : 0;
            for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
                if (!S->streams((int) il, kind)) { continue; }
                const size_t slab = model.uma_stream_slab_bytes((int) il, kind);
                ggml_tensor * slot = S->slot((int) il, kind);
                if (S->device_slots) {
                    if ((size_t) kind >= S->device_stage_host.size() ||
                        S->device_stage_host[kind] == nullptr || S->device_stage_bytes[kind] < slab ||
                        !model.uma_stream_pread_expert((int) il, kind, (int) e, S->device_stage_host[kind])) {
                        throw std::runtime_error(format("uma stream warm-start: pread failed (il=%u kind=%d e=%u)", il, kind, e));
                    }
                    ggml_backend_tensor_set(slot, S->device_stage_host[kind], (size_t) r * slab, slab);
                } else {
                    char * base = (char *) slot->data;
                    if (!model.uma_stream_pread_expert((int) il, kind, (int) e, base + (size_t) r * slab)) {
                        throw std::runtime_error(format("uma stream warm-start: pread failed (il=%u kind=%d e=%u)", il, kind, e));
                    }
                }
            }
        }
        L.tick = n_slots; // runtime LRU ticks continue above the seed
        seeded++;
    }
    fprintf(stderr, "uma: stream warm-start: seeded %u layers x top-%u experts, pinned top-%u (from %s)\n",
            seeded, n_slots, H, freq_path);
}

// uma-moe fork M6: reseed the resident set [0, s_new) from the per-layer freq
// ranking, preading each expert into the (freshly allocated) slot buffers and
// republishing the LRU + expert->slot table. Clears any prior residency first, so
// this is the single "rebuild the resident set at size s_new" primitive shared by
// warm-start's runtime equivalent and every resize.
void llama_context::uma_stream_reseed_resident(uint32_t s_new) {
    if (uma_stream && uma_stream->device_slots) {
        throw std::runtime_error("uma stream reseed: fixed CUDA device slots do not support live resize");
    }
    const uint32_t n_layer  = model.hparams.n_layer();
    const uint32_t n_expert = uma_stream->n_expert;
    const uint32_t n_used   = model.hparams.n_expert_used;
    if (s_new < n_used || s_new > uma_stream->n_slots) {
        throw std::runtime_error(format("uma stream reseed S=%u outside [%u,%u]", s_new, n_used, uma_stream->n_slots));
    }
    const uint32_t H        = std::min(uma_stream->pin_h, s_new > n_used ? s_new - n_used : 0u);
    for (uint32_t il = 0; il < n_layer; il++) {
        if (!uma_stream->streams_layer((int) il)) { continue; }
        llama_uma_stream_layer_lru & L = uma_stream->lru[il];
        int32_t * tbl = uma_stream->expert_table((int) il) ? (int32_t *) uma_stream->expert_table((int) il)->data : nullptr;
        const std::vector<int32_t> & rank = uma_stream->ranked[il];
        // clear all residency: no slot occupied, every expert non-resident (-> sentinel 0)
        std::fill(L.slot_of_expert.begin(), L.slot_of_expert.end(), -1);
        std::fill(L.expert_in_slot.begin(), L.expert_in_slot.end(), -1);
        std::fill(L.last_used.begin(), L.last_used.end(), 0);
        std::fill(L.pinned.begin(), L.pinned.end(), 0);
        std::fill(L.pin_protected.begin(), L.pin_protected.end(), 0);
        L.n_newly = 0;
        if (tbl) { std::fill(tbl, tbl + n_expert, 0); }
        // seed [0, s_new) with the top-s_new ranked experts; pin the top-H
        uint32_t r = 0;
        for (size_t ri = 0; r < s_new && ri < rank.size(); ri++) {
            const int32_t e = rank[ri];
            if (e < 0 || (uint32_t) e >= n_expert) { continue; }
            bool ok = true;
            for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
                if (!uma_stream->streams((int) il, kind)) { continue; }
                const size_t slab = model.uma_stream_slab_bytes((int) il, kind);
                char * base = (char *) uma_stream->slot((int) il, kind)->data;
                if (!model.uma_stream_pread_expert((int) il, kind, (int) e, base + (size_t) r * slab)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                throw std::runtime_error(format("uma stream reseed: pread failed (il=%u slot=%u expert=%d)", il, r, e));
            }
            L.expert_in_slot[r] = e;
            L.slot_of_expert[e] = (int32_t) r;
            if (tbl) { tbl[e] = (int32_t) r; }
            L.last_used[r]     = (uint64_t) (s_new - r);
            L.pin_protected[r] = r < H ? 1 : 0;
            r++;
        }
        L.tick = s_new;
    }
}

// uma-moe fork M6: elastic runtime resize of the resident slot set. A live Metal
// buffer pins its host pages (measured), so an in-place madvise cannot shed
// phys_footprint - only releasing the buffer does. So resize FREES all slot buffers
// (dropping the footprint), REALLOCATES them at the new size, reseeds the resident
// set from the freq ranking, and forces one graph rebuild (the reused decode graph
// references the old, freed slot tensors). Runs decode-only in the post-sync GPU-idle
// window. Frees-before-allocs so the peak footprint is max(old, new), never the sum.
// The CUDA device buffer over-allocates each quantized weight tensor's last row to MATRIX_ROW_PADDING
// (=512 elems) and zero-fills that tail (ggml-cuda get_alloc_size + init_tensor), because the mmvq/mmq
// mul_mat_id kernels deliberately over-read a quant row past ne0 up to that boundary (common.cuh:176
// "to avoid out-of-bounds memory accesses"). Our CUDA HOST (pinned) slot buffer uses the CPU
// get_alloc_size (no such padding); on an INTEGRATED GPU (GB10) the page-tight pinned allocation faults
// on that over-read -> illegal memory access on the first decode (seen on gpt-oss-120b MXFP4, but NOT on
// discrete-L4 Q4_K_M whose looser UVA mapping absorbed it). Return the bytes to mirror the device
// padding for the CUDA-host path (0 otherwise); only bites quant models whose expert ne0 % 512 != 0.
static size_t uma_stream_cuda_row_pad(bool cuda_host, ggml_type type, int64_t ne0) {
    if (!cuda_host || !ggml_is_quantized(type)) { return 0; }
    const int64_t rem = ne0 % 512; // MATRIX_ROW_PADDING (ggml-cuda/common.cuh, not includable here)
    return rem == 0 ? 0 : ggml_row_size(type, 512 - rem);
}

// M7.0: allocate ONE GPU-readable slot buffer holding `data_bytes` of expert data, dispatching by
// platform. Metal: mmap(MAP_ANON) + the no-rset wrap (GPU reads in place, unwired/reclaimable), out_alloc
// = page-aligned mmap length (munmap on free). CUDA/Spark: the pinned host buffer type the GPU reads in
// place (cudaHostAlloc, UVA; Task-C precedent, results/m2-spark.md), out_alloc = 0 (buffer owns the host
// memory; cudaFreeHost on reset). CUDA device slots use the device buft's tensor-aware allocation
// size and init hook, which add and zero MATRIX_ROW_PADDING exactly as stock CUDA tensors do. All paths
// mark WEIGHTS.
bool llama_context::uma_stream_alloc_slot_buf(ggml_tensor * slot, size_t data_bytes,
                                              ggml_backend_buffer_t * out_buf,
                                              void ** out_host, size_t * out_alloc) {
    *out_buf = nullptr; *out_host = nullptr; *out_alloc = 0;
    if (slot == nullptr || ggml_nbytes(slot) != data_bytes) {
        return false;
    }
    if (uma_stream_use_cuda_device) {
        const size_t bytes = ggml_backend_buft_get_alloc_size(uma_stream_cuda_device_buft, slot);
        const size_t pad = bytes - data_bytes;
        typedef ggml_backend_buffer_t (*vmm_alloc_fn_t)(
            ggml_backend_buffer_type_t, size_t, size_t, size_t, size_t);
        const vmm_alloc_fn_t alloc_fn = (vmm_alloc_fn_t) uma_stream_vmm_alloc_fn;
        ggml_backend_buffer_t buf = alloc_fn == nullptr ? nullptr : alloc_fn(
            uma_stream_cuda_device_buft, (size_t) slot->nb[2], (size_t) slot->ne[2],
            (size_t) uma_stream->n_slots, pad);
        if (buf == nullptr) { return false; }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        slot->buffer = buf;
        slot->data   = ggml_backend_buffer_get_base(buf);
        if (ggml_backend_buffer_init_tensor(buf, slot) != GGML_STATUS_SUCCESS) {
            slot->buffer = nullptr;
            slot->data   = nullptr;
            ggml_backend_buffer_free(buf);
            return false;
        }
        *out_buf   = buf;
        *out_host  = slot->data; // device base; never host-dereferenced
        *out_alloc = 0;
        return true;
    }
    const size_t pad   = uma_stream_cuda_row_pad(uma_stream_use_cuda_host, slot->type, slot->ne[0]);
    const size_t bytes = data_bytes + pad; // == data_bytes on Metal (pad=0 when !cuda_host)
    if (uma_stream_use_cuda_host) {
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(uma_stream_cuda_host_buft, bytes);
        if (buf == nullptr) { return false; }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        *out_buf   = buf;
        *out_host  = ggml_backend_buffer_get_base(buf);
        *out_alloc = 0; // buffer-owned host memory (freed via reset); no munmap
        if (pad) { memset((char *) *out_host + data_bytes, 0, pad); } // zero the kernel over-read tail
        slot->buffer = buf;
        slot->data   = *out_host;
        return true;
    }
    typedef ggml_backend_buffer_t (*wrap_fn_t)(ggml_backend_dev_t, void *, size_t, size_t);
    const wrap_fn_t wrap_fn = (wrap_fn_t) uma_stream_wrap_fn;
    const size_t page  = 16384;
    const size_t alloc = ((bytes + page - 1) / page) * page;
    void * host = mmap(nullptr, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (host == MAP_FAILED) { return false; }
    ggml_backend_buffer_t wrap = wrap_fn(uma_stream_gpu_dev, host, alloc, bytes);
    if (wrap == nullptr) { munmap(host, alloc); return false; }
    ggml_backend_buffer_set_usage(wrap, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    *out_buf   = wrap;
    *out_host  = host;
    *out_alloc = alloc;
    slot->buffer = wrap;
    slot->data   = host;
    return true;
}

// M7.0: free ONE slot buffer (dispatch). Release the buffer FIRST (Metal: unwire the wrap
// while it still views the host; CUDA: cudaFreeHost the owned host), then munmap the mmap'd
// host on Metal (alloc > 0). A CUDA buffer owns its host (alloc == 0) -> no munmap.
void llama_context::uma_stream_free_slot_buf(size_t i) {
    uma_stream->slot_buf[i].reset();
    if (uma_stream->slot_host[i] && uma_stream->slot_alloc[i] > 0) {
        munmap(uma_stream->slot_host[i], uma_stream->slot_alloc[i]);
    }
    uma_stream->slot_host[i]  = nullptr;
    uma_stream->slot_alloc[i] = 0;
    if (i < uma_stream->slots.size() && uma_stream->slots[i] != nullptr) {
        uma_stream->slots[i]->data   = nullptr;
        uma_stream->slots[i]->buffer = nullptr;
    }
}

void llama_context::uma_stream_resize(uint32_t s_new, bool park) {
    if (!uma_stream || !uma_stream->decouple ||
        (uma_stream_wrap_fn == nullptr && !uma_stream_use_cuda_host)) {
        return;
    }
    const uint32_t smax = uma_stream->n_slots;
    const uint32_t smin = uma_resize_smin >= 0 ? (uint32_t) uma_resize_smin : smax;
    // M7 improvement (1): a park command may go LEGALLY below the coherence knee to the
    // minimum viable resident set (n_expert_used) -- it is not decoding, so coherence does
    // not apply and no distress is raised. Any normal (non-park) resize keeps the knee floor.
    const uint32_t floor = park ? model.hparams.n_expert_used : smin;
    uint32_t target = s_new;
    if (target < floor) {
        if (!park) {
            uma_stream->n_distress++;
            fprintf(stderr, "uma: DISTRESS: budget demands S=%u < knee=%u; holding at knee (admission signal for M7)\n", s_new, smin);
        }
        target = floor;
    }
    if (target > smax) {
        target = smax;
    }
    const bool parked_before = uma_stream->parked;
    const uint32_t cur = uma_stream->n_slots_active;
    if (target == cur) {
        uma_stream->parked = park; // a normal no-op target still unparks
        return;
    }

    if (uma_stream->device_slots) {
        typedef bool (*vmm_resize_fn_t)(ggml_backend_buffer_t, size_t);
        typedef bool (*vmm_info_fn_t)(ggml_backend_buffer_t, size_t *, size_t *, size_t *, size_t *, size_t *);
        typedef bool (*vmm_stats_fn_t)(ggml_backend_buffer_t, uint64_t *, uint64_t *, uint64_t *, uint64_t *, void **);
        const vmm_resize_fn_t resize_fn = (vmm_resize_fn_t) uma_stream_vmm_resize_fn;
        const vmm_info_fn_t info_fn = (vmm_info_fn_t) uma_stream_vmm_info_fn;
        const vmm_stats_fn_t stats_fn = (vmm_stats_fn_t) uma_stream_vmm_stats_fn;
        if (resize_fn == nullptr || info_fn == nullptr || stats_fn == nullptr) {
            throw std::runtime_error("uma stream resize: CUDA VMM resize procedures unavailable");
        }

        ggml_backend_synchronize(uma_stream_cuda_backend);
        const size_t foot_before  = llama_uma_phys_footprint_mib();
        const size_t avail_before = llama_uma_avail_reclaim_mib();
        const int64_t t0_us = ggml_time_us();
        uint64_t mapped_before = 0;
        uint64_t map_ops_before = 0, unmap_ops_before = 0;
        uint64_t map_bytes_before = 0, unmap_bytes_before = 0;
        const uint64_t miss_before = uma_stream->n_miss;
        const uint64_t h2d_before = uma_stream->n_h2d_miss;
        std::vector<void *> bases_before;
        uint32_t n_buffers = 0;
        for (const auto & owner : uma_stream->slot_buf) {
            if (!owner) { continue; }
            size_t active = 0, max = 0, mapped = 0;
            if (!info_fn(owner.get(), &active, &max, &mapped, nullptr, nullptr) || active != cur || max != smax) {
                throw std::runtime_error("uma stream resize: inconsistent CUDA VMM slot buffer before resize");
            }
            uint64_t map_ops = 0, unmap_ops = 0, map_bytes = 0, unmap_bytes = 0;
            void * base = nullptr;
            if (!stats_fn(owner.get(), &map_ops, &unmap_ops, &map_bytes, &unmap_bytes, &base)) {
                throw std::runtime_error("uma stream resize: missing CUDA VMM slot-buffer stats");
            }
            map_ops_before += map_ops;
            unmap_ops_before += unmap_ops;
            map_bytes_before += map_bytes;
            unmap_bytes_before += unmap_bytes;
            bases_before.push_back(base);
            mapped_before += mapped;
            n_buffers++;
        }

        std::vector<llama_uma_stream_layer_lru> lru_before;
        std::vector<std::vector<int32_t>> tables_before;
        if (target > cur) {
            lru_before = uma_stream->lru;
            tables_before.resize(uma_stream->lru.size());
            for (uint32_t il = 0; il < uma_stream->lru.size(); il++) {
                const int32_t * tbl = uma_stream->expert_table((int) il) ?
                    (const int32_t *) uma_stream->expert_table((int) il)->data : nullptr;
                if (tbl) {
                    tables_before[il].assign(tbl, tbl + uma_stream->n_expert);
                }
            }
        }

        // A shrink first unpublishes every tail resident while all backing remains
        // mapped. No graph can consume a tail id after this point; the GPU is idle.
        if (target < cur) {
            for (uint32_t il = 0; il < uma_stream->lru.size(); il++) {
                if (!uma_stream->streams_layer((int) il)) { continue; }
                llama_uma_stream_layer_lru & L = uma_stream->lru[il];
                int32_t * tbl = uma_stream->expert_table((int) il) ?
                    (int32_t *) uma_stream->expert_table((int) il)->data : nullptr;
                for (uint32_t s = target; s < cur; s++) {
                    const int32_t e = L.expert_in_slot[s];
                    if (e >= 0) {
                        L.slot_of_expert[e] = -1;
                        if (tbl) { tbl[e] = 0; }
                    }
                    L.expert_in_slot[s] = -1;
                    L.last_used[s] = 0;
                    L.pinned[s] = 0;
                    L.pin_protected[s] = 0;
                }
                L.n_newly = 0;
            }
        }

        int64_t t_free_us = t0_us;
        int64_t t_alloc_us = t0_us;
        std::vector<ggml_backend_buffer_t> changed;
        changed.reserve(n_buffers);
        if (target < cur) {
            for (const auto & owner : uma_stream->slot_buf) {
                if (owner && !resize_fn(owner.get(), target)) {
                    GGML_ABORT("uma stream resize: CUDA VMM tail unmap failed");
                }
            }
            t_free_us = ggml_time_us();
            t_alloc_us = t_free_us;
        } else {
            for (const auto & owner : uma_stream->slot_buf) {
                if (!owner) { continue; }
                if (!resize_fn(owner.get(), target)) {
                    for (ggml_backend_buffer_t done : changed) {
                        if (!resize_fn(done, cur)) {
                            GGML_ABORT("uma stream resize: CUDA VMM grow rollback failed");
                        }
                    }
                    uma_stream->parked = parked_before;
                    throw std::runtime_error("uma stream resize: CUDA VMM grow allocation failed");
                }
                changed.push_back(owner.get());
            }
            t_free_us = t0_us;
            t_alloc_us = ggml_time_us();
        }
        const size_t foot_after_free = target < cur ? llama_uma_phys_footprint_mib() : foot_before;
        const size_t avail_after_free = target < cur ? llama_uma_avail_reclaim_mib() : avail_before;
        uint32_t seeded = 0;
        uint64_t seed_bytes = 0;
        int64_t t_reseed_us = 0;
        uint64_t mapped_after = 0;
        uint64_t map_ops_after = 0, unmap_ops_after = 0;
        uint64_t map_bytes_after = 0, unmap_bytes_after = 0;
        bool base_stable = true;

        try {
            // Every persistent tensor retains its buffer and base address. Only its
            // active expert dimension changes; zero the CUDA quant tail at the new end.
            for (ggml_tensor * st : uma_stream->slots) {
                if (st == nullptr) { continue; }
                st->ne[2] = (int64_t) target;
                st->nb[3] = st->nb[2] * (size_t) target;
                if (ggml_backend_buffer_init_tensor(st->buffer, st) != GGML_STATUS_SUCCESS) {
                    uma_stream_force_rebuild = true;
                    throw std::runtime_error("uma stream resize: CUDA VMM tensor reinitialization failed");
                }
            }
            uma_stream_force_rebuild = true;

            // Grow fills only the newly mapped tail from the frozen ranking. Retained
            // slots and their LRU ages are byte-for-byte untouched. An expert is
            // published only after all three kinds have reached CUDA and synchronized.
            if (target > cur) {
            for (uint32_t il = 0; il < uma_stream->lru.size(); il++) {
                if (!uma_stream->streams_layer((int) il)) { continue; }
                llama_uma_stream_layer_lru & L = uma_stream->lru[il];
                int32_t * tbl = uma_stream->expert_table((int) il) ?
                    (int32_t *) uma_stream->expert_table((int) il)->data : nullptr;
                const std::vector<int32_t> & rank = uma_stream->ranked[il];
                size_t ri = 0;
                for (uint32_t s = cur; s < target; s++) {
                    int32_t e = -1;
                    while (ri < rank.size()) {
                        const int32_t candidate = rank[ri++];
                        if (candidate >= 0 && (uint32_t) candidate < uma_stream->n_expert &&
                            L.slot_of_expert[candidate] < 0) {
                            e = candidate;
                            break;
                        }
                    }
                    if (e < 0) {
                        throw std::runtime_error(format("uma stream resize: no ranked expert for grow il=%u slot=%u", il, s));
                    }
                    for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
                        if (!uma_stream->streams((int) il, kind)) { continue; }
                        const size_t slab = model.uma_stream_slab_bytes((int) il, kind);
                        if ((size_t) kind >= uma_stream->device_stage_host.size() ||
                            uma_stream->device_stage_host[kind] == nullptr ||
                            uma_stream->device_stage_bytes[kind] < slab ||
                            !model.uma_stream_pread_expert((int) il, kind, e,
                                                           uma_stream->device_stage_host[kind])) {
                            throw std::runtime_error(format(
                                "uma stream resize: grow pread failed il=%u kind=%d slot=%u expert=%d",
                                il, kind, s, e));
                        }
                    }
                    uint64_t uploaded = 0;
                    for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
                        if (!uma_stream->streams((int) il, kind)) { continue; }
                        const size_t slab = model.uma_stream_slab_bytes((int) il, kind);
                        ggml_backend_tensor_set_async(
                            uma_stream_cuda_backend, uma_stream->slot((int) il, kind),
                            uma_stream->device_stage_host[kind], (size_t) s * slab, slab);
                        uploaded += slab;
                    }
                    ggml_backend_synchronize(uma_stream_cuda_backend);
                    L.expert_in_slot[s] = e;
                    L.slot_of_expert[e] = (int32_t) s;
                    L.last_used[s] = L.tick++;
                    L.pinned[s] = 0;
                    if (tbl) { tbl[e] = (int32_t) s; }
                    seeded++;
                    seed_bytes += uploaded;
                }
                L.n_newly = 0;
            }
            }

            uma_stream->n_slots_active = target;
            uma_stream->parked = park; // publish serving state only with committed S
            t_reseed_us = ggml_time_us();
            size_t bi = 0;
            for (const auto & owner : uma_stream->slot_buf) {
                if (!owner) { continue; }
                size_t active = 0, max = 0, mapped = 0;
                if (!info_fn(owner.get(), &active, &max, &mapped, nullptr, nullptr) ||
                    active != target || max != smax) {
                    throw std::runtime_error("uma stream resize: inconsistent CUDA VMM slot buffer after resize");
                }
                uint64_t map_ops = 0, unmap_ops = 0, map_bytes = 0, unmap_bytes = 0;
                void * base = nullptr;
                if (!stats_fn(owner.get(), &map_ops, &unmap_ops, &map_bytes, &unmap_bytes, &base)) {
                    throw std::runtime_error("uma stream resize: missing final CUDA VMM slot-buffer stats");
                }
                if (bi >= bases_before.size() || base != bases_before[bi]) { base_stable = false; }
                bi++;
                map_ops_after += map_ops;
                unmap_ops_after += unmap_ops;
                map_bytes_after += map_bytes;
                unmap_bytes_after += unmap_bytes;
                mapped_after += mapped;
            }
            if (bi != bases_before.size() || !base_stable) {
                throw std::runtime_error("uma stream resize: CUDA VMM base address changed");
            }
        } catch (...) {
            if (target < cur) {
                // Shrink has already discarded physical tail contents. A failure
                // after that point cannot safely resume inference, so fail-stop.
                GGML_ABORT("uma stream resize: irreversible CUDA VMM shrink failed");
            }
            ggml_backend_synchronize(uma_stream_cuda_backend);
            uma_stream->lru = lru_before;
            for (uint32_t il = 0; il < tables_before.size(); il++) {
                int32_t * tbl = uma_stream->expert_table((int) il) ?
                    (int32_t *) uma_stream->expert_table((int) il)->data : nullptr;
                if (tbl && !tables_before[il].empty()) {
                    std::copy(tables_before[il].begin(), tables_before[il].end(), tbl);
                }
            }
            uma_stream->n_slots_active = cur;
            for (ggml_tensor * st : uma_stream->slots) {
                if (st == nullptr) { continue; }
                st->ne[2] = (int64_t) cur;
                st->nb[3] = st->nb[2] * (size_t) cur;
            }
            for (const auto & owner : uma_stream->slot_buf) {
                if (owner && !resize_fn(owner.get(), cur)) {
                    GGML_ABORT("uma stream resize: CUDA VMM grow rollback unmap failed");
                }
            }
            for (ggml_tensor * st : uma_stream->slots) {
                if (st != nullptr && ggml_backend_buffer_init_tensor(st->buffer, st) != GGML_STATUS_SUCCESS) {
                    GGML_ABORT("uma stream resize: CUDA VMM grow rollback tensor init failed");
                }
            }
            uma_stream->parked = parked_before;
            uma_stream_force_rebuild = true;
            throw;
        }
        // Fixed-S decode keeps the validated Task-21 fast path. A committed live
        // S change can invalidate residents that the next token selects, so every
        // subsequent decode graph services cold experts before use.
        uma_stream->decode_exact_after_resize = true;
        if (map_ops_after < map_ops_before || unmap_ops_after < unmap_ops_before ||
            map_bytes_after < map_bytes_before || unmap_bytes_after < unmap_bytes_before) {
            GGML_ABORT("uma stream resize: CUDA VMM operation counters went backwards");
        }
        const uint64_t map_ops_delta = map_ops_after - map_ops_before;
        const uint64_t unmap_ops_delta = unmap_ops_after - unmap_ops_before;
        const uint64_t map_bytes_delta = map_bytes_after - map_bytes_before;
        const uint64_t unmap_bytes_delta = unmap_bytes_after - unmap_bytes_before;
        const uint64_t vmm_bytes = map_bytes_delta + unmap_bytes_delta;
        const bool mapped_accounting_ok = target > cur ?
            (mapped_after >= mapped_before && mapped_after - mapped_before == map_bytes_delta && unmap_bytes_delta == 0) :
            (mapped_before >= mapped_after && mapped_before - mapped_after == unmap_bytes_delta && map_bytes_delta == 0);
        const uint64_t miss_delta = uma_stream->n_miss - miss_before;
        const uint64_t h2d_delta = uma_stream->n_h2d_miss - h2d_before;
        if (miss_delta != 0 || h2d_delta != 0 || !base_stable || !mapped_accounting_ok) {
            GGML_ABORT("uma stream resize: admission, stable-base, or VMM byte accounting invariant failed");
        }
        uma_stream->last_resize_free_us = (uint64_t) (t_free_us - t0_us);
        uma_stream->last_resize_alloc_us = (uint64_t) (t_alloc_us - t_free_us);
        uma_stream->last_resize_reseed_us = (uint64_t) (t_reseed_us - t_alloc_us);
        uma_stream->last_resize_old_s = cur;
        uma_stream->last_resize_new_s = target;
        uma_stream->last_resize_delta_s = cur > target ? cur - target : target - cur;
        uma_stream->last_resize_segments = (uint32_t) (map_ops_delta + unmap_ops_delta);
        uma_stream->last_resize_seeded = seeded;
        uma_stream->last_resize_vmm_bytes = vmm_bytes;
        uma_stream->last_resize_seed_bytes = seed_bytes;
        uma_stream->last_resize_map_ops = (uint32_t) map_ops_delta;
        uma_stream->last_resize_unmap_ops = (uint32_t) unmap_ops_delta;
        uma_stream->last_resize_map_bytes = map_bytes_delta;
        uma_stream->last_resize_unmap_bytes = unmap_bytes_delta;
        uma_stream->last_resize_miss_delta = miss_delta;
        uma_stream->last_resize_h2d_delta = h2d_delta;
        uma_stream->last_resize_base_stable = base_stable;
        uma_stream->last_resize_foot_after_free = foot_after_free;
        uma_stream->last_resize_avail_after_free = avail_after_free;
        uma_stream->n_resizes++;
        if (uma_stream->s_min_active == 0 || target < uma_stream->s_min_active) { uma_stream->s_min_active = target; }
        if (target > uma_stream->s_max_active) { uma_stream->s_max_active = target; }
        const size_t foot_after = llama_uma_phys_footprint_mib();
        fprintf(stderr,
            "uma: segmented device resize S %u -> %u: logical delta %u, actual map/unmap %u/%u, VMM map/unmap %.1f/%.1f MiB, stable-base %u, "
            "seeded %u experts / %.1f MiB (free %.3f + alloc %.3f + reseed %.3f = %.3f ms), "
            "request miss/H2D delta %llu/%llu, phys_footprint %zu -> %zu MiB, avail %zu -> %zu MiB\n",
            cur, target, uma_stream->last_resize_delta_s,
            uma_stream->last_resize_map_ops, uma_stream->last_resize_unmap_ops,
            map_bytes_delta / (1024.0 * 1024.0), unmap_bytes_delta / (1024.0 * 1024.0),
            (unsigned) base_stable, seeded, seed_bytes / (1024.0 * 1024.0),
            uma_stream->last_resize_free_us / 1000.0, uma_stream->last_resize_alloc_us / 1000.0,
            uma_stream->last_resize_reseed_us / 1000.0,
            (t_reseed_us - t0_us) / 1000.0,
            (unsigned long long) miss_delta, (unsigned long long) h2d_delta,
            foot_before, foot_after, avail_before,
            llama_uma_avail_reclaim_mib());
        uma_write_telemetry();
        return;
    }
    const size_t   foot_before  = llama_uma_phys_footprint_mib();
    const size_t   avail_before = llama_uma_avail_reclaim_mib();
    const int64_t  t0_us       = ggml_time_us();
    const uint32_t n_layer     = model.hparams.n_layer();

    // 1. free ALL slot buffers first (Metal: release wrap + munmap -> pages to the OS;
    //    CUDA: cudaFreeHost the pinned host), dropping phys_footprint/VmRSS to the non-expert
    //    baseline before any new alloc.
    for (uint32_t il = 0; il < n_layer; il++) {
        for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
            const size_t i = (size_t) il * LLAMA_UMA_STREAM_N_KIND + kind;
            if (uma_stream->slots[i] == nullptr) { continue; }
            uma_stream_free_slot_buf(i);
        }
    }
    const int64_t t_free_us = ggml_time_us();
    const size_t foot_after_free  = llama_uma_phys_footprint_mib();
    const size_t avail_after_free = llama_uma_avail_reclaim_mib();
    uma_stream->last_resize_free_us          = (uint64_t) (t_free_us - t0_us);
    uma_stream->last_resize_foot_after_free  = foot_after_free;
    uma_stream->last_resize_avail_after_free = avail_after_free;
    fprintf(stderr, "uma: resize old-pool release point: phys_footprint %zu -> %zu MiB, avail %zu -> %zu MiB (%.1f ms)\n",
            foot_before, foot_after_free, avail_before, avail_after_free,
            (t_free_us - t0_us) / 1000.0);
    // 2. reallocate every slot buffer at the new size, with an OOM fallback ladder so a
    //    resize NEVER crashes the process: a grow that cannot get memory degrades to the
    //    pre-resize size (just resident, so it fits), then to the minimum, before it gives
    //    up. Each try_alloc rolls back its own partial allocations on failure.
    uint32_t eff = target;
    if (!uma_stream_try_alloc_slots(eff)) {
        fprintf(stderr, "uma: resize to S=%u could not allocate; falling back to S=%u\n", target, cur);
        eff = cur;
        if (!uma_stream_try_alloc_slots(eff)) {
            eff = floor; // only PARK may fall below the coherence knee
            if (!uma_stream_try_alloc_slots(eff)) {
                throw std::runtime_error("uma stream resize: cannot reallocate the slot pool (OOM)");
            }
        }
    }
    uma_stream->n_slots_active = eff;
    uma_stream->parked = park;
    const int64_t t_alloc_us = ggml_time_us();
    uma_stream->last_resize_alloc_us = (uint64_t) (t_alloc_us - t_free_us);

    // 3. reseed the resident set into the new buffers (pread from the GGUF)
    uma_stream_reseed_resident(eff);
    const int64_t t_reseed_us = ggml_time_us();
    uma_stream->last_resize_reseed_us = (uint64_t) (t_reseed_us - t_alloc_us);

    // 4. force the next decode to rebuild the graph (it references the freed tensors).
    //    the new wraps share the no-rset buft already allowed at setup, and
    //    sched_reset does not clear that allowlist, so no re-registration is needed.
    uma_stream_force_rebuild = true;
    uma_stream->n_resizes++;
    if (uma_stream->s_min_active == 0 || eff < uma_stream->s_min_active) { uma_stream->s_min_active = eff; }
    if (eff > uma_stream->s_max_active) { uma_stream->s_max_active = eff; }
    const size_t foot_after = llama_uma_phys_footprint_mib();
    fprintf(stderr, "uma: resize S %u -> %u: phys_footprint %zu -> %zu MiB (free %.1f + alloc %.1f + reseed %.1f = %.1f ms), avail %zu MiB\n",
            cur, eff, foot_before, foot_after,
            (t_free_us - t0_us) / 1000.0, (t_alloc_us - t_free_us) / 1000.0,
            (t_reseed_us - t_alloc_us) / 1000.0, (t_reseed_us - t0_us) / 1000.0,
            llama_uma_avail_reclaim_mib());
    uma_write_telemetry(); // M7.1: publish the new state (incl. any distress) immediately
}

// uma-moe fork M6: (re)allocate every streaming slot buffer at S_new (mmap + no-rset wrap),
// mutating the persistent slot tensors. Precondition: the slot buffers are already freed
// (slot_host == nullptr). Returns false + rolls back its own partial allocations on any
// mmap/wrap failure, so the caller can retry at a smaller size (the OOM fallback ladder).
bool llama_context::uma_stream_try_alloc_slots(uint32_t s_new) {
    if (uma_stream && uma_stream->device_slots) {
        return false;
    }
    if (s_new < model.hparams.n_expert_used || s_new > uma_stream->n_slots) {
        return false;
    }
    const uint32_t  n_layer = model.hparams.n_layer();
    std::vector<size_t> done; // indices allocated in THIS call, for rollback
    auto rollback = [&]() { for (size_t i : done) { uma_stream_free_slot_buf(i); } };
    for (uint32_t il = 0; il < n_layer; il++) {
        for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
            const size_t i = (size_t) il * LLAMA_UMA_STREAM_N_KIND + kind;
            ggml_tensor * st = uma_stream->slots[i];
            if (st == nullptr) { continue; }
            const size_t slab       = (size_t) st->nb[2];
            const size_t slot_bytes = (size_t) s_new * slab;
            st->ne[2] = (int64_t) s_new;
            st->nb[3] = st->nb[2] * (size_t) s_new;
            ggml_backend_buffer_t buf; void * host; size_t alloc;
            if (!uma_stream_alloc_slot_buf(st, slot_bytes, &buf, &host, &alloc)) {
                rollback();
                return false;
            }
            uma_stream->slot_host[i]  = host;
            uma_stream->slot_alloc[i] = alloc;
            uma_stream->slot_buf[i].reset(buf);
            done.push_back(i);
        }
    }
    return true;
}

// M7 improvement (1): control-file sentinel for a serving-state PARK command (distinct
// from -1 "hold"). The coordinator writes the token "park" to KV-only a not-decoding tenant.
static constexpr int32_t UMA_CTRL_PARK = -2;

// uma-moe fork M6: rate-limited controller tick (decode-only, post-sync). Advances
// the decode-token clock, then resolves a target S from the commanded schedule
// (SCHED, fires at its exact token points) or the closed-loop budget signal (CTRL,
// polled every period tokens) and calls uma_stream_resize().
void llama_context::uma_stream_controller_tick() {
    uma_resize_dtoken++;
    while (uma_resize_sched_i < uma_resize_sched.size() &&
           uma_resize_sched[uma_resize_sched_i].first <= uma_resize_dtoken) {
        const int32_t target = uma_resize_sched[uma_resize_sched_i].second;
        uma_resize_sched_i++;
        uma_stream_resize((uint32_t) (target < 0 ? 0 : target));
    }
    // M7.1: external control (the cross-tenant coordinator drives S) takes precedence over
    // the local CTRL watermark; telemetry is exported whenever its path is set. Rate-limited.
    const bool external  = !uma_control_path.empty();
    const bool ctrl      = uma_resize_lowmib > 0;
    const bool telemetry = !uma_telemetry_path.empty();
    if (!external && !ctrl && !telemetry) {
        return; // SCHED-only or armed-idle
    }
    if (++uma_resize_tick < uma_resize_period) {
        return;
    }
    uma_resize_tick = 0;
    if (external) {
        const int32_t target = uma_read_control();
        if (target == UMA_CTRL_PARK) {
            if (!uma_stream->parked) {
                // KV-only: resize to the min viable resident set, legally below the knee.
                uma_stream_resize(model.hparams.n_expert_used, /*park=*/true);
            }
        } else if (target >= 0 && ((uint32_t) target != uma_stream->n_slots_active || uma_stream->parked)) {
            uma_stream_resize((uint32_t) target); // a normal target unparks + clamps below-knee -> distress
        }
    } else if (ctrl) {
        const size_t   avail = llama_uma_avail_reclaim_mib();
        const uint32_t cur   = uma_stream->n_slots_active;
        const uint32_t step  = (uint32_t) (uma_resize_step > 0 ? uma_resize_step : 1);
        if ((int32_t) avail < uma_resize_lowmib && cur > (uint32_t) uma_resize_smin) {
            uma_stream_resize(cur > step ? cur - step : 0); // pressure: shed toward the knee
        } else if (uma_resize_highmib > 0 && (int32_t) avail > uma_resize_highmib && cur < uma_stream->n_slots) {
            uma_stream_resize(cur + step); // eased: grow toward S_max
        }
    }
    if (telemetry) {
        uma_write_telemetry();
    }
}

// M7.1: read a target S from the external control file (the cross-tenant coordinator
// writes it). Returns the parsed slot count, or -1 if the file is absent/unreadable/
// unparseable (the controller then holds the current S). uma_stream_resize clamps a
// below-knee target and raises distress, so an external command cannot force incoherence.
int32_t llama_context::uma_read_control() {
    FILE * f = fopen(uma_control_path.c_str(), "r");
    if (f == nullptr) {
        return -1;
    }
    char tok[32] = {0};
    int32_t ret = -1;                       // absent/garbage -> hold current S
    if (fscanf(f, "%31s", tok) == 1) {
        if (strcmp(tok, "park") == 0) {
            ret = UMA_CTRL_PARK;            // -2: serving-state park (KV-only, legal below-knee)
        } else {
            char * end = nullptr;
            const long v = strtol(tok, &end, 10);
            if (end != tok && *end == '\0' && v >= 0 && v <= 1000000) {
                ret = (int32_t) v;         // a valid S target (resize clamps below-knee -> distress)
            }
        }
    }
    fclose(f);
    return ret;
}

// M7.1: atomically export the per-tenant state the arbiter consumes (write a temp file
// then rename over the target, so the coordinator never reads a partial line). Flat KV,
// matching the fork's profile-artifact style. phys_footprint + s_active let the arbiter
// derive F (fixed) = footprint - S*slot_bytes.
void llama_context::uma_write_telemetry() {
    if (uma_telemetry_path.empty() || !uma_stream) {
        return;
    }
    const std::string tmp = uma_telemetry_path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "w");
    if (f == nullptr) {
        return;
    }
    fprintf(f,
            "s_active %u\ns_ceiling %u\nn_expert_used %u\nn_miss %llu\nn_read %llu\n"
            "n_h2d_miss %llu\nn_h2d_bytes %llu\n"
            "n_prefill_tiles %llu\nn_prefill_cold_miss %llu\nn_prefill_h2d_miss %llu\nn_prefill_h2d_bytes %llu\n"
            "n_prefill_service_fail %llu\nn_prefill_substitute %llu\nprefill_max_distinct %u\n"
            "n_decode_service %llu\nn_decode_cold_miss %llu\nn_decode_h2d_miss %llu\nn_decode_h2d_bytes %llu\n"
            "n_decode_service_fail %llu\nn_decode_substitute %llu\n"
            "n_decode_late_calls %llu\nn_decode_late_cold_miss %llu\nn_decode_late_h2d_miss %llu\nn_decode_late_h2d_bytes %llu\n"
            "n_decode_late_fail %llu\ndecode_exact_after_resize %u\n"
            "n_distress %llu\nn_overflow %llu\nn_resizes %llu\ns_min_reached %u\ns_max_reached %u\nphys_footprint_mib %zu\n"
            "last_resize_free_ms %.3f\nlast_resize_alloc_ms %.3f\nlast_resize_reseed_ms %.3f\n"
            "last_resize_old_s %u\nlast_resize_new_s %u\nlast_resize_delta_s %u\nlast_resize_segments %u\n"
            "last_resize_seeded %u\nlast_resize_vmm_bytes %llu\nlast_resize_seed_bytes %llu\n"
            "last_resize_map_ops %u\nlast_resize_unmap_ops %u\nlast_resize_map_bytes %llu\nlast_resize_unmap_bytes %llu\n"
            "last_resize_miss_delta %llu\nlast_resize_h2d_delta %llu\nlast_resize_base_stable %u\n"
            "last_resize_foot_after_free_mib %zu\nlast_resize_avail_after_free_mib %zu\n"
            "parked %u\n",
            uma_stream->n_slots_active, uma_stream->n_slots, model.hparams.n_expert_used,
            (unsigned long long) uma_stream->n_miss, (unsigned long long) uma_stream->n_read,
            (unsigned long long) uma_stream->n_h2d_miss, (unsigned long long) uma_stream->n_h2d_bytes,
            (unsigned long long) uma_stream->n_prefill_tiles,
            (unsigned long long) uma_stream->n_prefill_cold_miss,
            (unsigned long long) uma_stream->n_prefill_h2d_miss,
            (unsigned long long) uma_stream->n_prefill_h2d_bytes,
            (unsigned long long) uma_stream->n_prefill_service_fail,
            (unsigned long long) uma_stream->n_prefill_substitute,
            uma_stream->prefill_max_distinct,
            (unsigned long long) uma_stream->n_decode_service,
            (unsigned long long) uma_stream->n_decode_cold_miss,
            (unsigned long long) uma_stream->n_decode_h2d_miss,
            (unsigned long long) uma_stream->n_decode_h2d_bytes,
            (unsigned long long) uma_stream->n_decode_service_fail,
            (unsigned long long) uma_stream->n_decode_substitute,
            (unsigned long long) uma_stream->n_decode_late_calls,
            (unsigned long long) uma_stream->n_decode_late_cold_miss,
            (unsigned long long) uma_stream->n_decode_late_h2d_miss,
            (unsigned long long) uma_stream->n_decode_late_h2d_bytes,
            (unsigned long long) uma_stream->n_decode_late_fail,
            (unsigned) uma_stream->decode_exact_after_resize,
            (unsigned long long) uma_stream->n_distress, (unsigned long long) uma_stream->n_overflow, (unsigned long long) uma_stream->n_resizes,
            uma_stream->s_min_active, uma_stream->s_max_active, llama_uma_phys_footprint_mib(),
            uma_stream->last_resize_free_us / 1000.0, uma_stream->last_resize_alloc_us / 1000.0,
            uma_stream->last_resize_reseed_us / 1000.0,
            uma_stream->last_resize_old_s, uma_stream->last_resize_new_s,
            uma_stream->last_resize_delta_s, uma_stream->last_resize_segments,
            uma_stream->last_resize_seeded,
            (unsigned long long) uma_stream->last_resize_vmm_bytes,
            (unsigned long long) uma_stream->last_resize_seed_bytes,
            uma_stream->last_resize_map_ops, uma_stream->last_resize_unmap_ops,
            (unsigned long long) uma_stream->last_resize_map_bytes,
            (unsigned long long) uma_stream->last_resize_unmap_bytes,
            (unsigned long long) uma_stream->last_resize_miss_delta,
            (unsigned long long) uma_stream->last_resize_h2d_delta,
            (unsigned) uma_stream->last_resize_base_stable,
            uma_stream->last_resize_foot_after_free, uma_stream->last_resize_avail_after_free,
            (unsigned) (uma_stream->parked ? 1 : 0));
    fclose(f);
    rename(tmp.c_str(), uma_telemetry_path.c_str());
}

void llama_context::uma_stream_setup() {
    const uint32_t k = model.uma_stream_k();
    if (k == 0) {
        if (llama_uma_stream_device_slots_enabled()) {
            throw std::runtime_error("LLAMA_UMA_STREAM_DEVICE_SLOTS requested but streaming manifest is inactive");
        }
        return; // streaming off (default runs)
    }
    const uint32_t n_expert      = model.hparams.n_expert;
    const uint32_t n_expert_used = model.hparams.n_expert_used;

    // LLAMA_UMA_STREAM_S is the initial active resident set; SMAX is the
    // independent controller ceiling. An omitted SMAX preserves old S=fixed behavior.
    const llama_uma_stream_s_config s_cfg = llama_uma_stream_parse_s(n_expert_used, n_expert);
    const uint32_t n_slots_initial = s_cfg.initial;
    const uint32_t n_slots         = s_cfg.ceiling;
    const bool device_slots_requested = llama_uma_stream_device_slots_enabled();

    // GPU backend + the fork's no-rset Metal wrap entry (the WO-C zero-copy path)
    ggml_backend_t gpu_backend = nullptr;
    for (ggml_backend_t b : backend_ptrs) {
        ggml_backend_dev_t d = ggml_backend_get_device(b);
        if (d == nullptr) {
            continue;
        }
        const enum ggml_backend_dev_type dt = ggml_backend_dev_type(d);
        if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_backend = b;
            break;
        }
    }
    if (gpu_backend == nullptr) {
        throw std::runtime_error("uma stream: LLAMA_UMA_STREAM_K needs a GPU backend (Metal in S1.1.1)");
    }
    ggml_backend_dev_t gpu_dev = ggml_backend_get_device(gpu_backend);
    ggml_backend_reg_t gpu_reg = gpu_dev == nullptr ? nullptr : ggml_backend_dev_backend_reg(gpu_dev);
    typedef ggml_backend_buffer_t (*buffer_mapped_norset_t)(ggml_backend_dev_t, void *, size_t, size_t);
    buffer_mapped_norset_t wrap_fn = gpu_reg == nullptr ? nullptr :
        (buffer_mapped_norset_t) ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_metal_device_buffer_mapped_norset");
    // M7.0: Metal uses the no-rset wrap; CUDA/Spark uses the pinned host buffer type the GPU
    // reads in place (the Task-C precedent, results/m2-spark.md; llama-uma.cpp:525 routes
    // weights to exactly this buft). cudaFreeHost on resize drops VmRSS as munmap does on Metal.
    if (wrap_fn == nullptr) {
        uma_stream_cuda_host_buft = ggml_backend_dev_host_buffer_type(gpu_dev);
        if (uma_stream_cuda_host_buft == nullptr) {
            throw std::runtime_error("uma stream: no Metal no-rset wrap AND no host buffer type (unsupported backend)");
        }
        uma_stream_use_cuda_host = true;
    }
    if (device_slots_requested) {
        if (wrap_fn != nullptr || !uma_stream_use_cuda_host) {
            throw std::runtime_error("LLAMA_UMA_STREAM_DEVICE_SLOTS is CUDA-only");
        }
        uma_stream_cuda_device_buft = ggml_backend_dev_buffer_type(gpu_dev);
        if (uma_stream_cuda_device_buft == nullptr ||
            strncmp(ggml_backend_buft_name(uma_stream_cuda_device_buft), "CUDA", 4) != 0) {
            throw std::runtime_error("LLAMA_UMA_STREAM_DEVICE_SLOTS requires the CUDA device buffer type");
        }
        uma_stream_use_cuda_device = true;
        uma_stream_cuda_backend = gpu_backend;
        uma_stream_vmm_alloc_fn = gpu_reg == nullptr ? nullptr :
            ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_cuda_vmm_slot_buffer_alloc");
        uma_stream_vmm_resize_fn = gpu_reg == nullptr ? nullptr :
            ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_cuda_vmm_slot_buffer_resize");
        uma_stream_vmm_info_fn = gpu_reg == nullptr ? nullptr :
            ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_cuda_vmm_slot_buffer_get_info");
        uma_stream_vmm_stats_fn = gpu_reg == nullptr ? nullptr :
            ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_cuda_vmm_slot_buffer_get_stats");
        if (uma_stream_vmm_alloc_fn == nullptr || uma_stream_vmm_resize_fn == nullptr ||
            uma_stream_vmm_info_fn == nullptr || uma_stream_vmm_stats_fn == nullptr) {
            throw std::runtime_error(
                "LLAMA_UMA_STREAM_DEVICE_SLOTS requires CUDA VMM segmented-buffer support");
        }
    }
    // M6: keep the device + wrap entry so a runtime resize can reallocate slot buffers
    uma_stream_gpu_dev = gpu_dev;
    uma_stream_wrap_fn = (void *) wrap_fn;

    // build the resident slot pool ONCE (context lifetime)
    if (!uma_stream) {
        uma_stream = std::make_unique<llama_uma_stream_state>();
        uma_stream->model    = &model;
        uma_stream->n_slots        = n_slots;
        uma_stream->n_slots_active = n_slots_initial;
        uma_stream->s_min_active   = n_slots_initial;
        uma_stream->s_max_active   = n_slots_initial;
        uma_stream->n_expert = n_expert;
        // decouple mode (Part 1): GPU-gather decode routing needs a static, warm-started
        // resident set (the slots do not change during decode), so HOTFREQ is required.
        const bool adapt    = getenv("LLAMA_UMA_STREAM_ADAPT") != nullptr; // Step 3: online maintenance
        const bool decouple = adapt || getenv("LLAMA_UMA_STREAM_DECOUPLE") != nullptr; // adapt needs the table
        if (device_slots_requested && !decouple) {
            throw std::runtime_error("LLAMA_UMA_STREAM_DEVICE_SLOTS requires DECOUPLE/ADAPT");
        }
        if (device_slots_requested && n_slots_initial < n_expert && !adapt) {
            throw std::runtime_error("compressed LLAMA_UMA_STREAM_DEVICE_SLOTS requires LLAMA_UMA_STREAM_ADAPT");
        }
        uma_stream->decouple = decouple;
        uma_stream->adapt    = adapt;
        uma_stream->device_slots = device_slots_requested;
        uma_stream->device_backend = device_slots_requested ? gpu_backend : nullptr;
        if (decouple && getenv("LLAMA_UMA_STREAM_HOTFREQ") == nullptr) {
            throw std::runtime_error("uma stream: LLAMA_UMA_STREAM_DECOUPLE/ADAPT requires LLAMA_UMA_STREAM_HOTFREQ (initial slot seed)");
        }
        const uint32_t n_layer = model.hparams.n_layer();
        uma_stream->slots.assign((size_t) n_layer * LLAMA_UMA_STREAM_N_KIND, nullptr);
        uma_stream->slot_buf.resize((size_t) n_layer * LLAMA_UMA_STREAM_N_KIND); // M6: resizable slot buffers
        uma_stream->slot_host.assign((size_t) n_layer * LLAMA_UMA_STREAM_N_KIND, nullptr);
        uma_stream->slot_alloc.assign((size_t) n_layer * LLAMA_UMA_STREAM_N_KIND, 0);
        uma_stream->uds.assign((size_t) n_layer * LLAMA_UMA_STREAM_N_KIND, {});
        uma_stream->admit_uds.assign((size_t) n_layer, {});
        uma_stream->prefill_uds.assign((size_t) n_layer, {});
        uma_stream->lru.assign((size_t) n_layer, {});
        uma_stream->slot_of_expert.assign((size_t) n_layer, nullptr);
        uma_stream->topk.assign((size_t) n_layer, nullptr);
        uma_stream->ranked.assign((size_t) n_layer, {}); // M6: per-layer freq ranking for eager grow
        uma_stream->device_stage_buf.resize(LLAMA_UMA_STREAM_N_KIND);
        uma_stream->device_stage_host.assign(LLAMA_UMA_STREAM_N_KIND, nullptr);
        uma_stream->device_stage_bytes.assign(LLAMA_UMA_STREAM_N_KIND, 0);

        // one extra tensor per streaming layer for the decouple expert->slot table
        const size_t n_slabs_max = (size_t) k * (LLAMA_UMA_STREAM_N_KIND + 1);
        ggml_init_params ip = { ggml_tensor_overhead() * (n_slabs_max + 8), nullptr, /*.no_alloc =*/ true };
        uma_stream->meta_ctx.reset(ggml_init(ip));
        if (!uma_stream->meta_ctx) {
            throw std::runtime_error("uma stream: failed to init slot meta context");
        }
        ggml_context * mctx = uma_stream->meta_ctx.get();

        const size_t page = 16384; // Apple Silicon vm_page_size; newBufferWithBytesNoCopy needs a page-aligned base + length
        for (uint32_t il = 0; il < k; il++) {
            const auto & layer = model.layers[il];
            ggml_tensor * srcs[LLAMA_UMA_STREAM_N_KIND];
            srcs[LLAMA_UMA_STREAM_GATE] = layer.ffn_gate_exps;
            srcs[LLAMA_UMA_STREAM_UP]   = layer.ffn_up_exps;
            srcs[LLAMA_UMA_STREAM_DOWN] = layer.ffn_down_exps;
            bool layer_streams = false;
            for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
                ggml_tensor * src = srcs[kind];
                if (src == nullptr || model.uma_stream_slab_bytes(il, kind) == 0) {
                    continue;
                }
                const size_t i = (size_t) il * LLAMA_UMA_STREAM_N_KIND + kind;
                // slot tensor is [ne0, ne1, active S] (compressed expert dim)
                const int64_t slot_ne[GGML_MAX_DIMS] = { src->ne[0], src->ne[1], (int64_t) n_slots_initial, 1 };
                ggml_tensor * st = ggml_new_tensor(mctx, src->type, GGML_MAX_DIMS, slot_ne);
                if (st == nullptr) {
                    throw std::runtime_error("uma stream: failed to create a slot tensor");
                }
                char name[64];
                snprintf(name, sizeof(name), "uma.slot.%u.%d", il, kind);
                ggml_set_name(st, name);
                if (st->nb[2] != src->nb[2]) {
                    throw std::runtime_error(format("uma stream: slot stride %zu != source %zu (il=%u kind=%d)", (size_t) st->nb[2], (size_t) src->nb[2], il, kind));
                }

                // Allocate only the initial active window, not the controller ceiling.
                // A later grow releases this buffer and reallocates at the target S.
                const size_t slot_bytes = (size_t) n_slots_initial * (size_t) src->nb[2];
                ggml_backend_buffer_t buf; void * host; size_t alloc;
                if (!uma_stream_alloc_slot_buf(st, slot_bytes, &buf, &host, &alloc)) {
                    throw std::runtime_error(format("uma stream: slot buffer alloc failed il=%u kind=%d", il, kind));
                }
                uma_stream->slot_host[i]  = host;
                uma_stream->slot_alloc[i] = alloc;
                uma_stream->slot_buf[i].reset(buf);

                uma_stream->slots[i] = st;
                uma_stream->uds[i]   = { uma_stream.get(), (int) il, kind };
                if (device_slots_requested) {
                    uma_stream->device_stage_bytes[kind] = std::max(
                        uma_stream->device_stage_bytes[kind], (size_t) src->nb[2]);
                }
                layer_streams = true;
            }
            if (layer_streams) {
                uma_stream->admit_uds[il] = { uma_stream.get(), (int) il, false };
                uma_stream->prefill_uds[il] = { uma_stream.get(), (int) il, true };
                llama_uma_stream_layer_lru & L = uma_stream->lru[il];
                L.slot_of_expert.assign(n_expert, -1);
                L.expert_in_slot.assign(n_slots, -1);
                L.last_used.assign(n_slots, 0);
                L.pinned.assign(n_slots, 0);
                L.pin_protected.assign(n_slots, 0);
                L.newly_admitted.assign(n_slots, 0);

                // decouple mode: per-layer expert->slot table (I32 [1,n_expert,1,1]) on a
                // StorageModeShared no-rset wrap. The GPU gathers slot_ids from it via
                // ggml_get_rows (no forced-CPU op). Warm-start seeds it; 0 = sentinel slot.
                if (decouple) {
                    const size_t tbl_bytes = (size_t) n_expert * sizeof(int32_t);
                    void * thost = nullptr;
                    ggml_backend_buffer_t twrap = nullptr;
                    if (uma_stream_use_cuda_host) {
                        // CUDA: the buffer owns thost (freed via wraps.clear(); NOT host_bases)
                        twrap = ggml_backend_buft_alloc_buffer(uma_stream_cuda_host_buft, tbl_bytes);
                        if (twrap == nullptr) {
                            throw std::runtime_error(format("uma stream: table buffer alloc failed il=%u", il));
                        }
                        thost = ggml_backend_buffer_get_base(twrap);
                        memset(thost, 0, tbl_bytes); // non-resident experts -> sentinel slot 0
                        ggml_backend_buffer_set_usage(twrap, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                        uma_stream->wraps.emplace_back(twrap);
                    } else {
                        const size_t talloc = ((tbl_bytes + page - 1) / page) * page;
                        if (posix_memalign(&thost, page, talloc) != 0 || thost == nullptr) {
                            throw std::runtime_error("uma stream: table host allocation failed");
                        }
                        twrap = ((buffer_mapped_norset_t) uma_stream_wrap_fn)(gpu_dev, thost, talloc, tbl_bytes);
                        if (twrap == nullptr) {
                            free(thost);
                            throw std::runtime_error(format("uma stream: table wrap failed il=%u", il));
                        }
                        memset(thost, 0, talloc); // non-resident experts -> sentinel slot 0
                        ggml_backend_buffer_set_usage(twrap, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                        uma_stream->host_bases.push_back(thost); // Metal: posix_memalign'd (dtor free())
                        uma_stream->wraps.emplace_back(twrap);
                    }
                    const int64_t tbl_ne[GGML_MAX_DIMS] = { 1, (int64_t) n_expert, 1, 1 };
                    ggml_tensor * tbl = ggml_new_tensor(mctx, GGML_TYPE_I32, GGML_MAX_DIMS, tbl_ne);
                    if (tbl == nullptr) {
                        throw std::runtime_error("uma stream: failed to create expert table tensor");
                    }
                    char tname[64];
                    snprintf(tname, sizeof(tname), "uma.table.%u", il);
                    ggml_set_name(tbl, tname);
                    tbl->buffer = twrap;
                    tbl->data   = thost;
                    uma_stream->slot_of_expert[il] = tbl;
                }
            }
        }
        if (device_slots_requested) {
            typedef bool (*vmm_info_fn_t)(ggml_backend_buffer_t, size_t *, size_t *, size_t *, size_t *, size_t *);
            typedef bool (*vmm_stats_fn_t)(ggml_backend_buffer_t, uint64_t *, uint64_t *, uint64_t *, uint64_t *, void **);
            const vmm_info_fn_t info_fn = (vmm_info_fn_t) uma_stream_vmm_info_fn;
            const vmm_stats_fn_t stats_fn = (vmm_stats_fn_t) uma_stream_vmm_stats_fn;
            uint64_t mapped = 0, reserved = 0;
            size_t granularity = 0;
            uint64_t mappings = 0;
            for (const auto & owner : uma_stream->slot_buf) {
                if (!owner) { continue; }
                size_t active = 0, max = 0, bytes = 0, reserve = 0, gran = 0;
                if (!info_fn(owner.get(), &active, &max, &bytes, &reserve, &gran) ||
                    active != n_slots_initial || max != n_slots) {
                    throw std::runtime_error("uma stream: CUDA VMM slot-buffer invariant failed at setup");
                }
                mapped += bytes;
                reserved += reserve;
                granularity = gran;
                uint64_t map_ops = 0;
                if (!stats_fn(owner.get(), &map_ops, nullptr, nullptr, nullptr, nullptr)) {
                    throw std::runtime_error("uma stream: CUDA VMM slot-buffer stats unavailable at setup");
                }
                mappings += map_ops;
            }
            fprintf(stderr,
                "uma: segmented device slot backing: %llu actual mappings, %.1f MiB mapped / %.1f MiB VA reserved, %.1f MiB granularity\n",
                (unsigned long long) mappings, mapped / (1024.0 * 1024.0), reserved / (1024.0 * 1024.0),
                granularity / (1024.0 * 1024.0));
        }
        fprintf(stderr, "uma: stream slot pool built: %zu tables, S active/ceiling/expert %u/%u/%u (%s, GPU in-place), layers [0,%u)\n",
                uma_stream->wraps.size(), n_slots_initial, n_slots, n_expert,
                uma_stream_use_cuda_device ? "CUDA device VMM-segmented" :
                (uma_stream_use_cuda_host ? "CUDA pinned host" : "Metal StorageModeShared no-rset"), k);

        if (device_slots_requested) {
            for (int kind = 0; kind < LLAMA_UMA_STREAM_N_KIND; kind++) {
                const size_t bytes = uma_stream->device_stage_bytes[kind];
                if (bytes == 0) { continue; }
                ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(uma_stream_cuda_host_buft, bytes);
                if (buf == nullptr) {
                    throw std::runtime_error(format("uma stream: device staging alloc failed kind=%d", kind));
                }
                uma_stream->device_stage_host[kind] = ggml_backend_buffer_get_base(buf);
                uma_stream->device_stage_buf[kind].reset(buf);
            }
            fprintf(stderr, "uma: device-slot miss staging: pinned HOST slabs ready for async H2D\n");
        }

        // the slot pool is resident; the streamed experts were loaded only to
        // validate offsets + shape the slots. Discard their RAM FIRST (before the
        // warm-fill faults slot pages in) so the ~18 GB expert transient and the slot
        // pages are never both resident - large-S all-layers safety. Cold experts
        // stream from the fd on a miss (the footprint give-back).
        // Fix #2: under LLAMA_UMA_STREAM_LAZYLOAD the loader never read the experts,
        // so there is nothing resident to free - skip the no-op madvise (its report
        // would misleadingly count untouched virtual pages as "discarded").
        const char * lz = getenv("LLAMA_UMA_STREAM_LAZYLOAD");
        const bool lazy = lz != nullptr && lz[0] != '\0' && lz[0] != '0';
        if (lazy) {
            fprintf(stderr, "uma: stream lazy-load: streamed experts were never resident; slot pool is the sole copy\n");
        } else {
            model.uma_stream_free_excluded();
        }

        // seed the hot working set into the (now sole) resident slots + pin, preading
        // from the dup'd fd (the free above does not close it).
        uma_stream_warm_start(uma_stream.get(), model);

        // M6 give-back controller config (parsed once). Armed by LLAMA_UMA_STREAM_SMIN
        // (the knee = a hard floor). Requires DECOUPLE (the GPU-gather routing that makes
        // the active window respected). Drivers: SCHED = commanded (tok:S,...); CTRL via
        // LOWMIB/HIGHMIB watermarks on the M4 avail signal, STEP slots per move.
        if (const char * env_smin = getenv("LLAMA_UMA_STREAM_SMIN")) {
            if (!uma_stream->decouple) {
                throw std::runtime_error("uma stream: LLAMA_UMA_STREAM_SMIN (M6 resize) requires DECOUPLE/ADAPT");
            }
            char * end = nullptr;
            const long smin = strtol(env_smin, &end, 10);
            if (end == env_smin || *end != '\0' || smin < (long) n_expert_used ||
                smin > (long) n_slots_initial) {
                throw std::runtime_error(format("invalid LLAMA_UMA_STREAM_SMIN '%s' (want %u..initial S=%u)",
                                                env_smin, n_expert_used, n_slots_initial));
            }
            uma_resize_smin = (int32_t) smin;
        }
        if (const char * env_p = getenv("LLAMA_UMA_STREAM_RESIZE_PERIOD")) {
            const long p = strtol(env_p, nullptr, 10);
            if (p > 0) { uma_resize_period = (int32_t) p; }
        }
        if (const char * env_sched = getenv("LLAMA_UMA_STREAM_SCHED")) {
            const char * p = env_sched; // "tok:S,tok:S,..."
            while (*p) {
                char * e1 = nullptr;
                char * e2 = nullptr;
                const long tok = strtol(p, &e1, 10);
                if (e1 == p || *e1 != ':') { break; }
                const long s = strtol(e1 + 1, &e2, 10);
                if (e2 == e1 + 1) { break; }
                uma_resize_sched.emplace_back((int64_t) tok, (int32_t) s);
                p = (*e2 == ',') ? e2 + 1 : e2;
            }
            std::sort(uma_resize_sched.begin(), uma_resize_sched.end());
        }
        if (const char * env_lo = getenv("LLAMA_UMA_STREAM_LOWMIB")) {
            uma_resize_lowmib = (int32_t) strtol(env_lo, nullptr, 10);
        }
        if (const char * env_hi = getenv("LLAMA_UMA_STREAM_HIGHMIB")) {
            uma_resize_highmib = (int32_t) strtol(env_hi, nullptr, 10);
        }
        if (const char * env_step = getenv("LLAMA_UMA_STREAM_STEP")) {
            const long st = strtol(env_step, nullptr, 10);
            if (st > 0) { uma_resize_step = (int32_t) st; }
        }
        if (const char * env_ctl = getenv("LLAMA_UMA_STREAM_CONTROL")) {
            uma_control_path = env_ctl;
        }
        if (const char * env_tel = getenv("LLAMA_UMA_STREAM_TELEMETRY")) {
            uma_telemetry_path = env_tel;
        }
        if (uma_resize_smin >= 0) {
            fprintf(stderr, "uma: M6 controller armed: S in [%d,%u], period %d tok, sched=%zu pts, CTRL lo/hi=%d/%d MiB step %d\n",
                    uma_resize_smin, n_slots, uma_resize_period, uma_resize_sched.size(),
                    uma_resize_lowmib, uma_resize_highmib, uma_resize_step);
            if (!uma_control_path.empty() || !uma_telemetry_path.empty()) {
                fprintf(stderr, "uma: M7.1 arbiter channel: control='%s' telemetry='%s'\n",
                        uma_control_path.c_str(), uma_telemetry_path.c_str());
            }
        } else if (!uma_resize_sched.empty() || uma_resize_lowmib > 0) {
            fprintf(stderr, "uma: WARNING M6 driver (SCHED/CTRL) set but LLAMA_UMA_STREAM_SMIN unset; controller disabled\n");
        }
    }

    // register the slot + table bufts on the CURRENT sched (re-run on every sched
    // (re)creation). slot_buf are the resizable slot buffers; wraps are the fixed tables.
    for (const auto & buf : uma_stream->slot_buf) {
        if (!buf) { continue; }
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        ggml_backend_sched_allow_weights_buft(sched.get(), gpu_backend, buft);
        if (!uma_stream->device_slots) {
            ggml_backend_sched_allow_weights_buft(sched.get(), backend_cpu, buft);
        }
    }
    for (const auto & wrap : uma_stream->wraps) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(wrap.get());
        ggml_backend_sched_allow_weights_buft(sched.get(), gpu_backend, buft);
        ggml_backend_sched_allow_weights_buft(sched.get(), backend_cpu, buft);
    }
}

void llama_context::uma_allow_weights_bufts() {
    uma_stream_setup();
    if (!uma_router || !uma_router->placement_active()) {
        return;
    }
    // zero-copy staged reads (C1): distinct CPU buffers holding designated
    // std-layout expert weights -> (tensors to retarget, max tensor bytes)
    std::map<ggml_backend_buffer_t, std::pair<std::vector<ggml_tensor *>, size_t>> std_wrap_targets;
    for (uint32_t il = 0; il < uma_router->n_cpu_layers; il++) {
        const auto & layer = model.layers[il];
        for (ggml_tensor * t : { layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps, layer.ffn_gate_up_exps }) {
            if (t == nullptr) {
                continue;
            }
            ggml_backend_buffer_t buf = t->buffer;
            if (buf == nullptr) {
                throw std::runtime_error(format("uma: expert tensor %s has no buffer", t->name));
            }
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            {
                // std/repack layouts: the designated weights are CPU-resident
                // by design. std = plain CPU (is_host, staging-eligible),
                // repack = CPU extra buft (is_host false, CPU-only readable).
                // No registration and no gpu pin: the CPU backend owns them,
                // batches take the default sched path (std: op_offload
                // staging; repack: CPU gemv, the priced collapse tier).
                ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
                const bool cpu_resident = buft_dev == nullptr || ggml_backend_dev_type(buft_dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
                if (uma_router->layout != LLAMA_UMA_LAYOUT_DEFAULT && cpu_resident) {
                    const bool want_std = uma_router->layout == LLAMA_UMA_LAYOUT_STD;
                    if (want_std != ggml_backend_buft_is_host(buft)) {
                        throw std::runtime_error(format("uma: layout %s expert weights for %s landed in %s - load-time layout did not stick", want_std ? "std" : "repack", t->name, ggml_backend_buft_name(buft)));
                    }
                    // std: collect for the zero-copy wrap after the scan
                    // (uma_wrap_std_buffers pins batches and registers the
                    // wrapped buft there, or falls back to the staged pin).
                    // repack stays unpinned: CPU-only readable by design.
                    if (want_std) {
                        auto & tgt = std_wrap_targets[buf];
                        tgt.first.push_back(t);
                        tgt.second = std::max(tgt.second, ggml_nbytes(t));
                    }
                    if (uma_bufts_logged.insert(buft).second) {
                        fprintf(stderr, "uma: layout %s verified: expert weights in %s\n", want_std ? "std" : "repack", ggml_backend_buft_name(buft));
                    }
                    continue;
                }
                if (uma_router->layout == LLAMA_UMA_LAYOUT_REPACK) {
                    throw std::runtime_error(format("uma: layout repack expert weights for %s landed in %s - load-time layout did not stick", t->name, ggml_backend_buft_name(buft)));
                }
            }
            if (ggml_backend_buft_is_host(buft)) {
                // CUDA route: the load-time injection put these weights in the
                // device's pinned host buft. The CPU reads them natively; the
                // owning GPU backend is registered for in-place reads and pins
                // n_tokens > 1 batches (see graph_get_cb). A plain CPU buft
                // here means the load-time placement did NOT stick (pinned
                // alloc fell back, or no CUDA device) - refuse rather than
                // silently measure a different mechanism.
                ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
                if (buft_dev == nullptr || ggml_backend_dev_type(buft_dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    throw std::runtime_error(format("uma: cpu-static expert weights for %s landed in %s - load-time placement did not stick", t->name, ggml_backend_buft_name(buft)));
                }
                ggml_backend_t dev_backend = nullptr;
                for (ggml_backend_t b : backend_ptrs) {
                    if (ggml_backend_get_device(b) == buft_dev) {
                        dev_backend = b;
                        break;
                    }
                }
                if (dev_backend == nullptr) {
                    throw std::runtime_error(format("uma: no backend for device %s owning weights buft %s (device excluded from this context, e.g. via --device?)", ggml_backend_dev_name(buft_dev), ggml_backend_buft_name(buft)));
                }
                ggml_backend_sched_allow_weights_buft(sched.get(), dev_backend, buft);
                uma_router->gpu_pin_backend = dev_backend;
                if (uma_bufts_logged.insert(buft).second) {
                    fprintf(stderr, "uma: weights buft %s registered for %s in-place reads\n", ggml_backend_buft_name(buft), ggml_backend_name(dev_backend));
                }
                continue;
            }
            const char * buft_name = ggml_backend_buft_name(buft);
            if (!uma_buft_host_visible(buft_name)) {
                throw std::runtime_error(format("uma: cpu-static needs host-visible expert weights, refusing buffer type %s for %s", buft_name, t->name));
            }
            ggml_backend_sched_allow_weights_buft(sched.get(), backend_cpu, buft);
            // zero-copy re-run path (sched recreation or a second context on
            // the same model): a std wrap presents as a GPU-device mapped
            // buft - re-register the GPU engine and the pin on this sched
            if (uma_router->layout == LLAMA_UMA_LAYOUT_STD) {
                ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
                if (buft_dev != nullptr &&
                    (ggml_backend_dev_type(buft_dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                     ggml_backend_dev_type(buft_dev) == GGML_BACKEND_DEVICE_TYPE_IGPU)) {
                    for (ggml_backend_t b : backend_ptrs) {
                        if (ggml_backend_get_device(b) == buft_dev) {
                            ggml_backend_sched_allow_weights_buft(sched.get(), b, buft);
                            uma_router->gpu_pin_backend = b;
                            break;
                        }
                    }
                }
            }
            if (uma_bufts_logged.insert(buft).second) {
                fprintf(stderr, "uma: weights buft %s registered for CPU in-place reads\n", buft_name);
            }
        }
    }
    uma_wrap_std_buffers(std_wrap_targets);
}

void llama_context::uma_wrap_std_buffers(const std::map<ggml_backend_buffer_t, std::pair<std::vector<ggml_tensor *>, size_t>> & targets) {
    if (targets.empty()) {
        return;
    }
    // the batch engine: first GPU/IGPU backend, same choice the staged path
    // pinned to
    ggml_backend_t gpu_backend = nullptr;
    for (ggml_backend_t b : backend_ptrs) {
        ggml_backend_dev_t d = ggml_backend_get_device(b);
        if (d == nullptr) {
            continue;
        }
        const enum ggml_backend_dev_type dt = ggml_backend_dev_type(d);
        if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_backend = b;
            break;
        }
    }
    if (gpu_backend == nullptr) {
        // no GPU in this context: the weights stay plain CPU tensors and the
        // CPU backend computes on them natively (nothing to wrap or pin)
        return;
    }
    // fork-private Metal entry point, reached via proc address so this file
    // needs no Metal includes; absent on other backends
    typedef ggml_backend_buffer_t (*buffer_mapped_norset_t)(ggml_backend_dev_t, void *, size_t, size_t);
    ggml_backend_dev_t gpu_dev = ggml_backend_get_device(gpu_backend);
    ggml_backend_reg_t gpu_reg = gpu_dev == nullptr ? nullptr : ggml_backend_dev_backend_reg(gpu_dev);
    buffer_mapped_norset_t wrap_fn = gpu_reg == nullptr ? nullptr :
        (buffer_mapped_norset_t) ggml_backend_reg_get_proc_address(gpu_reg, "ggml_backend_metal_device_buffer_mapped_norset");
    // capacity-mode gate (C3 finding): the wrap is unwired, but around
    // submission the driver must make the WHOLE mapped span resident next to
    // the wired working set. Under a k-forcing cap the budget slack is
    // smaller than the excluded bytes by construction, so full-span
    // zero-copy pp can never fit and llama_decode fails closed with res=-3
    // (clean refusal - measured at C16=12724 MiB, session 20260731-183946;
    // uncapped the same path is free, results/wo-c2-zerocopy-20260731.md).
    // Gate on budget headroom: zero-copy only when free >= span + slack
    // (slack also covers KV/compute buffers not yet allocated at this
    // point); otherwise fall back to the staged path - the priced capacity
    // tier, C16-record behavior. LLAMA_UMA_ZEROCOPY=0/1 forces the decision
    // for ablations (forcing 1 under a tight cap reproduces the refusal -
    // measurement use only).
    bool do_wrap = wrap_fn != nullptr;
    if (do_wrap) {
        const char * env_zc = getenv("LLAMA_UMA_ZEROCOPY");
        if (env_zc != nullptr && env_zc[0] != '\0') {
            do_wrap = env_zc[0] == '1';
            fprintf(stderr, "uma: layout std: zero-copy FORCED %s (LLAMA_UMA_ZEROCOPY)\n", do_wrap ? "on" : "off");
        } else {
            size_t mapped_total = 0;
            for (const auto & [buf, tgt] : targets) {
                GGML_UNUSED(tgt);
                mapped_total += ggml_backend_buffer_get_size(buf);
            }
            const size_t slack = (size_t) 2048 * 1024 * 1024;
            size_t mem_free = 0, mem_total = 0;
            ggml_backend_dev_memory(gpu_dev, &mem_free, &mem_total);
            if (mem_free > mem_total) {
                // Metal reports free = budget - allocated, which underflows
                // when allocations exceed a lowered budget - treat as no room
                mem_free = 0;
            }
            do_wrap = mem_free >= mapped_total + slack;
            fprintf(stderr, "uma: layout std: zero-copy gate: budget free %zu MiB vs span %zu + slack 2048 MiB -> %s\n",
                    mem_free / (1024*1024), mapped_total / (1024*1024), do_wrap ? "zero-copy" : "staged");
        }
    }
    if (!do_wrap) {
        // staged path (pre-C1 mechanism): batches are pinned to the GPU
        // backend and the sched copies used experts into each split
        if (uma_router->gpu_pin_backend == nullptr) {
            uma_router->gpu_pin_backend = gpu_backend;
            fprintf(stderr, "uma: layout std: batches pinned to %s (staged in-split reads)\n", ggml_backend_name(gpu_backend));
        }
        return;
    }
    for (const auto & [buf, tgt] : targets) {
        const std::vector<ggml_tensor *> & tensors = tgt.first;
        void * base = ggml_backend_buffer_get_base(buf);
        const size_t size = ggml_backend_buffer_get_size(buf);
        for (ggml_tensor * t : tensors) {
            const char * p = (const char *) t->data;
            if (p < (const char *) base || p + ggml_nbytes(t) > (const char *) base + size) {
                throw std::runtime_error(format("uma: expert tensor %s lies outside its buffer - refusing zero-copy wrap", t->name));
            }
        }
        ggml_backend_buffer_t wrap = wrap_fn(gpu_dev, base, size, tgt.second);
        if (wrap == nullptr) {
            throw std::runtime_error("uma: zero-copy wrap of a CPU expert buffer failed");
        }
        ggml_backend_buffer_set_usage(wrap, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // lifetime: the wrap only views the CPU buffer's pages; the model
        // holds it so it dies with the weights (before them, see impl)
        model.uma_hold_wrap_buffer(wrap);
        for (ggml_tensor * t : tensors) {
            t->buffer = wrap; // t->data unchanged - same pages, now GPU-bindable
        }
        ggml_backend_buffer_type_t wrap_buft = ggml_backend_buffer_get_type(wrap);
        ggml_backend_sched_allow_weights_buft(sched.get(), gpu_backend, wrap_buft); // pp reads in place
        ggml_backend_sched_allow_weights_buft(sched.get(), backend_cpu, wrap_buft); // decode reads the same pages
        fprintf(stderr, "uma: layout std: zero-copy wrap: %zu expert tensors in %s (%.1f MiB, no rset, in-place reads)\n",
                tensors.size(), ggml_backend_buft_name(wrap_buft), size / (1024.0 * 1024.0));
    }
    uma_router->gpu_pin_backend = gpu_backend;
    fprintf(stderr, "uma: layout std: batches pinned to %s (zero-copy in-place reads)\n", ggml_backend_name(gpu_backend));
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }

        // uma-moe: route the designated layers' expert matmuls per pass. The
        // weights are read in place by both engines (registered at reserve
        // time): single-token decode pins them to the CPU backend, and on the
        // CUDA route batches are pinned to the GPU backend - with
        // host-resident weights the default assignment is a batch-size
        // heuristic (CPU below 32 tokens, op_offload above), and placement
        // must stay policy-owned at every batch size. On Metal the weights
        // are device-resident, so batches need no pin (default is all-GPU).
        // note: for models with per-expert scales/biases the cb'd tensor is
        // the trailing mul/add_id, not the mul_mat_id itself; cpu-static
        // targets plain-MoE graphs (Qwen3-MoE class) for now
        // expert-id channel: keep the on-device topk readable and cache its
        // pointer per layer; refreshed at every (re)build, read post-sync
        if (uma_router && il >= 0 && uma_router->observe_experts && strcmp(name, "ffn_moe_topk") == 0) {
            ggml_set_output(cur);
            uma_router->observe_experts_cache(il, cur);
        }
        // decouple miss detection (Part 2): cache the streaming layer's topk (marked
        // output) so synchronize() can count non-resident selections post-sync. uma_router
        // is null in streaming mode, so this is the streaming-path equivalent.
        if (uma_stream && uma_stream->decouple && il >= 0 && strcmp(name, "ffn_moe_topk") == 0 &&
            uma_stream->streams_layer(il)) {
            ggml_set_output(cur);
            uma_stream->topk[il] = cur;
        }

        // Streaming slot/table tensors are host-visible weights. CUDA intentionally does
        // not advertise CUDA_Host as a generally supported device buft, so the default
        // batch-1 heuristic can place both the table gather and expert matmuls on CPU.
        // The scheduler allowlist established by uma_stream_setup proves these exact bufts
        // GPU-readable; explicitly pin their graph nodes to that GPU device.
        if (uma_stream && il >= 0 && uma_stream->streams_layer(il) &&
            (strcmp(name, "ffn_moe_decouple_gather") == 0 ||
             strcmp(name, "ffn_moe_gate")           == 0 ||
             strcmp(name, "ffn_moe_up")             == 0 ||
             strcmp(name, "ffn_moe_gate_up")        == 0 ||
             strcmp(name, "ffn_moe_down")           == 0)) {
            for (const auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == uma_stream_gpu_dev &&
                    ggml_backend_supports_op(backend.get(), cur)) {
                    ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                    break;
                }
            }
        }

        // uma-moe fork M5 S1.1.1: the streaming fill op (GGML_OP_CUSTOM, CPU-only)
        // is forced onto the CPU backend. Its output aliases a Metal weights slot,
        // so the default from_buffer assignment would abort (no backend supports
        // both that buft and CUSTOM); a forced assignment skips that path.
        if (il >= 0 && (strcmp(name, "ffn_moe_stream_admit") == 0 ||
                        strcmp(name, "ffn_moe_stream_gate")  == 0 ||
                        strcmp(name, "ffn_moe_stream_up")    == 0 ||
                        strcmp(name, "ffn_moe_stream_down")  == 0)) {
            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
        }

        if (uma_router && il >= 0 && uma_router->placement_active() && (uint32_t) il < uma_router->n_cpu_layers) {
            if (strcmp(name, "ffn_moe_gate")    == 0 ||
                strcmp(name, "ffn_moe_up")      == 0 ||
                strcmp(name, "ffn_moe_gate_up") == 0 ||
                strcmp(name, "ffn_moe_down")    == 0) {
                if (uma_router->layer_on_cpu(il, ubatch.n_tokens)) {
                    if (ggml_backend_supports_op(backend_cpu, cur)) {
                        ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend_cpu);
                    }
                } else if (uma_router->gpu_pin_backend != nullptr) {
                    if (ggml_backend_supports_op(uma_router->gpu_pin_backend, cur)) {
                        ggml_backend_sched_set_tensor_backend(sched.get(), cur, uma_router->gpu_pin_backend);
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.n_tensors != mbuf.n_tensors || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
            }
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        model->hparams.n_layer_nextn == 0) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
