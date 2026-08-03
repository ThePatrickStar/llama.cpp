#pragma once

// uma-moe fork M5 S1.1.1: expert residency STREAMING - a resident slot pool the
// GPU reads in place while the CPU preads the selected experts into it on demand.
// S1.1.1 uses full-size slots (slot_id == expert_id, no remap) to prove the
// pread-fill and the CPU-fill -> GPU-read sync before the S1.1.2 compression.
// The slot tensors and fill userdata are context-lifetime persistent (NOT in the
// per-build graph ctx): on graph reuse build_moe_ffn is not re-run, so the reused
// custom-op node keeps referencing them.

#include "ggml.h"
#include "ggml-cpp.h"

#include <cstdlib>
#include <vector>

struct llama_model;

// kind index == llama_uma_stream_kind (gate/up/down) in llama-model.h; there are 3.
#define LLAMA_UMA_STREAM_N_KIND_LOCAL 3

// per-(layer, kind) fill op userdata; context-persistent, stable address
struct llama_uma_stream_fill_ud {
    const llama_model * model = nullptr;
    int                 il    = 0;
    int                 kind  = 0;
};

struct llama_uma_stream_state {
    const llama_model * model = nullptr;

    // index = il*3 + kind. slots[i]==nullptr means (il,kind) is not streaming.
    std::vector<ggml_tensor *>            slots;
    std::vector<llama_uma_stream_fill_ud> uds;

    // context-lifetime resources. The Metal wraps only view host_bases (noCopy),
    // so they are released BEFORE the pages are freed; meta_ctx holds only tensor
    // structs (no data) and can go last.
    ggml_context_ptr                     meta_ctx;
    std::vector<ggml_backend_buffer_ptr> wraps;
    std::vector<void *>                  host_bases;

    ~llama_uma_stream_state() {
        wraps.clear();
        for (void * p : host_bases) {
            free(p);
        }
    }

    bool streams(int il, int kind) const {
        const size_t i = (size_t) il * LLAMA_UMA_STREAM_N_KIND_LOCAL + (size_t) kind;
        return i < slots.size() && slots[i] != nullptr;
    }
    ggml_tensor * slot(int il, int kind) const {
        return slots[(size_t) il * LLAMA_UMA_STREAM_N_KIND_LOCAL + (size_t) kind];
    }
    // read-only userdata; fill only reads model/il/kind
    void * ud(int il, int kind) const {
        return (void *) &uds[(size_t) il * LLAMA_UMA_STREAM_N_KIND_LOCAL + (size_t) kind];
    }
};

// ggml_custom_op_t (forced onto the CPU backend by the graph cb): pread each
// selected expert (dst->src[1], I32) into its full-size slot at dst->data +
// expert_id*slab. n_tasks==1, so only ith==0 does work.
void llama_uma_stream_fill(ggml_tensor * dst, int ith, int nth, void * userdata);
