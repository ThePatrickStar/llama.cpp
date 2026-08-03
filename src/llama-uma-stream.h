#pragma once

// uma-moe fork M5 Serial-S1: expert residency STREAMING - a resident slot pool
// the GPU reads in place while the CPU preads the selected experts into it on
// demand. S1.1.1 proved the pread-fill + the CPU-fill -> GPU-read sync with
// full-size slots (slot_id == expert_id). S1.1.2 COMPRESSES to S < n_expert
// slots with a per-layer LRU slot table (expert -> slot): an admit op assigns
// each token's selected experts to slots (evicting the LRU cold slot on a miss),
// preads the newly-admitted experts, and emits slot_ids so the matmul routes to
// slots. THIS is where the footprint saving lands (serial ~25% at S~=96).
// Slot tensors, LRU tables, and op userdata are context-lifetime persistent (NOT
// in the per-build graph ctx): on graph reuse build_moe_ffn is not re-run, so the
// reused custom-op nodes keep referencing them; the LRU state evolves per token.

#include "ggml.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

struct llama_model;
struct llama_uma_stream_state;

// process phys_footprint in MiB (TASK_VM_INFO on Darwin, 0 elsewhere). Defined in
// llama-model.cpp; read at context teardown for the supply-curve steady state.
size_t llama_uma_phys_footprint_mib();

// per-(layer, kind) fill op userdata; context-persistent, stable address
struct llama_uma_stream_fill_ud {
    llama_uma_stream_state * state = nullptr;
    int                      il    = 0;
    int                      kind  = 0;
};

// per-layer admit op userdata (shared across the layer's 3 kinds)
struct llama_uma_stream_admit_ud {
    llama_uma_stream_state * state = nullptr;
    int                      il    = 0;
};

// per-layer LRU slot table. slot s holds expert_in_slot[s] (-1 = empty); expert e
// is resident in slot_of_expert[e] (-1 = not resident). Eviction picks the slot
// with the smallest last_used that is NOT pinned this pass (in-batch experts are
// pinned so a batch never evicts an expert it still needs).
struct llama_uma_stream_layer_lru {
    std::vector<int32_t>  slot_of_expert; // n_expert
    std::vector<int32_t>  expert_in_slot; // n_slots
    std::vector<uint64_t> last_used;      // n_slots
    std::vector<uint64_t> pinned;         // n_slots (== pass when pinned this pass)
    std::vector<uint8_t>  pin_protected;  // n_slots (1 = hot/warm-start, never an eviction victim)
    std::vector<int32_t>  newly_admitted; // experts admitted this pass (<= n_slots)
    uint32_t n_newly = 0;
    uint64_t tick    = 0;                 // monotonic LRU clock
    uint64_t pass    = 0;                 // monotonic admit-pass counter
};

struct llama_uma_stream_state {
    const llama_model * model = nullptr;
    uint32_t n_slots      = 0;  // S (slots per layer,kind); == n_expert => no compression
    uint32_t n_expert     = 0;
    uint64_t n_miss       = 0;  // expert preads (slot misses) over the context lifetime
    uint64_t n_read       = 0;  // total expert-reads (admits) over the context lifetime

    // index = il*3 + kind. slots[i]==nullptr means (il,kind) is not streaming.
    std::vector<ggml_tensor *>                slots;
    std::vector<llama_uma_stream_fill_ud>     uds;        // per (il,kind)
    std::vector<llama_uma_stream_admit_ud>    admit_uds;  // per il
    std::vector<llama_uma_stream_layer_lru>   lru;        // per il

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
        const size_t i = (size_t) il * 3 + (size_t) kind;
        return i < slots.size() && slots[i] != nullptr;
    }
    // true if any kind of layer il streams (=> the layer has an admit op)
    bool streams_layer(int il) const {
        return streams(il, 0) || streams(il, 1) || streams(il, 2);
    }
    ggml_tensor * slot(int il, int kind) const {
        return slots[(size_t) il * 3 + (size_t) kind];
    }
    void * ud(int il, int kind) const {
        return (void *) &uds[(size_t) il * 3 + (size_t) kind];
    }
    void * admit_ud(int il) const {
        return (void *) &admit_uds[(size_t) il];
    }
};

// ggml_custom_4d op (forced onto CPU by the graph cb): reads selected_experts
// (dst->src[0], I32 [n_expert_used, n_tokens]), assigns each selected expert a
// slot via the layer LRU (admitting misses, recording them in lru[il]), and
// writes slot_ids into dst (same shape). Deterministic given the token sequence.
void llama_uma_stream_admit(ggml_tensor * dst, int ith, int nth, void * userdata);

// ggml_custom_inplace op (forced onto CPU): preads the experts admitted THIS pass
// (lru[il].newly_admitted) into their slots (dst->data + slot*slab). dst->src[1]
// is slot_ids (a dependency anchor so this runs after admit). Output aliases the
// slot tensor so the matmul weight depends on the fill = the CPU->GPU sync.
void llama_uma_stream_fill(ggml_tensor * dst, int ith, int nth, void * userdata);
