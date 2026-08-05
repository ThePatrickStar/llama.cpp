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
#include <sys/mman.h>
#include <vector>

struct llama_model;
struct llama_uma_stream_state;

// process phys_footprint in MiB (TASK_VM_INFO on Darwin, 0 elsewhere). Defined in
// llama-model.cpp; read at context teardown for the supply-curve steady state.
size_t llama_uma_phys_footprint_mib();

// M6 controller: reclaimable-available system memory in MiB = (free+inactive+
// speculative)*page via host_statistics64 (Darwin, 0 elsewhere). Allocation-free.
// The M4 budget signal, read in-process by the give-back controller. Mirrors
// scripts/mem_sensor.c. Defined in llama-model.cpp.
size_t llama_uma_avail_reclaim_mib();

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
    uint32_t n_slots      = 0;  // S ceiling: max slots per (layer,kind) the controller grows to.
    uint32_t n_slots_active = 0; // M6: current resident slots. The slot BUFFER is this size and is
                                 // REALLOCATED on resize (a live Metal buffer pins its pages, so an
                                 // in-place madvise cannot shed phys_footprint - only release does).
    uint32_t n_expert     = 0;
    uint64_t n_miss       = 0;  // expert preads (slot misses) over the context lifetime
    uint64_t n_read       = 0;  // total expert-reads (admits) over the context lifetime
    bool     decouple     = false; // LLAMA_UMA_STREAM_DECOUPLE: GPU-gather decode routing (Part 1)
    bool     adapt        = false; // LLAMA_UMA_STREAM_ADAPT: online resident-set maintenance (Part 2 Step 3)

    // M6 give-back controller telemetry + eager-grow support
    uint64_t n_resizes    = 0;  // runtime S-resize events over the context lifetime
    uint64_t n_distress   = 0;  // times a shed target below the knee was clamped (M7 signal)
    uint32_t s_min_active = 0;  // smallest n_slots_active reached (0 = unset)
    uint32_t s_max_active = 0;  // largest n_slots_active reached
    // per-streaming-layer expert ids ranked hottest-first (from the warm-start freq
    // dump); grow re-warms newly-active slots from this ranking. ranked[il] size n_expert.
    std::vector<std::vector<int32_t>> ranked;

    // index = il*3 + kind. slots[i]==nullptr means (il,kind) is not streaming.
    std::vector<ggml_tensor *>                slots;
    std::vector<llama_uma_stream_fill_ud>     uds;        // per (il,kind)
    std::vector<llama_uma_stream_admit_ud>    admit_uds;  // per il
    std::vector<llama_uma_stream_layer_lru>   lru;        // per il
    // decouple mode: per-layer expert->slot table, I32 [1,n_expert,1,1], Metal
    // StorageModeShared (GPU-readable, CPU-writable). The GPU gathers slot_ids from
    // it via ggml_get_rows instead of a forced-CPU admit op. slot_of_expert[il]==nullptr
    // if not built. Static (warm-start-seeded) in Part 1; background-updated in Part 2.
    std::vector<ggml_tensor *>                slot_of_expert; // per il
    // decouple miss detection (Part 2): per-layer cached topk (selected_experts,
    // marked output) read post-sync to count non-resident selections. No per-layer
    // CPU op (would reintroduce the sync tax) - one D2H per layer per token, post-sync.
    std::vector<ggml_tensor *>                topk;           // per il

    // context-lifetime resources. The Metal wraps only view host_bases (noCopy),
    // so they are released BEFORE the pages are freed; meta_ctx holds only tensor
    // structs (no data) and can go last. wraps/host_bases hold the FIXED table
    // buffers; the resizable slot buffers live in slot_buf/slot_host (per il*3+kind).
    ggml_context_ptr                     meta_ctx;
    std::vector<ggml_backend_buffer_ptr> wraps;
    std::vector<void *>                  host_bases;

    // M6: per-(il,kind) resizable slot buffers. mmap'd (not posix_memalign) so a
    // resize's munmap returns pages to the OS UNCONDITIONALLY - no malloc large-cache
    // retention, no reusable-page under-charge - giving a clean phys_footprint.
    std::vector<ggml_backend_buffer_ptr> slot_buf;   // release before slot_host is unmapped
    std::vector<void *>                  slot_host;
    std::vector<size_t>                  slot_alloc;  // mmap length per (il,kind), for munmap
    uint32_t pin_h = 0;   // warm-start hot-pin count; reseed reuses it, clamped to the new S

    ~llama_uma_stream_state() {
        slot_buf.clear();  // release resizable slot buffers before their hosts (Metal wrap / CUDA cudaFreeHost)
        wraps.clear();     // release the fixed table buffers
        for (size_t i = 0; i < slot_host.size(); i++) {
            // Metal: mmap'd host (alloc > 0) -> munmap. CUDA: the buffer owns the host
            // (alloc == 0, freed via slot_buf.clear() above) -> nothing to munmap.
            if (slot_host[i] && slot_alloc[i] > 0) {
                munmap(slot_host[i], slot_alloc[i]);
            }
        }
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
    ggml_tensor * expert_table(int il) const {
        return (size_t) il < slot_of_expert.size() ? slot_of_expert[(size_t) il] : nullptr;
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
