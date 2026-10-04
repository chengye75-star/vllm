# NVFP4 KV cache on consumer Blackwell (sm_120 / sm_121) — tested fork

Branch: `nvfp4-kv-sm120-tested` — a rebase of [vllm-project/vllm#46329](https://github.com/vllm-project/vllm/pull/46329) onto `main` @ `155d23cb008f` (14/14 commits, no conflicts), **plus build/runtime fixes found while testing on 2× RTX 5060 Ti 16G**.

If you're here because `--kv-cache-dtype nvfp4` explodes on your RTX 50xx / GB10 box: upstream stock vLLM rejects it (FlashInfer trtllm-gen path requires SM100 datacenter). This branch restores the FA2 decode route for NVFP4 KV on consumer Blackwell.

## Verified results (2× 5060 Ti 16G, TP2, RadixArk/Qwen3.8-27B-NVFP4, vLLM 0.30.0 + FlashInfer 0.7.0.post1)

### Quality
| Check | Result |
|---|---|
| Needle-in-haystack 106K & 133K ctx, 4 depths | ✅ 8/8 |
| Prefix caching, 106K repeat | ✅ 72.5s → 6.8s (10.7×) |
| Greedy determinism (4× identical) | ✅ byte-identical |
| Tool calls (auto / required / follow-up / plain) | ✅ 4/4 with `--tool-call-parser qwen3_coder` |
| Vision (multimodal image QA, tower loaded) | ✅ reads rendered text, 12s first request |

### Single-stream decode (temp=0, 200-token probes)
| Config | count | recall | list | creative | KV pool |
|---|---|---|---|---|---|
| nvfp4 KV, no MTP | 39.3 | 39.0 | 39.3 | 39.3 | 394,545 |
| + MTP K=1 | 56.0 | 55.5 | 50.6 | 49.7 | 310,126 |
| + MTP K=2 | 75.6 | 74.6 | 60.1 | 55.1 | 281,538 |
| + MTP K=3 | **93.7** | **90.2** | 66.1 | 56.8 | 269,117 |
| fp8_e4m3 KV (no MTP) | — | — | — | — | 209,465 |
| vision tower + nvfp4 (no MTP) | — | — | — | — | 225,000 |

Accept length scales 2.00 → 3.00 → 4.00 with K; MTP gain tracks content predictability (low-entropy text scales almost linearly, creative saturates by K=2). All numbers vs. an SGLang 0.5.21 baseline on the same box: engines match within ±1.3% without MTP.

## Required launch flags (three real traps found in testing)

```bash
vllm serve <nvfp4-model> --kv-cache-dtype nvfp4 \
  --kernel-config '{"enable_flashinfer_autotune": false}' \
  --disable-custom-all-reduce \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --max-model-len 150000   # see KV budget note
```

1. **`--kernel-config '{"enable_flashinfer_autotune": false}'` is mandatory for MTP + nvfp4 KV.** With FlashInfer autotune enabled, its benchmarking can land inside CUDA graph capture → `CUDA error: operation not permitted when stream is capturing` at startup. Non-deterministic (timing-dependent): K=1 booted once out of three tries. Disabling it uses the cached-tactic path and booted 4/4 green afterwards. Worth an upstream fix (defer autotune out of capture).
2. **`--disable-custom-all-reduce` on non-standard multi-GPU topologies.** vLLM enables its custom all-reduce by default (`Using ['CUSTOM', 'PYNCCL']`). On this box (PCIe ACS override enabled, patched open GPU kernel modules for consumer P2P), TP workers died *silently mid-inference* — the failure surface is wild peer writes through CUDA IPC on a topology the custom AR assumes is stock. Downstream symptoms look nothing like a comms bug (random heap corruption / interpreter segfaults in unrelated components). Same root cause independently confirmed on the SGLang side of this box (`--disable-custom-all-reduce` is mandatory there too; NCCL P2P path is stable). Cost is ~nil on 16G consumer cards; measured NCCL(P2P+PHB) was even slightly faster than custom AR.
3. **KV budget with MTP + parsers.** MTP draft weights + `--tool-call-parser`/`--reasoning-parser` squeeze the KV pool; at `--max-model-len 196000` startup was rejected (`3.42 GiB KV needed > 2.78 GiB available`). Either lower max-model-len (~150K) or raise `--gpu-memory-utilization`. Note `qwen3_5_mtp` method name is deprecated → use `mtp` on newer vLLM.

## Building from this branch (source build gotchas, all hit personally)

- **Host compiler**: CUDA 13.1 headers do not compile with g++-15 (Ubuntu 25.10+ default). `apt install g++-13` and build with
  `CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=/usr/bin/g++-13`.
- The NVFP4 KV write kernel needs `arch=compute_120a,code=sm_120a` (`cvt.e2m1x2.f32` is an sm_120**a** instruction; plain `sm_120` is rejected by ptxas). The branch's CMake already targets it correctly; this matters if you hand-compile.
- **nvcc/cudafe++/ptxas/cc1plus segfault randomly under high `-j`** on this toolchain combo. Workaround that produced 406/406 objects: `ninja -k 0 -j 8` in a retry loop (incremental — crashed files get recompiled next round). Don't panic on first failure.
- `uv`'s editable-install metadata hook SIGSEGVs on vllm's pyproject here; plain `pip install --no-build-isolation --no-deps -e .` works.
- Python 3.12, `flashinfer-python==0.7.0.post1` (from the flashinfer.ai wheel index; PyPI has no matching `flashinfer-cubin`).

## Known engine-level issues seen during testing (possibly box-specific)
- `fp8_e4m3` KV + torch.compile: worker crashed with an interpreter segfault 3/3 tries; `--enforce-eager` boots. NVFP4 KV never hit this.
- `fp8_e5m2` KV is rejected for fp8 checkpoints by design (clear error).
- Early-startup random `free(): invalid size` / tokenizer-pyo3 segfaults occurred a few times on this box regardless of config — a retry booted fine. Post-mortem attributes most of this family to the custom-AR/topology issue above (flag 2), not to JIT.
- When TP workers die they may leak their VRAM under renamed process titles — `nvidia-smi` PID-based kill before each boot avoids phantom `NCCL error: unhandled cuda error`.
- Vision + MTP combined: not yet re-tested with `--disable-custom-all-reduce`; the single observed crash predates the root-cause finding. The drafter code path explicitly whitelists `Qwen3_5ForConditionalGeneration`, so expect it to work.

## Not tested / out of scope
- Vision + MTP under the safe flag list above
- GB10 / sm_121 (upstream issue #49011 reports green)
- Long run soak (>1h sustained mixed load)

*Rebased + tested by @chengye75-star, Oct 2026. Upstream PR: #46329 — go thank the original authors; this fork just carries the rebase and the launch-flag findings.*
