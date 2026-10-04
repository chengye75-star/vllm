# NVFP4 KV cache on consumer Blackwell (sm_120 / sm_121) — tested fork

Branch: `nvfp4-kv-sm120-tested` — a rebase of [vllm-project/vllm#46329](https://github.com/vllm-project/vllm/pull/46329) onto `main` @ `155d23cb008f` (14/14 commits, no conflicts), **plus build/runtime fixes found while testing on 2× RTX 5060 Ti 16G**.

If you're here because `--kv-cache-dtype nvfp4` explodes on your RTX 50xx / GB10 box: upstream stock vLLM rejects it (FlashInfer trtllm-gen path requires SM100 datacenter). This branch restores the FA2 decode route for NVFP4 KV on consumer Blackwell.

## What's verified here (2× 5060 Ti 16G, TP2, RadixArk/Qwen3.8-27B-NVFP4, vLLM 0.30.0 + FlashInfer 0.7.0.post1)

| Check | Result |
|---|---|
| `--kv-cache-dtype nvfp4` boot | ✅ 394,545-token KV pool (fp8: 209,465) |
| + MTP spec decode (`qwen3_5_mtp`, K=1) | ✅ accept_len 1.83–2.00 @ rate 0.77–1.00 |
| Needle-in-haystack 106K & 133K ctx, 4 depths | ✅ 8/8 |
| Prefix caching, 106K repeat | ✅ 72.5s → 6.8s (10.7×) |
| Greedy determinism (4× identical) | ✅ byte-identical |
| Single-stream decode (K=1 MTP) | 50–57 t/s vs 39 t/s no-MTP |

## Required launch flags (two real traps found in testing)

```bash
vllm serve <nvfp4-model> --kv-cache-dtype nvfp4 \
  --kernel-config '{"enable_flashinfer_autotune": false}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'  # optional
```

1. **`--kernel-config '{"enable_flashinfer_autotune": false}'` is mandatory for MTP + nvfp4 KV.** With FlashInfer autotune enabled, its benchmarking can land inside CUDA graph capture → `CUDA error: operation not permitted when stream is capturing` at startup. Non-deterministic (timing-dependent): K=1 booted once out of three tries. Disabling it uses the cached-tactic path and booted 2/2 green. Worth an upstream fix (defer autotune out of capture).
2. **fp8 KV (`fp8_e4m3`) + torch.compile crashed the worker with an interpreter segfault on this box (CUDA 13.1 / glibc-heavy stack); `--enforce-eager` or nvfp4 both boot.** e5m2 is rejected by design for fp8 checkpoints.

## Building from this branch (source build gotchas, all hit personally)

- **Host compiler**: CUDA 13.1 headers do not compile with g++-15 (Ubuntu 25.10+ default). `apt install g++-13` and build with
  `CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=/usr/bin/g++-13`.
- The NVFP4 KV write kernel needs `arch=compute_120a,code=sm_120a` (`cvt.e2m1x2.f32` is an sm_120**a** instruction; plain `sm_120` is rejected by ptxas). The branch's CMake already targets it correctly; this matters if you hand-compile.
- **nvcc/cudafe++/cc1plus segfault randomly under high `-j`** on this toolchain combo. Workaround that produced 406/406 objects: `ninja -k 0 -j 8` in a retry loop (incremental — crashed files get recompiled next round). Don't panic on first failure.
- `uv`'s editable-install metadata hook SIGSEGVs on vllm's pyproject here; plain `pip install --no-build-isolation --no-deps -e .` works.
- Python 3.12, `flashinfer-python==0.7.0.post1` (from the flashinfer.ai wheel index; PyPI has no matching `flashinfer-cubin`).

## Not tested / out of scope
- Vision tower (`--language-model-only` was used throughout; multimodal NVFP4 KV untested here)
- Tool-call parsing with nvfp4 KV (blocked by the autotune trap above; battery ready, rerunning)
- GB10 / sm_121 (upstream issue #49011 reports green)

Repro of every number above: see the GitHub Release notes attached to this branch (test scripts + engine logs).

*Rebased + tested by @chengye75-star, Oct 2026. Upstream PR: #46329 — go thank the original authors; this fork just carries the rebase and the two launch-flag findings.*
