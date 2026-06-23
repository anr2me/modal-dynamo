# Dynamo + LMCache on Modal: SGLang, vLLM, and TensorRT-LLM

Serverless Qwen3-8B-FP8 inference on [Modal](https://modal.com), with each request served through [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)'s frontend and a KV-cache offload layer. Three `@app.cls` services, one per inference backend, each on its own port:

| Class | Port | Backend | KV-cache offload |
|---|---|---|---|
| `DynamoSGLangLMCache` | 8000 | SGLang | LMCache (`--enable-lmcache`, in-process) |
| `DynamoVLLMLMCache` | 8001 | vLLM | LMCache (`LMCacheConnectorV1` via `--kv-transfer-config`) |
| `DynamoTRTLLMLMCache` | 8002 | TensorRT-LLM | LMCache (`KvCacheConnector`, in-process) |

Adapted from Modal's [SGLang + memory snapshots example](https://modal.com/docs/examples/sglang_snapshot), extended to run each engine behind Dynamo's frontend instead of bare `sglang.launch_server`, with LMCache wired in for KV-cache offload to CPU RAM.

## Before you use this: how "integrated" is each backend, really?

The three classes are **not equally mature or equally supported**. This matters if you're choosing which one to build on:

- **`DynamoVLLMLMCache`** — the real, jointly-documented, production-grade integration. NVIDIA and LMCache both maintain this connector. Start here if you just want something that works.
- **`DynamoSGLangLMCache`** — real and functional, but LMCache runs *inside* the SGLang process as a side-by-side cache layer, not a connector that Dynamo's router is aware of. NVIDIA's own Dynamo+LMCache docs page covers vLLM only; SGLang support is asserted by LMCache's own blog/docs but isn't (as of writing) reflected in NVIDIA's integration docs.
- **`DynamoTRTLLMLMCache`** — real, but **bleeding-edge**. It depends on [NVIDIA/TensorRT-LLM#12626](https://github.com/NVIDIA/TensorRT-LLM/pull/12626) (a connector preset registry) and a matching LMCache adapter that, as of writing, only exists on LMCache's `dev` branch. Neither has shipped in a stable release. The image in this repo builds both from source. Treat this one as experimental until both ship stably — see [LMCache's TensorRT-LLM quickstart](https://docs.lmcache.ai/getting_started/quickstart.html) for the latest status.

None of the three get Dynamo's actual headline feature — KV-aware routing and disaggregated prefill/decode across multiple workers — since each class runs as a single aggregated worker in a single GPU container. Multi-node disaggregation would need a materially different architecture (separate Modal services for the frontend, prefill workers, and decode workers, wired together over the network).

## Architecture

Each class runs **two subprocesses in one Modal container**:

```
                    ┌────────────────────────────────┐
  HTTP request      │ Modal container (1x H100 GPU)        │
 ───────────────►│                                      │
                    │  dynamo.frontend  ──────────────┐ │
                    │  (OpenAI-compatible API, /health) │ │
                    │       │                           │ │
                    │       ▼                          │ │
                    │  dynamo.<sglang|vllm|trtllm>      │ │
                    │  (inference engine + LMCache)     │ │
                    │       │                           │ │
                    │       ▼                          │ │
                    │  CPU RAM (offloaded KV cache)     │ │
                    └──────────────────────────────┘ │
                          watchdog threads fate-share ────┘
                          both processes (see below)
```

- **`dynamo.frontend`** is the HTTP entry point — what Modal's `@modal.web_server` forwards traffic to. It also exposes `/health`.
- **`dynamo.<backend>`** is the actual inference engine, wrapped by Dynamo's runtime. It exposes a separate "system server" (`DYN_SYSTEM_PORT`) with `/metrics` and the `/engine/release_memory_occupation` / `/engine/resume_memory_occupation` routes used for Modal's memory snapshotting.
- **Discovery** uses `--discovery-backend file` — no etcd or NATS cluster needed, since each class is a single aggregated worker with nothing to discover.
- **Crash watchdog**: the two subprocesses fate-share. If either one dies for any reason while the container is serving traffic, a background thread terminates the other, so the container never silently runs half-alive. See `start_crash_watchdog()` in the source.
- **Memory snapshotting** (Modal's GPU + CPU memory snapshot feature) is preserved from the original SGLang example: each worker is warmed up and put to sleep before the snapshot is taken, then woken on restore.

## Requirements

- A [Modal](https://modal.com) account and the `modal` CLI (`pip install modal && modal setup`)
- Enough GPU quota for at least one L40S
- For `DynamoTRTLLMLMCache`: patience — this build pulls LMCache's `dev` branch and may require building TensorRT-LLM from source if the base image doesn't already include [PR #12626](https://github.com/NVIDIA/TensorRT-LLM/pull/12626)

## Deploy

```bash
modal deploy dynamo_multi_backend.py
```

This builds all three container images (SGLang, vLLM, TensorRT-LLM) and creates one Modal App (`example-dynamo-multi-backend`) with three deployed classes. Each gets its own URL, e.g.:

```
https://your-workspace--example-dynamo-multi-backend-dynamosglanglmcache.modal.run
https://your-workspace--example-dynamo-multi-backend-dynamovllmlmcache.modal.run
https://your-workspace--example-dynamo-multi-backend-dynamotrtllmlmcache.modal.run
```

Each exposes an OpenAI-compatible `/v1/chat/completions` endpoint and Swagger docs at `/docs`.

## Test

A `local_entrypoint` is provided per backend:

```bash
modal run dynamo_multi_backend.py::test_sglang
modal run dynamo_multi_backend.py::test_vllm
modal run dynamo_multi_backend.py::test_trtllm
```

Each sends two requests sharing a system-prompt prefix, to exercise the KV-cache-reuse path — watch the container logs for cache-hit messages from LMCache.

> `modal run` creates an *ephemeral* app, which disables memory snapshotting. To see snapshot speedups, `modal deploy` first, then hit the deployed URL directly (see Modal's [memory snapshot guide](https://modal.com/docs/guide/memory-snapshot)).

## Configuration

Key constants near the top of the file:

| Constant | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-8B-FP8` | Model to serve |
| `MODEL_REVISION` | pinned commit hash | HF revision (SGLang/vLLM only — see note below) |
| `N_GPUS` | `1` | GPUs per container (tensor-parallel size) |
| `MAX_INPUTS` | `1000` | Max concurrent requests per replica |
| `TARGET_INPUTS` | `10` | Target concurrency before scaling up |
| `*_LMCACHE_MAX_LOCAL_CPU_GB` | `20` | CPU RAM budget for offloaded KV cache, per backend |

**Note on `MODEL_REVISION`:** pinned for `dynamo.sglang` and `dynamo.vllm`, both of which accept a standard `--revision` flag. It's intentionally **not** passed to `dynamo.trtllm` — no public example confirms that flag exists for this backend, and TensorRT-LLM's CLI conventions differ from vLLM/SGLang's in places. Confirm against your installed version's `--help` output before adding it back.

## Known gaps / things to verify before relying on this in production

- **Untested end-to-end.** This was built and reviewed against current documentation, but not run against live GPU hardware. Treat first deploys as a smoke test, especially for `DynamoTRTLLMLMCache`.
- **Base image tags** (`vllm/vllm-openai:v0.11.0`, `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.2.1`) may drift — check they still exist and are current before deploying.
- **TensorRT-LLM revision pinning** is unresolved (see above).
- **SGLang's LMCache integration** is not a Dynamo-aware connector — it's LMCache and Dynamo's router running side by side, uncoordinated. Don't expect Dynamo's KV-aware routing to factor in what LMCache has cached.

## License / attribution

Based on Modal's [`sglang_snapshot.py`](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/sglang_snapshot.py) example. See [Modal's documentation](https://modal.com/docs) and [NVIDIA Dynamo's documentation](https://docs.nvidia.com/dynamo) for the underlying platforms.
