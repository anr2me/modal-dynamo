"""
Serverless Qwen 3.6-27B with NVIDIA Dynamo (SGLang / vLLM / TensorRT-LLM) +
KV-cache offload on Modal.

Adapted from: https://modal.com/docs/examples/sglang_snapshot

=============================================================================
READ THIS FIRST: these three classes are NOT equally "LMCache-integrated"
=============================================================================
You asked for three @app.cls services - SGLang, vLLM, TensorRT-LLM - each
paired with LMCache under Dynamo. That pairing is real for exactly one of
the three. Here's the honest state of each, checked against NVIDIA's and
LMCache's own docs:

  DynamoSGLangLMCache  - `dynamo.sglang` + SGLang's native `--enable-lmcache`
                         flag. Real, but NOT a coordinated Dynamo<->LMCache
                         connector: NVIDIA's own LMCache integration doc
                         (https://docs.nvidia.com/dynamo/integrations/kv-cache-integrations/lm-cache)
                         documents the vLLM backend only, with no SGLang
                         section. LMCache's own blog (March 2026) claims
                         engine-level SGLang support, but no documented,
                         wired SGLang<->Dynamo connector exists at the time
                         of writing. LMCache runs *inside* the SGLang
                         process; Dynamo's router has no visibility into it.

  DynamoVLLMLMCache    - `dynamo.vllm` + a real, documented `kv_connector`:
                         --kv-transfer-config
                         '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
                         This is the one NVIDIA and LMCache both fully
                         document and jointly maintain. If you want the
                         actual, supported Dynamo+LMCache integration, this
                         is it.

  DynamoTRTLLMLMCache  - `dynamo.trtllm` + LMCache's real TensorRT-LLM
                         KvCacheConnector integration (in-process mode):
                         https://docs.lmcache.ai/integrations/tensorrt_llm.html
                         CORRECTION from an earlier pass of this file: this
                         integration does exist, contrary to what was said
                         before. As of this writing it depends on UNRELEASED
                         code on both sides - NVIDIA/TensorRT-LLM PR #12626
                         (the connector preset registry) and a matching
                         LMCache adapter that's only on LMCache's `dev`
                         branch - so the image below builds both from
                         source. Once both ship in stable releases, the
                         from-source installs in the image can be replaced
                         with normal pinned-version installs (see the
                         comment above the image definition).

All three remain SINGLE Modal @app.cls / single GPU container deployments,
aggregated mode (one worker does prefill + decode), using
--discovery-backend file so no real etcd/NATS cluster is needed. None of
these get Dynamo's actual headline feature - KV-aware routing / disaggregated
serving across many workers - since that requires more than one worker.
Each class listens on its own port so all three could, in principle, run as
separate Modal services from the same file without colliding if you ever
deployed them simultaneously for an apples-to-apples comparison.
=============================================================================
"""

import asyncio
import os
import subprocess
import threading
import time

import aiohttp
import modal
import modal.experimental

# `requests` is used by helper functions (warmup/sleep/wake_up/wait_ready)
# that run inside the container for ALL THREE backend classes below, each
# of which builds its own Modal Image. Rather than guard this import behind
# any single image's `.imports()` block (which would only cover that one
# image's container), we rely on `requests` already being present in all
# three base images (it's a transitive dependency of huggingface-hub,
# vllm, and the Dynamo runtime images alike). If you swap to a leaner base
# image for any backend, add `.uv_pip_install("requests")` to that image.
try:
    import requests
except ImportError:
    requests = None  # only relevant for local (non-container) import-time checks

MINUTES = 60  # seconds
IDLETIME = 2 * MINUTES

MODEL_NAME = "Qwen/Qwen3.6-27B-FP8"
MODEL_REVISION = (
    "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"  # latest commit
)

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

DG_CACHE_VOL = modal.Volume.from_name("deepgemm-cache", create_if_missing=True)
DG_CACHE_PATH = "/root/.cache/deep_gemm"

N_GPUS = 1
GPU = f"L40S:{N_GPUS}"

TARGET_INPUTS = 10
MAX_INPUTS = 1000

MAX_CONTAINERS = 1

LMCACHE_MAX_LOCAL_CPU_GB = 20


# -----------------------------------------------------------------------
# Ports - each backend's frontend gets its own port so the three classes
# never collide if deployed together.
# -----------------------------------------------------------------------
SGLANG_FRONTEND_PORT = 8000
SGLANG_SYSTEM_PORT = 8081

VLLM_FRONTEND_PORT = 8001
VLLM_SYSTEM_PORT = 8082

TRTLLM_FRONTEND_PORT = 8002
TRTLLM_SYSTEM_PORT = 8083


# =========================================================================
# Shared helpers
# =========================================================================
# All three `dynamo.<backend>` workers expose the same "system server"
# pattern (DYN_SYSTEM_PORT -> /metrics, /engine/release_memory_occupation,
# /engine/resume_memory_occupation) and the same Dynamo frontend pattern
# (--http-port -> /health, /v1/chat/completions). So the lifecycle helpers
# below are parameterized by port rather than duplicated per backend.
# See https://docs.nvidia.com/dynamo/backends/sg-lang/reference-guide for
# the canonical description of this split (SGLang's reference guide, but
# the pattern is shared by dynamo.vllm and dynamo.trtllm too).


def make_warmup(frontend_port: int, model_name: str):
    def warmup():
        payload = {
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 16,
            "model": model_name,
        }
        for _ in range(3):
            requests.post(
                f"http://127.0.0.1:{frontend_port}/v1/chat/completions",
                json=payload,
                timeout=10,
            ).raise_for_status()

    return warmup


def make_sleep(system_port: int):
    def sleep():
        requests.post(
            f"http://127.0.0.1:{system_port}/engine/release_memory_occupation",
            json={},
        ).raise_for_status()

    return sleep


def make_wake_up(system_port: int):
    def wake_up():
        requests.post(
            f"http://127.0.0.1:{system_port}/engine/resume_memory_occupation",
            json={},
        ).raise_for_status()

    return wake_up


def wait_ready(
    process: subprocess.Popen, frontend_port: int, timeout: int = 5 * MINUTES
):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:{frontend_port}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(1)
    raise TimeoutError(f"Dynamo server not ready within timeout of {timeout} seconds")


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def start_crash_watchdog(frontend_process: subprocess.Popen, worker_process: subprocess.Popen):
    """Make the frontend and worker processes fate-share: if either one
    exits for any reason (crash, OOM kill, etc.) while the container is
    serving traffic, terminate the other so the container doesn't keep
    running half-alive and silently failing requests.

    A try/finally around the two Popen() calls would NOT catch this: Popen
    returns as soon as the child is spawned, so a crash that happens later
    - after startup() has already returned and the container is serving
    traffic - doesn't raise anywhere and never reaches a finally block.
    Watching each process for the container's lifetime (via .wait(), which
    blocks until that specific process exits) is what's actually needed.

    Runs as two daemon threads so they never block container shutdown.
    """

    def _watch_and_kill_sibling(to_watch: subprocess.Popen, to_kill: subprocess.Popen):
        to_watch.wait()  # blocks until `to_watch` exits, for any reason
        if to_kill.poll() is None:  # only kill if the sibling is still alive
            to_kill.terminate()

    threading.Thread(
        target=_watch_and_kill_sibling,
        args=(frontend_process, worker_process),
        daemon=True,
    ).start()
    threading.Thread(
        target=_watch_and_kill_sibling,
        args=(worker_process, frontend_process),
        daemon=True,
    ).start()


async def probe(url, model_name, messages=None, timeout=5 * MINUTES):
    if messages is None:
        messages = [{"role": "user", "content": "Tell me a joke."}]

    deadline = time.time() + timeout
    async with aiohttp.ClientSession(base_url=url) as session:
        while time.time() < deadline:
            try:
                await _send_request(session, model_name, messages)
                return
            except asyncio.TimeoutError:
                await asyncio.sleep(1)
    raise TimeoutError(f"No response from server within {timeout} seconds")


async def _send_request(
    session: aiohttp.ClientSession,
    model: str,
    messages: list,
    timeout: int | None = None,
) -> None:
    async with session.post(
        "/v1/chat/completions",
        json={"messages": messages, "model": model},
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        print((await resp.json())["choices"][0]["message"]["content"])


def compile_deep_gemm():
    if int(os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "1")):
        subprocess.run(
            f"python3 -m sglang.compile_deep_gemm --model-path {MODEL_NAME} --revision {MODEL_REVISION} --tp {N_GPUS}",
            shell=True,
        )


def get_secrets() -> list[modal.Secret]:
    """Prefer Modal Secret 'huggingface-secret'; fall back to local HF_TOKEN env. 
    Public models work even when both are absent (warned)."""
    secrets = []
    # Try with 'huggingface-secret'
    try:
        s = modal.Secret.from_name("huggingface-secret")
        s.hydrate()  # from_name is lazy, force the existence check here
        secrets.append(s)
    except modal.exception.NotFoundError:
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            print(
                "Warning: no Modal Secret 'huggingface-secret' and no HF_TOKEN env. "
                "Public models will download with throttled bandwidth; "
                "gated models will fail."
            )
        secrets.append(modal.Secret.from_dict({"HF_TOKEN": token}))
    return secrets


app = modal.App(name="dynamo")


# =========================================================================
# 1) SGLang backend
# =========================================================================
sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.13.post1-runtime", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(["pip", "uv"], extra_options="--upgrade")
    .uv_pip_install("huggingface-hub>=0.36.0", "requests")
    # --prerelease=allow is required by lmcache's SGLang integration
    # per LMCache's own quickstart docs.
    .uv_pip_install(
        ["ai-dynamo[sglang]", "lmcache"], extra_options="--no-build-isolation", pre=True
    )
    .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"})
    .env({"SGLANG_ENABLE_JIT_DEEPGEMM": "1"})
)
sglang_image = sglang_image.run_function(
    compile_deep_gemm,
    volumes={DG_CACHE_PATH: DG_CACHE_VOL, HF_CACHE_PATH: HF_CACHE_VOL},
    gpu=GPU,
    secrets=get_secrets(),
)
sglang_image = sglang_image.env({"TORCHINDUCTOR_COMPILE_THREADS": "1"})

# LMCache config file content for the SGLang worker. Per
# https://docs.lmcache.ai/getting_started/quickstart.html: chunk_size 256 is
# the documented production default; max_local_cpu_size is the CPU RAM
# budget in GB for offloaded KV blocks - tune to your container's available
# RAM, leaving headroom for everything else running.
SGLANG_LMCACHE_CONFIG_PATH = "/root/lmcache_config_sglang.yaml"
SGLANG_LMCACHE_MAX_LOCAL_CPU_GB = LMCACHE_MAX_LOCAL_CPU_GB
SGLANG_LMCACHE_CONFIG_YAML = f"""\
chunk_size: 256
local_cpu: true
use_layerwise: true
max_local_cpu_size: {SGLANG_LMCACHE_MAX_LOCAL_CPU_GB}
"""


@app.cls(
    max_containers=MAX_CONTAINERS,
    scaledown_window=IDLETIME,
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, DG_CACHE_PATH: DG_CACHE_VOL},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    secrets=get_secrets(),
)
@modal.concurrent(target_inputs=TARGET_INPUTS, max_inputs=MAX_INPUTS)
class DynamoSGLangLMCache:
    """Dynamo frontend + dynamo.sglang worker with SGLang's native
    --enable-lmcache flag. See the module docstring: this is LMCache running
    inside the SGLang process, not a coordinated Dynamo<->LMCache connector.
    """

    @modal.enter(snap=True)
    def startup(self):
        with open(SGLANG_LMCACHE_CONFIG_PATH, "w") as f:
            f.write(SGLANG_LMCACHE_CONFIG_YAML)

        frontend_cmd = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--http-host",
            "0.0.0.0",
            "--http-port",
            f"{SGLANG_FRONTEND_PORT}",
            "--discovery-backend",
            "file",
        ]
        self.frontend_process = subprocess.Popen(frontend_cmd)

        worker_cmd = [
            "python3",
            "-m",
            "dynamo.sglang",
            "--model-path",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--discovery-backend",
            "file",
            "--tp",
            f"{N_GPUS}",
            "--cuda-graph-max-bs",
            f"{MAX_INPUTS}",
            "--max-running-requests",
            f"{MAX_INPUTS}",
            "--enable-metrics",
            "--enable-memory-saver",  # enable offload, for snapshotting
            "--enable-weights-cpu-backup",  # enable offload, for snapshotting
            "--enable-lmcache",
        ]
        worker_env = {
            **os.environ,
            "LMCACHE_CONFIG_FILE": SGLANG_LMCACHE_CONFIG_PATH,
            "DYN_SYSTEM_PORT": f"{SGLANG_SYSTEM_PORT}",
        }
        self.worker_process = subprocess.Popen(worker_cmd, env=worker_env)

        start_crash_watchdog(self.frontend_process, self.worker_process)

        wait_ready(self.frontend_process, SGLANG_FRONTEND_PORT)
        make_warmup(SGLANG_FRONTEND_PORT, MODEL_NAME)()
        make_sleep(SGLANG_SYSTEM_PORT)()

    @modal.enter(snap=False)
    def wake_up_(self):
        make_wake_up(SGLANG_SYSTEM_PORT)()

    @modal.web_server(port=SGLANG_FRONTEND_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.worker_process.terminate()
        self.frontend_process.terminate()


# =========================================================================
# 2) vLLM backend - the real, documented Dynamo<->LMCache connector
# =========================================================================
# Per https://docs.nvidia.com/dynamo/integrations/kv-cache-integrations/lm-cache
# LMCache is wired in via vLLM's own KVTransferConfig / kv_connector
# mechanism: --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1",
# "kv_role":"kv_both"}'. LMCache's behavior (chunk size, CPU cache size,
# etc.) is then controlled by the LMCACHE_* environment variables below,
# which the connector reads directly - no separate YAML file needed here
# (unlike the SGLang path).
vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.23.0", add_python="3.12")
    .entrypoint([])
    .apt_install(["clang", "llvm"])
    .uv_pip_install(["pip", "uv"], extra_options="--upgrade")
    .uv_pip_install("huggingface-hub>=0.36.0", "requests")
    .uv_pip_install(["numpy", "torch~=2.11.0", "torchao~=0.17.0", "torchvision~=0.26.0", "torchaudio~=2.11.0"], extra_options="--upgrade", index_url="https://download.pytorch.org/whl/cu130") # xformers
    .uv_pip_install(
        ["ai-dynamo[vllm]", "lmcache"], extra_options="--no-build-isolation --only-binary lmcache", pre=True
    )
    .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"})
    .env({"TORCHINDUCTOR_COMPILE_THREADS": "1"})
)

VLLM_KV_TRANSFER_CONFIG = (
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
)
VLLM_LMCACHE_MAX_LOCAL_CPU_GB = f"{LMCACHE_MAX_LOCAL_CPU_GB}"


@app.cls(
    max_containers=MAX_CONTAINERS,
    scaledown_window=IDLETIME,
    image=vllm_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    secrets=get_secrets(),
)
@modal.concurrent(target_inputs=TARGET_INPUTS, max_inputs=MAX_INPUTS)
class DynamoVLLMLMCache:
    """Dynamo frontend + dynamo.vllm worker using the real LMCacheConnectorV1
    kv_connector. This is the one NVIDIA and LMCache jointly document and
    maintain - see the module docstring."""

    @modal.enter(snap=True)
    def startup(self):
        frontend_cmd = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--http-host",
            "0.0.0.0",
            "--http-port",
            f"{VLLM_FRONTEND_PORT}",
            "--discovery-backend",
            "file",
        ]
        self.frontend_process = subprocess.Popen(frontend_cmd)

        worker_cmd = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--discovery-backend",
            "file",
            "--tensor-parallel-size",
            f"{N_GPUS}",
            "--max-num-seqs",
            f"{MAX_INPUTS}",
            "--enable-metrics",
            "--kv-transfer-config",
            VLLM_KV_TRANSFER_CONFIG,
        ]
        worker_env = {
            **os.environ,
            "DYN_SYSTEM_PORT": f"{VLLM_SYSTEM_PORT}",
            # LMCache's own runtime config, read directly by
            # LMCacheConnectorV1 - no config file needed for this backend.
            "LMCACHE_CHUNK_SIZE": "256",
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": VLLM_LMCACHE_MAX_LOCAL_CPU_GB,
        }
        self.worker_process = subprocess.Popen(worker_cmd, env=worker_env)

        start_crash_watchdog(self.frontend_process, self.worker_process)

        wait_ready(self.frontend_process, VLLM_FRONTEND_PORT)
        make_warmup(VLLM_FRONTEND_PORT, MODEL_NAME)()
        make_sleep(VLLM_SYSTEM_PORT)()

    @modal.enter(snap=False)
    def wake_up_(self):
        make_wake_up(VLLM_SYSTEM_PORT)()

    @modal.web_server(port=VLLM_FRONTEND_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.worker_process.terminate()
        self.frontend_process.terminate()


# =========================================================================
# 3) TensorRT-LLM backend - real LMCache connector (bleeding-edge install)
# =========================================================================
# UPDATE: LMCache does integrate with TensorRT-LLM, via TRT-LLM's own KV
# Cache Connector API (tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector).
# See https://docs.lmcache.ai/integrations/tensorrt_llm.html and the
# TensorRT-LLM tab of https://docs.lmcache.ai/getting_started/quickstart.html.
#
# AS OF THIS WRITING, THIS DEPENDS ON UNRELEASED CODE ON BOTH SIDES:
#   - NVIDIA/TensorRT-LLM PR #12626 (the connector preset registry) is not
#     in a stable TensorRT-LLM release.
#   - The matching LMCache adapter is only on LMCache's `dev` branch.
# Per LMCache's own quickstart, until both ship in a stable release you
# must build both from source:
#     uv pip install git+https://github.com/LMCache/LMCache.git@dev
#     # TensorRT-LLM from source: see
#     # https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html
# Once both ship stably, the install collapses to:
#     uv pip install lmcache "tensorrt_llm>=<version>" \
#         --extra-index-url https://pypi.nvidia.com
# The image below installs from source per the current (unstable) path.
# If you're reading this after both have shipped stably, switch to the
# simpler pinned-version install and drop the git+ source builds.
#
# WIRING: connector config uses the SAME --extra-engine-args YAML mechanism
# as Dynamo's KVBM (see the dynamo.trtllm KVBM guide:
# https://docs.nvidia.com/dynamo/components/kvbm/kvbm-guide), just pointed
# at LMCache's connector classes instead of KVBM's:
#     kv_connector_config:
#       connector_module: lmcache.integration.tensorrt_llm.tensorrt_adapter
#       connector_scheduler_class: LMCacheKvConnectorScheduler
#       connector_worker_class: LMCacheKvConnectorWorker
# This is in-process mode (LMCache runs as a singleton inside the TRT-LLM
# worker process) - the simpler of the two modes LMCache supports for
# TRT-LLM (the other being a standalone `lmcache-mp` server, useful if you
# want the cache to survive a worker crash or be shared across workers,
# which doesn't apply to this single-container deployment).
# Requires TensorRT-LLM >= 1.2.0 (the KvCacheConnector ABC was added then),
# and an LMCache build with the c_ops extension (verify with
# `python -c "import lmcache.c_ops"` inside the container).
trtllm_image = (
    modal.Image.from_registry("nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.2.1", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(["pip", "uv"], extra_options="--upgrade")
    .uv_pip_install("huggingface-hub>=0.36.0", "requests")
    # LMCache's TensorRT-LLM connector is only on the `dev` branch until
    # NVIDIA/TensorRT-LLM#12626 and the matching adapter ship stably.
    #.uv_pip_install("git+https://github.com/LMCache/LMCache.git@dev", extra_options="--no-build-isolation", pre=True, gpu=GPU)
    .uv_pip_install("lmcache", extra_options="--no-build-isolation", pre=True)
    .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"})
    # PYTHONHASHSEED=0 is required by LMCache's TRT-LLM adapter: chunk
    # hashing depends on a stable hash() across runs/processes.
    .env({"PYTHONHASHSEED": "0"})
)

TRTLLM_LMCACHE_CONFIG_PATH = "/root/lmcache_trtllm_config.yaml"
TRTLLM_LMCACHE_MAX_LOCAL_CPU_GB = f"{LMCACHE_MAX_LOCAL_CPU_GB}"
TRTLLM_LMCACHE_CONFIG_YAML = """\
kv_cache_config:
  enable_block_reuse: true
kv_connector_config:
  connector_module: lmcache.integration.tensorrt_llm.tensorrt_adapter
  connector_scheduler_class: LMCacheKvConnectorScheduler
  connector_worker_class: LMCacheKvConnectorWorker
"""


@app.cls(
    max_containers=MAX_CONTAINERS,
    scaledown_window=IDLETIME,
    image=trtllm_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    secrets=get_secrets(),
)
@modal.concurrent(target_inputs=TARGET_INPUTS, max_inputs=MAX_INPUTS)
class DynamoTRTLLMLMCache:
    """Dynamo frontend + dynamo.trtllm worker using LMCache's real
    KvCacheConnector integration (in-process mode). Depends on unreleased
    TensorRT-LLM/LMCache code as of this writing - see the module and
    section docstrings above for the from-source install this requires."""

    @modal.enter(snap=True)
    def startup(self):
        with open(TRTLLM_LMCACHE_CONFIG_PATH, "w") as f:
            f.write(TRTLLM_LMCACHE_CONFIG_YAML)

        frontend_cmd = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--http-host",
            "0.0.0.0",
            "--http-port",
            f"{TRTLLM_FRONTEND_PORT}",
            "--discovery-backend",
            "file",
        ]
        self.frontend_process = subprocess.Popen(frontend_cmd)

        worker_cmd = [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model-path",
            MODEL_NAME,
            # NOTE: unlike dynamo.sglang/dynamo.vllm, no public dynamo.trtllm
            # example confirms a --revision flag, and TensorRT-LLM's own CLI
            # (trtllm-serve) uses underscore-style flags in places (e.g.
            # --hf_revision) that don't always mirror vLLM/SGLang's
            # hyphenated style. Rather than guess and risk an outright CLI
            # parse error, MODEL_REVISION pinning is intentionally omitted
            # here. If you need a pinned revision for TRT-LLM, confirm the
            # correct flag name against your installed
            # TensorRT-LLM/Dynamo version's --help output first.
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--discovery-backend",
            "file",
            "--tensor-parallel-size",
            f"{N_GPUS}",
            "--max-batch-size",
            f"{MAX_INPUTS}",
            "--extra-engine-args",
            TRTLLM_LMCACHE_CONFIG_PATH,
        ]
        worker_env = {
            **os.environ,
            "DYN_SYSTEM_PORT": f"{TRTLLM_SYSTEM_PORT}",
            # LMCache's TRT-LLM adapter reads LMCacheEngineConfig the same
            # way the vLLM adapter does: LMCACHE_CONFIG_FILE for a YAML
            # file, otherwise individual LMCACHE_* env vars (used here).
            "LMCACHE_CHUNK_SIZE": "256",
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": TRTLLM_LMCACHE_MAX_LOCAL_CPU_GB,
        }
        self.worker_process = subprocess.Popen(worker_cmd, env=worker_env)

        start_crash_watchdog(self.frontend_process, self.worker_process)

        wait_ready(self.frontend_process, TRTLLM_FRONTEND_PORT)
        make_warmup(TRTLLM_FRONTEND_PORT, MODEL_NAME)()
        make_sleep(TRTLLM_SYSTEM_PORT)()

    @modal.enter(snap=False)
    def wake_up_(self):
        make_wake_up(TRTLLM_SYSTEM_PORT)()

    @modal.web_server(port=TRTLLM_FRONTEND_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.worker_process.terminate()
        self.frontend_process.terminate()


# =========================================================================
# Local entrypoints for testing
# =========================================================================
@app.local_entrypoint()
async def test_sglang(test_timeout=10 * MINUTES, prompt=None, twice=True):
    await _run_test(DynamoSGLangLMCache(), test_timeout, prompt, twice)


@app.local_entrypoint()
async def test_vllm(test_timeout=10 * MINUTES, prompt=None, twice=True):
    await _run_test(DynamoVLLMLMCache(), test_timeout, prompt, twice)


@app.local_entrypoint()
async def test_trtllm(test_timeout=10 * MINUTES, prompt=None, twice=True):
    await _run_test(DynamoTRTLLMLMCache(), test_timeout, prompt, twice)


async def _run_test(cls_instance, test_timeout, prompt, twice):
    url = cls_instance.serve.get_web_url()

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if prompt is None:
        prompt = "Explain the Singular Value Decomposition."

    content = [{"type": "text", "text": prompt}]
    messages = [system_prompt, {"role": "user", "content": content}]

    await probe(url, MODEL_NAME, messages, timeout=test_timeout)
    if twice:
        # Shares the same system-prompt prefix as the first request, to
        # exercise the KV-cache-reuse path (LMCache or KVBM, depending on
        # which class you're testing).
        messages[0]["content"] = "You are Jar Jar Binks."
        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await probe(url, MODEL_NAME, messages, timeout=1 * MINUTES)
