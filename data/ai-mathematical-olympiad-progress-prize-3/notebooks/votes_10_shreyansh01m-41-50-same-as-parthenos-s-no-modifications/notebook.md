# [41/50]Same as parthenos's, no modifications.

- **Author:** Siancy
- **Votes:** 404
- **Ref:** shreyansh01m/41-50-same-as-parthenos-s-no-modifications
- **URL:** https://www.kaggle.com/code/shreyansh01m/41-50-same-as-parthenos-s-no-modifications
- **Last run:** 2025-12-31 13:22:45.750000

---

```python
import time
import numpy as np
import os

start_time = time.time()
final_cutoff_time = start_time + (4 * 60 + 58) * 60  # 4h 55m

TOTAL_TIME = 4 * 60 * 60 + 58 * 60  # 4h 55m
NUM_QUESTIONS = 50
BUFFER_TIME = 60
```

```python
import subprocess

uninstall_proc = subprocess.Popen(
    ["pip", "uninstall", "--yes", "tensorflow", "matplotlib", "keras", "scikit-learn"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
```

```python
%%time
!find /kaggle/usr/lib -type f -print0 | xargs -0 -P 32 -n 500 cat > /dev/null
```

```python
def cache_model(path, exts=(".bin", ".pt", ".safetensors"), num_workers=None, chunk_mb=256):
    """Pre-read model weight files into OS page cache."""
    import os
    import multiprocessing
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def warmup_file(fpath):
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        with open(fpath, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                total += len(data)
        return fpath, total

    if os.path.isdir(path):
        files = [
            os.path.join(root, name)
            for root, _, names in os.walk(path)
            for name in names
            if name.endswith(exts)
        ]
        files.sort()
    else:
        files = [path]

    if not files:
        raise ValueError(f"No model files found under: {path}")

    if num_workers is None:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
        except Exception:
            num_workers = 4

    print(f"[cache_model] {len(files)} file(s), {num_workers} worker(s)")
    t0 = time.time()
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fpath, n = fut.result()
            total_bytes += n
            print(f"[{i}/{len(files)}] cached {os.path.basename(fpath)}")

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    print(f"[cache_model] total read ≈ {gb:.2f} GB in {elapsed:.2f}s")
    return total_bytes


cache_model("/kaggle/input/gpt-oss-120b/transformers/default/1", num_workers=16, chunk_mb=1024)
```

```python
%%time
# Copy vLLM compile cache if available
import os
if os.path.exists("/kaggle/input/gpt-oss-120b-cache-compile/torch_compile_cache"):
    !mkdir -p /root/.cache/vllm/
    !cp -r /kaggle/input/gpt-oss-120b-cache-compile/torch_compile_cache /root/.cache/vllm/
```

```python
uninstall_proc.wait()
```

```python
subprocess.run(["ls", "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"])
```

```python
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TIKTOKEN_ENCODINGS_BASE"] = "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"
```

# Python Tool with Jupyter Kernel

```python
%%writefile local_python_tool.py
"""Python tool using Jupyter kernel for stateful execution."""
import os
import queue
import threading
from abc import ABC, abstractmethod
from typing import AsyncIterator, Any
from uuid import UUID, uuid4

from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)


def add_libs(code: str) -> str:
    """Add common math libraries to code."""
    return "import math\nimport numpy as np\nimport sympy as sp\nfrom sympy import *\n" + code


def ensure_last_print(code: str) -> str:
    """Ensure the last expression is printed."""
    lines = code.strip().split("\n")
    if lines and "print(" not in lines[-1] and "import" not in lines[-1]:
        if "#" in lines[-1]:
            lines[-1] = lines[-1].split("#")[0]
        lines[-1] = "print(" + lines[-1] + ")"
    return "\n".join(lines)


class LocalJupyterSession:
    """Stateful Jupyter kernel session for code execution."""

    # Class-level lock and port counter to avoid port conflicts
    _port_lock = threading.Lock()
    _next_port = 50000
    _max_port = 65535  # Maximum valid port number

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        """Get next available ports for kernel connection."""
        import socket
        with cls._port_lock:
            ports = []
            attempts = 0
            max_attempts = 100  # Prevent infinite loop
            
            while len(ports) < count and attempts < max_attempts:
                start_port = cls._next_port
                # Check if port range is available
                available = True
                for i in range(count):
                    port = start_port + i
                    if port > cls._max_port:
                        # Wrap around to beginning of port range
                        start_port = 50000
                        port = start_port + i
                    
                    # Quick check if port is in use
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.settimeout(0.1)
                            result = s.connect_ex(('127.0.0.1', port))
                            if result == 0:
                                available = False
                                break
                    except Exception:
                        # If check fails, assume port might be in use
                        available = False
                        break
                
                if available:
                    ports = list(range(start_port, start_port + count))
                    cls._next_port = start_port + count
                    if cls._next_port > cls._max_port:
                        cls._next_port = 50000
                    break
                else:
                    # Try next range
                    cls._next_port += count
                    if cls._next_port > cls._max_port:
                        cls._next_port = 50000
                    attempts += 1
            
            if len(ports) < count:
                # Fallback: just return sequential ports without checking
                ports = list(range(cls._next_port, cls._next_port + count))
                cls._next_port += count
                if cls._next_port > cls._max_port:
                    cls._next_port = 50000
            
            return ports

    def __init__(self, connection_file: str | None = None, *, timeout: float = 120.0):
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:
            raise RuntimeError("jupyter_client package required") from exc

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client: BlockingKernelClient
        self._km: KernelManager | None = None

        if connection_file:
            from pathlib import Path
            connection_path = Path(connection_file).expanduser()
            if not connection_path.exists():
                raise FileNotFoundError(f"Connection file not found: {connection_path}")
            client = BlockingKernelClient()
            client.load_connection_file(str(connection_path))
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
        else:
            # Allocate unique ports to avoid conflicts when running multiple kernels
            ports = self._get_next_ports(5)
            km = None
            max_retries = 3
            for retry in range(max_retries):
                try:
                    km = KernelManager()
                    km.shell_port = ports[0]
                    km.iopub_port = ports[1]
                    km.stdin_port = ports[2]
                    km.hb_port = ports[3]
                    km.control_port = ports[4]
                    km.start_kernel()
                    client = km.blocking_client()
                    client.start_channels()
                    client.wait_for_ready(timeout=self._default_timeout)
                    self._client = client
                    self._km = km
                    self._owns_kernel = True
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        # Try different ports
                        ports = self._get_next_ports(5)
                        if km is not None:
                            try:
                                km.shutdown_kernel(now=True)
                            except Exception:
                                pass
                    else:
                        # Last retry failed, raise the exception
                        raise RuntimeError(f"Failed to start kernel after {max_retries} retries: {e}") from e

    def execute(self, code: str, *, timeout: float | None = None) -> str:
        """Execute code and return combined stdout/stderr.
        timeout: WALL-CLOCK seconds limit for this execution.
        """
        import time
        import queue as _queue
    
        client = self._client
        effective_timeout = float(timeout or self._default_timeout)
    
        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)
    
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        
        # Track if we've seen a timeout/interrupt to filter IPython internal errors
        _timeout_triggered = False
    
        start = time.time()
        poll = 0.5  # seconds: small polling interval so we can enforce wall-clock timeout
    
        def _timed_out() -> bool:
            return (time.time() - start) >= effective_timeout
    
        # iopub loop
        max_timeout_grace = 1.0  # Give kernel 1 seconds to clean up after interrupt
        timeout_grace_start = None
        
        while True:
            if _timed_out():
                if not _timeout_triggered:
                    _timeout_triggered = True
                    timeout_grace_start = time.time()
                    # interrupt the kernel to stop runaway execution
                    try:
                        # BlockingKernelClient usually has interrupt_kernel
                        client.interrupt_kernel()
                    except Exception:
                        try:
                            if self._owns_kernel and self._km is not None:
                                self._km.interrupt_kernel()
                        except Exception:
                            pass
                
                # After grace period, stop collecting messages and raise timeout
                if timeout_grace_start and (time.time() - timeout_grace_start) > max_timeout_grace:
                    raise TimeoutError(f"Python execution exceeded wall-time limit: {effective_timeout:.1f}s")
    
            try:
                msg = client.get_iopub_msg(timeout=poll)
            except _queue.Empty:
                if _timeout_triggered and timeout_grace_start and (time.time() - timeout_grace_start) > max_timeout_grace:
                    raise TimeoutError(f"Python execution exceeded wall-time limit: {effective_timeout:.1f}s")
                continue
    
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
    
            msg_type = msg.get("msg_type")
            content = msg.get("content", {})
            
            # After timeout is triggered, only collect essential messages and filter IPython errors
            if _timeout_triggered:
                # Only process status messages to detect idle state, ignore everything else
                if msg_type == "status":
                    if content.get("execution_state") == "idle":
                        break
                # Skip all other messages after timeout to avoid IPython internal errors
                continue
    
            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
    
        # shell reply (also wall-time protected)
        # Reuse timeout_grace_start from iopub loop if timeout was already triggered
        shell_timeout_grace_start = timeout_grace_start if _timeout_triggered else None
        
        while True:
            if _timed_out():
                if not _timeout_triggered:
                    _timeout_triggered = True
                    shell_timeout_grace_start = time.time()
                    try:
                        client.interrupt_kernel()
                    except Exception:
                        try:
                            if self._owns_kernel and self._km is not None:
                                self._km.interrupt_kernel()
                        except Exception:
                            pass
                
                # After grace period, stop collecting messages and raise timeout
                if shell_timeout_grace_start and (time.time() - shell_timeout_grace_start) > max_timeout_grace:
                    raise TimeoutError(f"Python execution exceeded wall-time limit: {effective_timeout:.1f}s")
    
            try:
                reply = client.get_shell_msg(timeout=poll)
            except _queue.Empty:
                if _timeout_triggered and shell_timeout_grace_start and (time.time() - shell_timeout_grace_start) > max_timeout_grace:
                    raise TimeoutError(f"Python execution exceeded wall-time limit: {effective_timeout:.1f}s")
                continue
    
            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue
    
            reply_content = reply.get("content", {})
            
            # After timeout, skip error messages to avoid IPython internal errors
            if _timeout_triggered and reply_content.get("status") == "error":
                # Skip IPython internal errors, just break to exit
                break
            
            if reply_content.get("status") == "error":
                traceback_data = reply_content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break
    
        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
    
        if stderr:
            stdout = f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        if not stdout.strip():
            stdout = "[WARN] No output. Use print() to see results."
        return stdout


    def close(self):
        import contextlib
        with contextlib.suppress(Exception):
            self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __del__(self):
        self.close()


class PythonTool:
    """Python execution tool using Jupyter kernel."""

    def __init__(self, execution_backend: str | None = None, local_jupyter_timeout: float = 60.0):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._execution_lock = threading.Lock()
        self._jupyter_session: LocalJupyterSession | None = None
        # Lazy initialization to avoid port conflicts during object creation
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        """Lazily initialize the Jupyter session."""
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = LocalJupyterSession(timeout=self._local_jupyter_timeout)

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """Use this tool to execute Python code. The code runs in a stateful Jupyter notebook. Use print() to see output."""

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name=self.get_tool_name())
        message = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message: Message, timeout: float | None = None) -> list[Message]:
        """Execute code from message using Jupyter kernel."""
        self._ensure_session()
        script = message.content[0].text
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(script, timeout=timeout)
            except TimeoutError as exc:
                output = f"[ERROR] {exc}"
            except Exception as exc:
                output = f"[ERROR] {exc}"
        return [self._make_response(output, channel=message.channel)]

    def close(self):
        if self._jupyter_session is not None:
            self._jupyter_session.close()
            self._jupyter_session = None

    def __del__(self):
        self.close()
```

# Imports and Setup

```python
import warnings
warnings.simplefilter('ignore')

import re
import math
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pandas as pd
import polars as pl
from openai import OpenAI
from transformers import set_seed, AutoTokenizer
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    ReasoningEffort,
    RenderConversationConfig,
)

from local_python_tool import PythonTool

# Load Harmony encoding for GPT-OSS
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Constants
SEED = 42
set_seed(SEED)
MAX_LEN = 64 * 1024
USE_BUDGET = False
K = 8  # Number of parallel samples

# Inference parameters (same as way-to-30 reference)
TEMPERATURE = 1.0
TOP_P = 1.0
MIN_P = 0.02
```

```python
class DynamicTimeBudget:
    """Manages dynamic time allocation with rollover from early stopping."""
    
    def __init__(self, total_time_seconds: float, num_questions: int, buffer_seconds: float = 60):
        self.total_time = total_time_seconds
        self.num_questions = num_questions
        self.buffer = buffer_seconds
        self.start_time = time.time()
        
        # Available time excluding buffer
        self.available_time = total_time_seconds - buffer_seconds
        
        # Track time usage
        self.time_used = 0
        self.questions_completed = 0
        self.time_saved = 0  # Accumulated time from early stops
        
    def get_deadline_for_question(self) -> float:
        """Calculate deadline for current question with rollover time."""
        questions_remaining = self.num_questions - self.questions_completed
        
        if questions_remaining <= 0:
            return time.time() + 60  # Emergency fallback
        
        # Base time per remaining question
        time_remaining = self.available_time - self.time_used
        base_time = time_remaining / questions_remaining
        
        # Add any saved time from early stopping
        allocated_time = base_time + self.time_saved
        
        # Reset saved time (it's now allocated to this question)
        self.time_saved = 0
        
        deadline = time.time() + allocated_time
        
        print(f"⏱️  Allocated {allocated_time:.1f}s for question {self.questions_completed + 1}")
        print(f"   (Base: {base_time:.1f}s, Rollover: {self.time_saved:.1f}s, Remaining: {questions_remaining} questions)")
        
        return deadline
    
    def record_question_completion(self, time_spent: float, early_stopped: bool = False):
        """Record completion and calculate time savings."""
        self.time_used += time_spent
        self.questions_completed += 1
        
        # If early stopped, calculate how much time was saved
        if early_stopped:
            questions_remaining = self.num_questions - self.questions_completed
            if questions_remaining > 0:
                expected_time = (self.available_time - self.time_used + time_spent) / (questions_remaining + 1)
                time_saved = max(0, expected_time - time_spent)
                self.time_saved += time_saved
                print(f"💰 Early stop saved {time_saved:.1f}s (total saved: {self.time_saved:.1f}s)")
```

# Start vLLM Server

```python
def start_vllm_server() -> subprocess.Popen:
    """Start vLLM server in background."""
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/kaggle/input/gpt-oss-120b/transformers/default/1",
        "--served-model-name", "gpt-oss",
        "--tensor-parallel-size", "1",
        "--max-num-seqs", "64",
        "--gpu-memory-utilization", "0.96",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "auto",
        "--max-model-len", str(MAX_LEN),
        "--stream-interval", "20",
    ]
    with open("./vllm.log", "w") as logfile:
        process = subprocess.Popen(
            command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
        )
    print("vLLM server started. Logs: ./vllm.log")
    return process


vllm_process = start_vllm_server()
```

# TIR Prompts

```python
# Option A: Exact same as way-to-30 (proven 30/50 on LB)
TIR_PROMPT_SIMPLE0 = """You are an elite olympiad mathematician solving a national/international-level problem with full rigor; reason carefully, justify all nontrivial steps, explore multiple solution strategies when helpful, check edge cases, and use Python tool for computation or verification if needed, then return only the final verified answer in \boxed{n}, where n ∈ [0,99999], and never guess."""

# Use simple version (same as way-to-30) - change to TIR_PROMPT_ENHANCED if needed
# TIR_PROMPTS = [TIR_PROMPT_SIMPLE]
```

```python
# Option A: Exact same as way-to-30 (proven 30/50 on LB)
TIR_PROMPT_SIMPLE2 = """Please reason step by step and use the python tool to solve the math problem.
Finally, Return only the verified final answer in \\boxed{}, where the answer is an integer in [0, 99999]. Never guess."""


# Use both prompts to encourage diverse reasoning (simple + enhanced)
TIR_PROMPTS = [TIR_PROMPT_SIMPLE2]
```

# Inferencer with Harmony Protocol

```python
import queue
from local_python_tool import PythonTool

python_pool = queue.Queue(maxsize=K)

for _ in range(K):
    t = PythonTool(execution_backend="jupyter", local_jupyter_timeout=60.0)
    python_pool.put(t)
print("Pool created!")
```

```python
import gc

CLEANUP_CODE = r"""
import gc
_keep = {
    "__builtins__", "__name__", "__doc__", "__package__", "__loader__", "__spec__",
    "np", "sp", "math",
}
g = globals()
for k in list(g.keys()):
    if k in _keep or k.startswith("_"):
        continue
    try:
        del g[k]
    except Exception:
        pass
gc.collect()
"""
print("yes")
```

```python
class HarmonyTIRInferencer:
    """Inferencer using Harmony protocol with Tool-Integrated Reasoning (TIR)."""

    def __init__(
        self,
        model_path: str,
        max_model_len: int = MAX_LEN,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        min_p: float = MIN_P,
        seed: int = SEED,
        k: int = K,
        use_budget: bool = USE_BUDGET,
        max_iter: int = 100,
    ):
        self.model_path = model_path
        self.model = "gpt-oss"
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.seed = seed
        self.k = k
        self.use_budget = use_budget
        self.max_iter = max_iter
        self.base_budget = 60 * 5.5  # 5.5 minutes base per problem
        self.budget = 370              # initial budget in seconds (~6.1 min for first problem)
        self.deadline = None

        # Initialize the OpenAI-compatible client pointing to local vLLM server
        self.client = OpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="sk-local",
            timeout=360,
        )
        self.stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def wait_server(self):
        """Wait until the vLLM server is ready to accept requests."""
        for _ in range(15 * 60):
            time.sleep(1)
            try:
                # List models to check if server is up
                print(self.client.models.list())
                return
            except Exception:
                continue
        raise RuntimeError("vLLM server failed to start")

    def get_num_samples(self) -> int:
        """Determine number of parallel samples to generate based on remaining budget."""
        if not self.use_budget:
            print(f"Budget disabled -> N: {self.k}")
            return self.k
        else:
            return self.k
            
    def apply_chat_template(self, prompt: str, python_tool: PythonTool) -> list[Message]:
        """Wrap user prompt into Harmony conversation format with system and tool info."""
        return [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                .with_tools(python_tool.tool_config)
            ),
            Message.from_role_and_content(Role.USER, prompt),
        ]

    def format_prompts(self, problem: str) -> list[str]:
        """Create multiple prompts (possibly with different TIR strategies) for one problem."""
        num_samples = self.get_num_samples()
        prompts = []
        for i in range(num_samples):
            # Alternate between the prompt templates for diversity
            tir_prompt = TIR_PROMPTS[i % len(TIR_PROMPTS)]
            prompts.append(problem + "\n\n" + tir_prompt)
        return prompts

    def inference(self, problem: str, deadline: float) -> tuple[int, float]:
        """Run the multi-sample inference for a single problem and return the final answer and saved time."""
        self.deadline = deadline
        start_time = time.time()
    
        prompts = self.format_prompts(problem)
        responses = self._inference_parallel(prompts)
    
        duration = time.time() - start_time
        saved_time = max(0.0, deadline - time.time())
    
        print(f"[Budget]: {(deadline - start_time):.2f}s")
        print(f"[inference] Took {duration:.2f}s")
        print(f"[Saved time]: {saved_time:.2f}s")
    
        return self.parse_responses(responses), saved_time

    
    def single_generate_tir(self, prompt: str, stop_event: threading.Event, seed_offset: int = 0) -> str:
        """Generate single TIR response with tool execution (dynamic timeouts)."""
        python_tool = None
    
        def _compute_req_timeout() -> float:
            # For vLLM request timeout
            CUSHION = 0.5
            MAX_REQ_TIMEOUT = 30.0
            MIN_ALLOW = 0.2
    
            if not getattr(self, "deadline", None):
                return MAX_REQ_TIMEOUT
    
            remaining = self.deadline - time.time()
            if remaining <= 0:
                return 0.0
    
            t = remaining - CUSHION
            if t <= 0:
                return 0.0
    
            return min(MAX_REQ_TIMEOUT, max(MIN_ALLOW, t))
    
        def _compute_py_timeout() -> float:
            # For python tool timeout
            PY_CUSHION = 1.0
            MAX_PY_TIMEOUT = 15.0
            MIN_ALLOW = 0.2
    
            if not getattr(self, "deadline", None):
                return MAX_PY_TIMEOUT
    
            remaining = self.deadline - time.time()
            t = remaining - PY_CUSHION
            if t <= 0:
                return 0.0
    
            return min(MAX_PY_TIMEOUT, max(MIN_ALLOW, t))
    
        try:
            # Use pool instead of creating new PythonTool
            try:
                python_tool = python_pool.get(timeout=30.0)
            except queue.Empty:
                print("⚠️ Failed to get python_tool from pool, creating new one")
                python_tool = PythonTool(execution_backend="jupyter")
                try:
                    python_tool._ensure_session()
                except Exception as e:
                    print(f"⚠️ python session init failed: {e}")
                    if python_tool is not None:
                        try:
                            python_tool.close()
                        except Exception:
                            pass
                    return ""
            else:
                # Verify session is still alive
                try:
                    if python_tool._jupyter_session is None:
                        python_tool._ensure_session()
                    # Quick health check: try to execute a simple command
                    test_output = python_tool._jupyter_session.execute("1+1", timeout=2.0)
                    if "[ERROR]" in test_output or "Traceback" in test_output:
                        # Session is broken, recreate it
                        try:
                            python_tool.close()
                        except Exception:
                            pass
                        python_tool._jupyter_session = None
                        python_tool._ensure_session()
                except Exception as e:
                    print(f"⚠️ python session health check failed: {e}, recreating")
                    try:
                        python_tool.close()
                    except Exception:
                        pass
                    python_tool._jupyter_session = None
                    try:
                        python_tool._ensure_session()
                    except Exception as e2:
                        print(f"⚠️ python session recreate failed: {e2}")
                        try:
                            python_pool.put(python_tool, block=False)
                        except queue.Full:
                            pass
                        return ""
    
            messages = self.apply_chat_template(prompt, python_tool)
            final_answer_found = ""
    
            for iteration in range(self.max_iter):
                # termination checks
                if stop_event and stop_event.is_set():
                    print("🛑 Stop signal received")
                    break
                if getattr(self, "deadline", None) and time.time() >= self.deadline:
                    print("⏰ Deadline reached")
                    break
                if final_answer_found:
                    break
    
                prompt_ids = encoding.render_conversation_for_completion(
                    Conversation.from_messages(messages), Role.ASSISTANT
                )
                max_tokens = self.max_model_len - len(prompt_ids)
                if max_tokens < 1:
                    print("⚠️ Context full")
                    break
    
                req_timeout = _compute_req_timeout()
                if req_timeout <= 0:
                    print("⏰ Not enough remaining time for vLLM request")
                    break
    
                token_buffer: list[int] = []
                token_buffer_str = ""
                breaking = False
    
                stream = None
                try:
                    stream = self.client.completions.create(
                        model=self.model,
                        prompt=prompt_ids,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        seed=self.seed + seed_offset,
                        stream=True,
                        extra_body=dict(
                            min_p=self.min_p,
                            stop_token_ids=self.stop_token_ids,
                            return_token_ids=True,
                        ),
                        timeout=req_timeout,
                    )
    
                    for chunk in stream:
                        try:
                            if stop_event and stop_event.is_set():
                                breaking = True
                                break
                            if getattr(self, "deadline", None) and time.time() >= self.deadline:
                                breaking = True
                                break
                            
                            # Safely extract chunk data
                            if not chunk.choices or len(chunk.choices) == 0:
                                continue
                            
                            choice = chunk.choices[0]
                            token_chunk = getattr(choice, 'token_ids', None) or []
                            text_chunk = getattr(choice, 'text', '') or ''
    
                            if token_chunk:
                                token_buffer.extend(token_chunk)
                                token_buffer_str += text_chunk
    
                            if len(token_buffer) > 60_000:
                                print("⚠️ Token limit")
                                breaking = True
                                break
    
                            # early stop when boxed appears
                            if "}" in text_chunk and self.extract_boxed_text(token_buffer_str) is not None:
                                final_answer_found = token_buffer_str
                                breaking = True
                                break
                        except StopIteration:
                            # Stream ended normally
                            break
                        except Exception as e:
                            print(f"⚠️ Error processing stream chunk: {e}")
                            # Continue processing, but mark as potentially broken
                            break
    
                except Exception as e:
                    print(f"⚠️ Error creating/reading stream: {e}")
                    breaking = True
                finally:
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
                        # Additional cleanup attempt
                        try:
                            del stream
                        except Exception:
                            pass
    
                if breaking:
                    break
    
                if not token_buffer:
                    continue
    
                # parse completion
                try:
                    new_messages = encoding.parse_messages_from_completion_tokens(
                        token_buffer, Role.ASSISTANT
                    )
                except Exception as e:
                    print(f"Error parsing completion: {e}")
                    break
    
                messages.extend(new_messages)
                last_message = messages[-1]
    
                if last_message.channel == "final" or token_buffer[-1] == 200002:
                    break
    
                if last_message.recipient == "python":
                    if stop_event and stop_event.is_set():
                        break
                    if getattr(self, "deadline", None) and time.time() >= self.deadline:
                        break
    
                    py_timeout = _compute_py_timeout()
                    if py_timeout <= 0 or py_timeout < 0.5:
                        print(f"⏰ Not enough remaining time for python ({py_timeout:.2f}s)")
                        break
    
                    print("🐍 Executing Python code...")
                    try:
                        response_msgs = python_tool.process_sync_plus(last_message, timeout=py_timeout)
                    except Exception as e:
                        # treat any python tool failure as terminal for this sample
                        print(f"⚠️ python tool failed: {e}")
                        break
    
                    messages.extend(response_msgs)
    
            if final_answer_found:
                return final_answer_found
    
            return encoding.decode_utf8(
                encoding.render_conversation_for_training(
                    Conversation.from_messages(messages),
                    RenderConversationConfig(auto_drop_analysis=False),
                )
            )
    
        except KeyboardInterrupt:
            # never swallow manual interrupts
            raise
        except Exception as e:
            import traceback
            print(f"Error in generation: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return ""
        finally:
            # Return tool to pool instead of closing it
            if python_tool is not None:
                try:
                    # Only return to pool if we got it from pool
                    # Check if tool is still healthy before returning
                    if python_tool._jupyter_session is not None:
                        try:
                            # Quick health check
                            test_output = python_tool._jupyter_session.execute("1+1", timeout=1.0)
                            if "[ERROR]" not in test_output and "Traceback" not in test_output:
                                # Tool is healthy, return to pool
                                try:
                                    python_pool.put(python_tool, block=False)
                                except queue.Full:
                                    # Pool is full, close the tool
                                    python_tool.close()
                            else:
                                # Tool is broken, close it
                                python_tool.close()
                        except Exception:
                            # Health check failed, close the tool
                            try:
                                python_tool.close()
                            except Exception:
                                pass
                    else:
                        # No session, safe to return to pool
                        try:
                            python_pool.put(python_tool, block=False)
                        except queue.Full:
                            pass
                except Exception as e:
                    # If anything goes wrong, try to close the tool
                    try:
                        python_tool.close()
                    except Exception:
                        pass
            

    def _inference_parallel(self, prompts: list[str]) -> list[str]:
        """Run multiple `single_generate_tir` in parallel and return all raw responses."""
        stop_event = threading.Event()
        answers_collected: List[int] = []
        raw_responses = [""] * len(prompts)
        majority_threshold = len(prompts) / 2  # more than half of the samples
    
        print(f"🚀 Sampling {len(prompts)} times (threshold: > {majority_threshold})...")
    
        executor = ThreadPoolExecutor(max_workers=self.k)
        futures = []
        future_to_idx = {}
        try:
            for i, p in enumerate(prompts):
                fut = executor.submit(self.single_generate_tir, p, stop_event, i)
                futures.append(fut)
                future_to_idx[fut] = i
    
            completed_count = 0
            for fut in as_completed(futures):
                idx = future_to_idx.get(fut, -1)
                if idx < 0:
                    continue
                    
                try:
                    result_text = fut.result(timeout=1.0)
                except Exception as e:
                    import traceback
                    print(f"Task exception for idx {idx}: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    result_text = ""
                
                raw_responses[idx] = result_text
                completed_count += 1
    
                ans = self.extract_boxed_text(result_text)
                if ans is not None:
                    answers_collected.append(ans)
                    counts = Counter(answers_collected)
                    if counts:
                        most_common_ans, count = counts.most_common(1)[0]
    
                        if count > majority_threshold:
                            print(f"🎯 Majority reached! {most_common_ans} appeared {count} times")
                            stop_event.set()
    
                            # best-effort: cancel those not started yet
                            for f in futures:
                                if f is not fut and not f.done():
                                    try:
                                        f.cancel()
                                    except Exception:
                                        pass
                            break
    
        except Exception as e:
            import traceback
            print(f"Error in _inference_parallel: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        finally:
            stop_event.set()
            # Ensure all futures are handled
            for fut in futures:
                if not fut.done():
                    try:
                        fut.cancel()
                    except Exception:
                        pass
            
            # Shutdown executor with timeout protection
            try:
                # Python 3.9+ supports timeout, but we'll use a workaround for compatibility
                import sys
                if sys.version_info >= (3, 9):
                    executor.shutdown(wait=True, timeout=60.0, cancel_futures=True)
                else:
                    # For older Python versions, use wait without timeout
                    executor.shutdown(wait=True)
            except TypeError:
                # timeout parameter not supported, use without it
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    executor.shutdown(wait=False)
            except Exception as e:
                print(f"Warning: executor shutdown had issues: {e}")
                # Force shutdown
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass
    
        return raw_responses


    def extract_boxed_text(self, text: str) -> int | None:
        """Extract a numeric answer from '\\boxed{}' or 'final answer is ...' in the text."""
        # Pattern for \boxed{NUMBER}
        pattern = r'oxed{(.*?)}'
        matches = re.findall(pattern, str(text))
        if matches:
            for match in reversed(matches):
                if match:
                    try:
                        # Remove commas/spaces and parse as number (float covers scientific notation if any)
                        clean_match = match.strip().replace(',', '').replace(' ', '')
                        val = int(float(clean_match[:20]))
                        if 0 <= val <= 99999:
                            return val
                    except Exception:
                        pass

        # Pattern for "final answer is X" or "Final Answer: X"
        pattern = r'(?i)final\s+answer\s*(?:is|:)?\s*(\d+)'
        matches = re.findall(pattern, text)
        if matches:
            for match in reversed(matches):
                if match:
                    try:
                        val = int(match)
                        if 0 <= val <= 99999:
                            return val
                    except Exception:
                        pass

        return None

    def parse_responses(self, responses: list[str]) -> int:
        """Decide on the final answer from all responses by majority vote (with tie-break)."""
        answers = [self.extract_boxed_text(r) for r in responses]

        # Filter out any None values (cases where no answer was extracted)
        valid_answers = [a for a in answers if a is not None]
        if not valid_answers:
            print("No valid answers found")
            return 8687

        counter = Counter(valid_answers)
        print(f"Answers: {counter}")

        # Majority vote: pick the most common answer; break ties by choosing the largest answer
        most_common_list = counter.most_common(2)
        if len(most_common_list) > 1 and most_common_list[0][1] == most_common_list[1][1]:
            tied_answers = [ans for ans, cnt in counter.items() if cnt == most_common_list[0][1]]
            answer = max(tied_answers)
        else:
            answer = most_common_list[0][0]
        return answer
```

```python
# time_budget_manager = DynamicTimeBudget(TOTAL_TIME, NUM_QUESTIONS, BUFFER_TIME)
```

```python
# Initialize the inferencer with the model path and parameters
inferencer = HarmonyTIRInferencer(
    "/kaggle/input/gpt-oss-120b/transformers/default/1",
    use_budget=USE_BUDGET,
    k=K,
)
```

```python
# inferencer.time_budget_manager = time_budget_manager
```

```python
inferencer.wait_server()
```

# Submission

```python
init_time = time.time()
cutoff_times = [int(x) for x in np.linspace(final_cutoff_time, init_time, 50 + 1)]
cutoff_times.pop()
```

```python
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    global correct_count, total_count, predictions, cutoff_times
    
    question_id = id_.item(0)
    question_text = question.item(0)

    print("------")
    print(f"ID: {question_id}")
    print(f"Question: {question_text[:200]}...")

    current_deadline = cutoff_times[-1]
    answer,saved_time = inferencer.inference(question_text, deadline=current_deadline)
    cutoff_times.pop()

    # ⏱️ Dynamically recompute cutoff_times and distribute saved_time
    if len(cutoff_times) > 0:
        now = time.time()
        num_remaining = len(cutoff_times)
        base_times = np.linspace(final_cutoff_time, now, num_remaining + 1)
        base_times = base_times[:-1]  # keep only N timestamps
        extra = saved_time / num_remaining
        cutoff_times = [int(t + extra) for t in base_times]

    # Store prediction
    predictions[question_id] = answer
    
    # Check accuracy if ground truth available
    total_count += 1
    if question_id in ground_truth:
        gt = ground_truth[question_id]
        is_correct = (answer == gt)
        if is_correct:
            correct_count += 1
        status = "✅" if is_correct else "❌"
        print(f"Answer: {answer} | Ground Truth: {gt} | {status}")
        print(f"📊 Running Accuracy: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    else:
        print(f"Answer: {answer}")
    
    print("------\n")

    return pl.DataFrame({"id": question_id, "answer": answer})
```

```python
# Load reference data and keep ground truth for accuracy calculation
df = pd.read_csv(
    "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
)

# Store ground truth answers for accuracy calculation (only in local mode)
ground_truth = dict(zip(df["id"], df["answer"])) if "answer" in df.columns else {}

# Create input file without answers
df.drop("answer", axis=1, errors="ignore").to_csv("reference.csv", index=False)

# Track predictions for accuracy calculation
predictions = {}
correct_count = 0
total_count = 0
```

```python
import kaggle_evaluation.aimo_3_inference_server

inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(("reference.csv",))
    
    # Print final accuracy summary
    if ground_truth and total_count > 0:
        print("\n" + "=" * 50)
        print("📊 FINAL ACCURACY SUMMARY")
        print("=" * 50)
        print(f"Correct: {correct_count}/{total_count}")
        print(f"Accuracy: {100*correct_count/total_count:.1f}%")
        print("=" * 50)
        
        # Show details
        print("\nDetails:")
        for qid, pred in predictions.items():
            if qid in ground_truth:
                gt = ground_truth[qid]
                status = "✅" if pred == gt else "❌"
                print(f"  {qid}: pred={pred}, gt={gt} {status}")
```