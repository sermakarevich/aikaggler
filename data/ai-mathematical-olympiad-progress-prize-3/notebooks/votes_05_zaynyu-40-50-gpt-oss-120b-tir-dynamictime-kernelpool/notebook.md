# [40/50]GPT-OSS-120B TIR+DynamicTime+KernelPool

- **Author:** ZaynYu
- **Votes:** 615
- **Ref:** zaynyu/40-50-gpt-oss-120b-tir-dynamictime-kernelpool
- **URL:** https://www.kaggle.com/code/zaynyu/40-50-gpt-oss-120b-tir-dynamictime-kernelpool
- **Last run:** 2026-01-07 07:01:28.057000

---

```python
%pip uninstall --yes 'keras' 'matplotlib' 'scikit-learn' 'tensorflow'
```

```python
import warnings
warnings.simplefilter('ignore')
```

```python
import os
import sys
import subprocess
```

```python
subprocess.run(['ls', '/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings'])
```

```python
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings'
```

```python
import gc
import re
import math
import time
import queue
import threading
import contextlib
from typing import Optional
from jupyter_client import KernelManager
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

import numpy as np
import pandas as pd
import polars as pl

from openai import OpenAI

from openai_harmony import (
    HarmonyEncodingName, 
    load_harmony_encoding, 
    SystemContent, 
    ReasoningEffort, 
    ToolNamespaceConfig, 
    Author, 
    Message, 
    Role, 
    TextContent, 
    Conversation
)

from transformers import set_seed
import kaggle_evaluation.aimo_3_inference_server
```

# Config

```python
class CFG:
    """Configuration for AIMO-3 solver."""

    # Prompts
    system_prompt = (
        'You are a world-class International Mathematical Olympiad (IMO) competitor. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'You must place the final integer answer inside \\boxed{}.'
    )
    
    tool_prompt = (
        'Use this tool to execute Python code. '
        'The environment is a stateful Jupyter notebook. '
        'You must use print() to output results.'
    )
    
    preference_prompt = (
        'Please reason step by step and use the python tool to solve the math problem. '
        'For extremely large numbers, find patterns from small cases instead of direct computation. '
        'Finally, Return only the verified final answer in \\boxed{}, where the answer is an integer in [0, 99999]. Never guess.'
    )

    # Model configuration
    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'

    # Timeout configuration (seconds)
    notebook_limit = (4 * 60 + 55) * 60  # 4h 55m
    high_problem_timeout = 900
    base_problem_timeout = 300
    server_timeout = 180
    session_timeout = 960
    jupyter_timeout = 15
    sandbox_timeout = 5

    # Generation parameters
    stream_interval = 200
    context_tokens = 65536
    buffer_tokens = 512
    search_window = 256
    batch_size = 256
    early_stop = 4  # Majority threshold
    attempts = 8
    workers = 16
    turns = 100
    seed = 43

    # Sampling parameters
    gpu_memory_utilization = 0.96
    temperature = 1.0
    top_p = 1.0
    min_p = 0.02
    
    # Answer validation
    min_answer = 0
    max_answer = 99999
    default_answer = 8687
```

```python
set_seed(CFG.seed)
```

# Template

```python
class AIMO3Template:
    """Template builder for chat messages with tool configuration."""

    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:
        """Build system content with model identity, reasoning effort, and tools."""
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        tool_config: ToolNamespaceConfig
    ) -> list[Message]:
        """Create initial message list with system and user messages."""
        system_content = self.get_system_content(system_prompt, tool_config)        
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]
```

# Jupiter Kernal Pool

```python
class AIMO3Sandbox:
    """Jupyter kernel sandbox for executing Python code in isolation."""

    _port_lock = threading.Lock()
    _next_port = 50000
    _max_port = 65535

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        """Allocate unique ports for kernel communication."""
        with cls._port_lock:
            ports = []
            start_port = cls._next_port
            
            for i in range(count):
                port = start_port + i
                if port > cls._max_port:
                    start_port = 50000
                    port = start_port + i
                ports.append(port)
            
            cls._next_port = start_port + count
            if cls._next_port > cls._max_port:
                cls._next_port = 50000
            
            return ports

    def __init__(self, timeout: float):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None
        self._closed = False
        
        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        # Initialize with common math libraries
        self.execute(
            'import math\n'
            'import numpy as np\n'
            'import sympy as sp\n'
            'from sympy import *\n'
            'import itertools\n'
            'import collections\n'
        )

    def _format_error(self, traceback: list[str]) -> str:
        """Clean up traceback for display."""
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return ''.join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:
        """Execute code and return combined stdout/stderr."""
        if self._closed or self._client is None:
            raise RuntimeError("Session has been closed")

        client = self._client
        effective_timeout = timeout or self._default_timeout
        
        msg_id = client.execute(
            code, 
            store_history=True, 
            allow_stdin=False, 
            stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > effective_timeout:
                try:
                    self._km.interrupt_kernel()
                except Exception:
                    pass
                return f'[ERROR] Execution timed out after {effective_timeout:.1f}s'

            try:
                msg = client.get_iopub_msg(timeout=0.5)
            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')
                if content.get('name') == 'stdout':
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)

            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])
                stderr_parts.append(self._format_error(traceback_list))

            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')
                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')

            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr

        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def is_alive(self) -> bool:
        """Check if the kernel session is still alive."""
        if self._closed or self._client is None:
            return False
        try:
            if self._km is not None:
                return self._km.is_alive()
            return True
        except Exception:
            return False

    def reset(self):
        """Reset kernel state."""
        self.execute(
            '%reset -f\n'
            'import math\n'
            'import numpy as np\n'
            'import sympy as sp\n'
            'from sympy import *\n'
            'import itertools\n'
            'import collections\n'
        )

    def close(self):
        """Close the session and cleanup resources."""
        if self._closed:
            return
        
        self._closed = True
        
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

        self._client = None
        self._km = None

    def __del__(self):
        self.close()
```

# Python Executor

```python
class AIMO3Tool:
    """Python code execution tool using Jupyter kernel sandbox."""

    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        """Lazily initialize the Jupyter session if not provided."""
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:
        """Wrap the last expression in print() if it's not already printing."""
        lines = code.strip().split('\n')
        
        if not lines:
            return code

        last_line = lines[-1].strip()

        # Skip if already has print, import, empty, or comment
        if not last_line or last_line.startswith('#'):
            return code
        if 'print' in last_line or 'import' in last_line:
            return code
        # Skip control flow statements
        if last_line.endswith(':') or last_line.startswith(('return', 'break', 'continue', 'pass', 'raise')):
            return code

        # Remove inline comment before wrapping
        if '#' in last_line:
            last_line = last_line.split('#')[0].strip()

        lines[-1] = f'print({last_line})'
        return '\n'.join(lines)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name='python', 
            description=self.instruction, 
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')

        if channel:
            message = message.with_channel(channel)

        return message

    def process_sync_plus(self, message: Message, timeout: float | None = None) -> list[Message]:
        """Execute code from message using Jupyter kernel."""
        # Validate message content
        if not message.content or len(message.content) == 0:
            return [self._make_response('[ERROR] Message has no content', channel=message.channel)]
        
        try:
            script = message.content[0].text
            if not script or not script.strip():
                return [self._make_response('[ERROR] Empty script provided', channel=message.channel)]
        except (AttributeError, IndexError, TypeError) as e:
            return [self._make_response(f'[ERROR] Failed to extract script: {e}', channel=message.channel)]

        self._ensure_session()
        final_script = self._ensure_last_print(script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script, timeout=timeout)
            except TimeoutError as exc:
                output = f'[ERROR] Execution timeout: {exc}'
            except RuntimeError as exc:
                output = f'[ERROR] Runtime error: {exc}'
            except Exception as exc:
                output = f'[ERROR] Unexpected error: {exc}'

        return [self._make_response(output, channel=message.channel)]

    def close(self):
        """Close the Python tool and cleanup resources."""
        with self._init_lock:
            if self._jupyter_session is not None and self._owns_session:
                try:
                    self._jupyter_session.close()
                except Exception:
                    pass
                finally:
                    self._jupyter_session = None

    def __del__(self):
        self.close()
```

# Solver

```python
class AIMO3Solver:
    """Main solver for AIMO-3 competition using vLLM and tool-integrated reasoning."""

    def __init__(self, cfg, port: int = 8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://127.0.0.1:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self._preload_model_weights()
        self.server_process = self._start_server()

        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key, 
            timeout=self.cfg.session_timeout
        )

        self._wait_for_server()
        self._initialize_kernels()

        self.notebook_start_time = time.time()
        self.problems_remaining = 50

    def _preload_model_weights(self) -> None:
        """Pre-read model weight files into OS page cache."""
        print(f'Loading model weights from {self.cfg.model_path} into OS Page Cache...')
        start_time = time.time()
        
        files_to_load = []
        total_size = 0

        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)

        def _read_file(path: str) -> None:
            with open(path, 'rb') as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass

        with ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(_read_file, files_to_load))

        elapsed = time.time() - start_time
        print(f'Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')

    def _start_server(self) -> subprocess.Popen:
        """Start vLLM server in background."""
        cmd = [
            sys.executable, 
            '-m', 
            'vllm.entrypoints.openai.api_server', 
            '--seed', 
            str(self.cfg.seed), 
            '--model', 
            self.cfg.model_path, 
            '--served-model-name', 
            self.cfg.served_model_name, 
            '--tensor-parallel-size', 
            '1', 
            '--max-num-seqs', 
            str(self.cfg.batch_size), 
            '--gpu-memory-utilization', 
            str(self.cfg.gpu_memory_utilization), 
            '--host', 
            '0.0.0.0', 
            '--port', 
            str(self.port), 
            '--dtype', 
            self.cfg.dtype, 
            '--kv-cache-dtype', 
            self.cfg.kv_cache_dtype, 
            '--max-model-len', 
            str(self.cfg.context_tokens), 
            '--stream-interval', 
            str(self.cfg.stream_interval), 
            '--async-scheduling', 
            '--enable-prefix-caching'
        ]

        self.log_file = open('vllm_server.log', 'w')

        return subprocess.Popen(
            cmd, 
            stdout=self.log_file, 
            stderr=subprocess.STDOUT, 
            start_new_session=True
        )

    def _wait_for_server(self):
        """Wait until the vLLM server is ready to accept requests."""
        print('Waiting for vLLM server...')
        start_time = time.time()

        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()

            if return_code is not None:
                self.log_file.flush()
                with open('vllm_server.log', 'r') as log_file:
                    logs = log_file.read()
                raise RuntimeError(f'Server died with code {return_code}. Logs:\n{logs}\n')

            try:
                self.client.models.list()
                elapsed = time.time() - start_time
                print(f'Server is ready (took {elapsed:.2f} seconds).\n')
                return
            except Exception:
                time.sleep(1)

        raise RuntimeError('Server failed to start (timeout).\n')

    def _initialize_kernels(self) -> None:
        """Initialize pool of Jupyter kernels."""
        print(f'Initializing {self.cfg.workers} persistent Jupyter kernels...')
        start_time = time.time()

        self.sandbox_pool = queue.Queue()

        def _create_sandbox():
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]
            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())

        elapsed = time.time() - start_time
        print(f'Kernels initialized in {elapsed:.2f} seconds.\n')

    def _scan_for_answer(self, text: str) -> int | None:
        """Extract the last valid \\boxed{} answer from text."""
        if text is None:
            return None
        text = str(text)
        
        # Pattern for \boxed{NUMBER}
        pattern = r'oxed\s*\{\s*([0-9,\s]+)\s*\}'
        matches = re.findall(pattern, text)

        for match in reversed(matches):
            try:
                clean_value = match.replace(',', '').replace(' ', '')
                value = int(float(clean_value[:20]))
                if self.cfg.min_answer <= value <= self.cfg.max_answer:
                    return value
            except (ValueError, TypeError):
                continue

        return None

    def _process_attempt(
        self, 
        problem: str, 
        system_prompt: str, 
        attempt_index: int, 
        stop_event: threading.Event, 
        deadline: float
    ) -> dict:
        """Process a single attempt to solve the problem."""
        empty_result = {
            'Attempt': attempt_index + 1, 
            'Answer': None, 
            'Python Calls': 0, 
            'Python Errors': 0, 
            'Response Length': 0
        }

        if stop_event.is_set() or time.time() > deadline:
            return empty_result

        local_tool = None
        sandbox = None
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer = None

        attempt_seed = self.cfg.seed + attempt_index

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)

            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout, 
                tool_prompt=self.cfg.tool_prompt, 
                sandbox=sandbox
            )

            encoding = self.encoding
            messages = self.template.apply_chat_template(
                system_prompt, 
                problem, 
                local_tool.tool_config
            )

            conversation = Conversation.from_messages(messages)

            for turn in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < self.cfg.buffer_tokens:
                    print(f'⚠️ Attempt {attempt_index + 1}: Context full!')
                    break

                try:
                    stream = self.client.completions.create(
                        model=self.cfg.served_model_name, 
                        temperature=self.cfg.temperature, 
                        max_tokens=max_tokens, 
                        prompt=prompt_ids, 
                        seed=attempt_seed, 
                        stream=True, 
                        extra_body={
                            'min_p': self.cfg.min_p, 
                            'stop_token_ids': self.stop_token_ids, 
                            'return_token_ids': True
                        }
                    )
                except Exception as e:
                    print(f'⚠️ Attempt {attempt_index + 1}: Stream creation failed: {e}')
                    break

                try:
                    token_buffer = []
                    text_chunks = []

                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break

                        if not chunk.choices or len(chunk.choices) == 0:
                            continue

                        new_tokens = getattr(chunk.choices[0], 'token_ids', None) or []
                        new_text = getattr(chunk.choices[0], 'text', '') or ''

                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)

                        if '}' in new_text:
                            search_text = ''.join(text_chunks[-self.cfg.search_window:])
                            answer = self._scan_for_answer(search_text)
                            if answer is not None:
                                final_answer = answer
                                break

                finally:
                    try:
                        stream.close()
                    except Exception:
                        pass

                if final_answer is not None:
                    break

                if not token_buffer:
                    break

                try:
                    new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                    if not new_messages:
                        break
                    conversation.messages.extend(new_messages)
                    last_message = new_messages[-1]
                except Exception as e:
                    print(f'⚠️ Attempt {attempt_index + 1}: Parse error: {e}')
                    break

                if last_message.channel == 'final':
                    answer_text = last_message.content[0].text
                    final_answer = self._scan_for_answer(answer_text)
                    break

                if last_message.recipient == 'python':
                    python_calls += 1
                    
                    # Calculate remaining time for python execution
                    remaining = deadline - time.time()
                    py_timeout = min(self.cfg.jupyter_timeout, max(1.0, remaining - 1.0))
                    
                    if py_timeout < 1.0:
                        break
                    
                    tool_responses = local_tool.process_sync_plus(last_message, timeout=py_timeout)
                    response_text = tool_responses[0].content[0].text

                    if response_text.startswith('[ERROR]') or 'Traceback' in response_text or 'Error:' in response_text:
                        python_errors += 1

                    conversation.messages.extend(tool_responses)

        except queue.Empty:
            print(f'⚠️ Attempt {attempt_index + 1}: Sandbox acquisition timeout')
            return empty_result
        except Exception as e:
            print(f'⚠️ Attempt {attempt_index + 1}: Exception: {e}')
            python_errors += 1

        finally:
            if sandbox is not None:
                try:
                    sandbox.reset()
                    self.sandbox_pool.put(sandbox)
                except Exception:
                    pass

        return {
            'Attempt': attempt_index + 1, 
            'Response Length': total_tokens, 
            'Python Calls': python_calls, 
            'Python Errors': python_errors, 
            'Answer': final_answer
        }

    def _select_answer(self, detailed_results: list) -> int:
        """Select final answer using majority voting with Python calls as tiebreaker."""
        stats = defaultdict(lambda: {'votes': 0, 'calls': 0})

        for result in detailed_results:
            answer = result['Answer']
            if answer is not None:
                stats[answer]['votes'] += 1
                stats[answer]['calls'] += result['Python Calls']

        if not stats:
            print(f'\nNo valid answers found. Returning default: {self.cfg.default_answer}\n')
            return self.cfg.default_answer

        # Sort by votes (primary), then by calls (secondary)
        sorted_stats = sorted(
            stats.items(), 
            key=lambda item: (item[1]['votes'], item[1]['calls']), 
            reverse=True
        )

        vote_data = [(answer, data['votes'], data['calls']) for answer, data in sorted_stats]
        vote_dataframe = pd.DataFrame(vote_data, columns=['Answer', 'Votes', 'Calls'])
        print(vote_dataframe.to_string())

        final_answer = sorted_stats[0][0]
        final_votes = sorted_stats[0][1]['votes']
        final_calls = sorted_stats[0][1]['calls']

        print(f'\n✅ Final Result: {final_answer} | Votes: {final_votes} | Calls: {final_calls}\n')
        return final_answer
        
    def solve_problem(self, problem: str) -> int:
        """Solve a math problem using multiple parallel attempts with majority voting."""
        problem_start_time = time.time()
        
        print(f'\n{"="*60}')
        print(f'Problem: {problem[:200]}...' if len(problem) > 200 else f'Problem: {problem}')
        print(f'{"="*60}\n')
        
        user_input = f'{problem}\n\n{self.cfg.preference_prompt}'

        # Calculate time budget
        elapsed_global = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed_global
        problems_left_others = max(0, self.problems_remaining - 1)
        reserved_time = problems_left_others * self.cfg.base_problem_timeout

        budget = max(self.cfg.base_problem_timeout, 
                     min(time_left - reserved_time, self.cfg.high_problem_timeout))
        deadline = time.time() + budget

        print(f'⏱️ Budget: {budget:.2f}s | Problems remaining: {self.problems_remaining}\n')

        detailed_results = []
        valid_answers = []
        stop_event = threading.Event()

        executor = ThreadPoolExecutor(max_workers=self.cfg.workers)

        try:
            futures = []
            for attempt_index in range(self.cfg.attempts):
                future = executor.submit(
                    self._process_attempt, 
                    user_input, 
                    self.cfg.system_prompt, 
                    attempt_index, 
                    stop_event, 
                    deadline
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    detailed_results.append(result)

                    if result['Answer'] is not None:
                        valid_answers.append(result['Answer'])
                        print(f"🎯 Attempt {result['Attempt']}: Answer={result['Answer']}, Calls={result['Python Calls']}")

                    # Early stop conditions:
                    # 1. Majority threshold reached
                    # 2. All but one worker finished (no need to wait for the last one)
                    counts = Counter(valid_answers).most_common(1)
                    should_stop = False
                    stop_reason = ""
                    
                    if counts and counts[0][1] >= self.cfg.early_stop:
                        should_stop = True
                        stop_reason = f'{counts[0][0]} has {counts[0][1]} votes (majority)'
                    elif len(detailed_results) >= self.cfg.attempts - 1:
                        should_stop = True
                        stop_reason = f'{len(detailed_results)}/{self.cfg.attempts} workers done'
                    
                    if should_stop:
                        print(f'\n🚀 Early stop: {stop_reason}\n')
                        stop_event.set()
                        for f in futures:
                            f.cancel()
                        break

                except Exception as exc:
                    print(f'⚠️ Future failed: {exc}')
                    continue

        finally:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self.problems_remaining = max(0, self.problems_remaining - 1)

        if detailed_results:
            results_dataframe = pd.DataFrame(detailed_results)
            results_dataframe['Answer'] = results_dataframe['Answer'].astype('Int64')
            print('\n📊 Results Summary:')
            print(results_dataframe.to_string())
            print()

        if not valid_answers:
            inference_time = time.time() - problem_start_time
            print(f'\n❌ No valid answers found. Returning default: {self.cfg.default_answer}')
            print(f'⏱️ Inference time: {inference_time:.2f}s\n')
            return self.cfg.default_answer

        final_answer = self._select_answer(detailed_results)
        inference_time = time.time() - problem_start_time
        print(f'⏱️ Inference time: {inference_time:.2f}s\n')
        return final_answer

    def __del__(self):
        """Clean up resources on destruction."""
        if hasattr(self, 'server_process') and self.server_process:
            self.server_process.terminate()
            self.server_process.wait()

        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try:
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()
                except Exception:
                    pass
```

# Initialize Solver

```python
solver = AIMO3Solver(CFG)
```

# Submission

```python
def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    global correct_count, total_count, predictions
    
    question_id = id_.item(0)
    question_text = question.item(0)
    
    print("------")
    print(f"ID: {question_id}")
    print(f"Question: {question_text[:200]}...")
    
    gc.disable()
    
    final_answer = solver.solve_problem(
        question_text,
     )
    
    gc.enable()
    gc.collect()
    
    predictions[question_id] = final_answer

    # Check accuracy if ground truth available
    total_count += 1
    if question_id in ground_truth:
        gt = ground_truth[question_id]
        is_correct = (final_answer == gt)
        if is_correct:
            correct_count += 1
        status = "✅" if is_correct else "❌"
        print(f"Answer: {final_answer} | Ground Truth: {gt} | {status}")
        print(f"📊 Running Accuracy: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    else:
        print(f"Answer: {final_answer}")
    
    print("------\n")
    
    return pl.DataFrame({'id': question_id, 'answer': final_answer})
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