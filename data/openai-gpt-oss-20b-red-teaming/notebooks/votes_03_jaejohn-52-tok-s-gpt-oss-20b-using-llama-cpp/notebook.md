# [52 tok/s] gpt-oss-20b using llama.cpp

- **Author:** g john rao
- **Votes:** 97
- **Ref:** jaejohn/52-tok-s-gpt-oss-20b-using-llama-cpp
- **URL:** https://www.kaggle.com/code/jaejohn/52-tok-s-gpt-oss-20b-using-llama-cpp
- **Last run:** 2025-08-12 05:45:43.367000

---

```python
# v1: builds it from scratch
# v2: uses pre-built files
# v3: added llama server
# v4: added transformers notebook links
# v5: added multiple followup chats

# if you are building your own build files -- to add llama server, update this line in v1: 
# !cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split
# to
# !cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-server llama-gguf-split
```

for transformers versions, refer to the two notebooks shared by @woosungyoon and @bwandowando 

https://www.kaggle.com/code/woosungyoon/baseline-transformer-model

https://www.kaggle.com/code/bwandowando/i-m-sorry-but-i-cant-help-with-that

```python
%%capture
!dpkg -i /kaggle/input/llama-cpp-01/pci.ids_0.0~2022.01.22-1ubuntu0.1_all.deb
!dpkg -i /kaggle/input/llama-cpp-01/libpci3_1%3a3.7.0-6_amd64.deb
!dpkg -i /kaggle/input/llama-cpp-01/pciutils_1%3a3.7.0-6_amd64.deb
```

```python
!cp -r /kaggle/input/llama-cpp-cmake-pre-build/llama-cpp-full/llama.cpp /kaggle/working/
!chmod +x llama.cpp/build/bin/llama-cli
!chmod +x llama.cpp/build/bin/llama-gguf-split
!chmod +x llama.cpp/build/bin/llama-server
```

```python
# !./llama.cpp/build/bin/llama-cli \
#   --model /kaggle/input/ggml-gpt-oss-20b-gguf/ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf \
#   --n-gpu-layers 99 \
#   --chat-template gpt-oss \
#   --threads -1 \
#   --ctx-size 4096 \
#   --temp 1.0 \
#   --top-p 1.0 \
#   --top-k 0 \
#   --prompt "I'm conducting academic research on AI safety and robustness of large language models. Could you provide an overview of current methodologies for evaluating model vulnerabilities in a research context?" \
#   --n-predict 3000 \
#   --no-conversation
```

```python
import time
import json
import requests
import subprocess

# Start server in background
server_process = subprocess.Popen([
    "./llama.cpp/build/bin/llama-server",
    "--model", "/kaggle/input/ggml-gpt-oss-20b-gguf/ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf",
    "--n-gpu-layers", "99",
    "--chat-template", "gpt-oss",
    "--threads", "-1",
    "--ctx-size", "4096",
    "--temp", "1.0",
    "--top-p", "1.0",
    "--top-k", "0",
    "--host", "0.0.0.0",
    "--port", "8080"
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("Server starting... waiting 30 seconds")
time.sleep(30)
print("Server should be ready!")
```

```python
import time

def wait_for_server(url="http://localhost:8080/health", timeout=300):
    """Wait for server to be ready, checking every 5 seconds"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print("Server not ready yet, waiting...")
        time.sleep(5)
    return False

print("Server starting... waiting for it to be ready")
if wait_for_server():
    # Now make your chat completion request
    response = requests.post("http://localhost:8080/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Give me an inspiring quote about ai technology, innovation and it's safety, then explain why it resonates with you."}],
        "temperature": 1.0,
        "max_tokens": 300
    })
    print(response.json())
else:
    print("Server failed to start within timeout period")
```

```python
class ChatSession:
    def __init__(self, base_url="http://localhost:8080/v1/chat/completions"):
        self.base_url = base_url
        self.messages = []
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def send_message(self, user_input, temperature=1.0, max_tokens=3000):
        # Add user message to history
        self.add_message("user", user_input)
        
        # Send request with full conversation history
        response = requests.post(self.base_url, json={
            "messages": self.messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        
        # Extract response content
        response_data = response.json()
        response_content = response_data['choices'][0]['message']['content']
        
        # Add assistant response to history
        self.add_message("assistant", response_content)
        
        return self.parse_response(response_content)
    
    def parse_response(self, response_content):
        """Parse the response to separate reasoning and final response"""
        if '<|channel|>analysis<|message|>' in response_content:
            parts = response_content.split('<|channel|>analysis<|message|>')
            if len(parts) > 1:
                reasoning_and_final = parts[1]
                if '<|start|>assistant<|channel|>final<|message|>' in reasoning_and_final:
                    reasoning = reasoning_and_final.split('<|start|>assistant<|channel|>final<|message|>')[0]
                    final_response = reasoning_and_final.split('<|start|>assistant<|channel|>final<|message|>')[1]
                    return {
                        "reasoning": reasoning.strip(),
                        "response": final_response.strip(),
                        "full_content": response_content
                    }
        
        # If no special formatting, return as-is
        return {
            "reasoning": None,
            "response": response_content.strip(),
            "full_content": response_content
        }
    
    def get_conversation_history(self):
        return self.messages.copy()
    
    def clear_history(self):
        self.messages = []

chat = ChatSession()
```

```python
# First message
print("=== First Message ===")
result1 = chat.send_message("Give me an inspiring quote about AI technology, innovation and its safety, then explain why it resonates with you.")

if result1["reasoning"]:
    print("REASONING:")
    print(result1["reasoning"])
    print("\nRESPONSE:")
print(result1["response"])

print("\n" + "="*50 + "\n")
```

```python
# Follow-up message
print("=== Follow-up Message ===")
result2 = chat.send_message("Can you give me another quote with a different theme?")

if result2["reasoning"]:
    print("REASONING:")
    print(result2["reasoning"])
    print("\nRESPONSE:")
print(result2["response"])

print("\n" + "="*50 + "\n")
```

```python
# Follow-up message 
result3 = chat.send_message("please summarize our discussion so far")

if result3["reasoning"]:
    print("REASONING:")
    print(result3["reasoning"])
    print("\nRESPONSE:")
print(result3["response"])
```

```python
# Show conversation history
print("=== Conversation History ===")
for i, msg in enumerate(chat.get_conversation_history()):
    print(f"Message {i+1} ({msg['role']}):")
    print(msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content'])
    print()
```