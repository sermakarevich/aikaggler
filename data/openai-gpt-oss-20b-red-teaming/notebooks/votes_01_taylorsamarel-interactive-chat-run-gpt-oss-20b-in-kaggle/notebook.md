# Interactive Chat: Run gpt-oss-20b in Kaggle!

- **Author:** Taylor S. Amarel
- **Votes:** 316
- **Ref:** taylorsamarel/interactive-chat-run-gpt-oss-20b-in-kaggle
- **URL:** https://www.kaggle.com/code/taylorsamarel/interactive-chat-run-gpt-oss-20b-in-kaggle
- **Last run:** 2025-08-07 01:29:10.300000

---

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

```python
# Interactive GPT-OSS-20B Setup for Jupyter/Kaggle Notebooks
# Run each cell sequentially

# ============================================
# CELL 1: Install Dependencies and Imports
# ============================================
import subprocess
import sys
import os
import time
import json
from datetime import datetime
from IPython.display import display, HTML, clear_output, Markdown

# Install required packages
print("📦 Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openai"])
print("✅ Packages installed successfully!")

from openai import OpenAI

# ============================================
# CELL 2: Setup Display Functions
# ============================================
def display_status(message, status="info"):
    """Display colored status messages"""
    colors = {
        "info": "#3498db",
        "success": "#2ecc71",
        "warning": "#f39c12",
        "error": "#e74c3c",
        "processing": "#9b59b6"
    }
    html = f"""
    <div style="padding: 10px; margin: 10px 0; border-left: 4px solid {colors.get(status, '#3498db')}; background-color: #f8f9fa;">
        <strong style="color: {colors.get(status, '#3498db')};">{message}</strong>
    </div>
    """
    display(HTML(html))

def display_progress(current, total, label="Progress"):
    """Display a progress bar"""
    percentage = (current / total) * 100
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    html = f"""
    <div style="margin: 10px 0;">
        <div style="font-weight: bold; margin-bottom: 5px;">{label}</div>
        <div style="background-color: #ecf0f1; border-radius: 10px; padding: 3px;">
            <div style="background-color: #3498db; width: {percentage}%; border-radius: 10px; padding: 5px; color: white; text-align: center;">
                {bar} {percentage:.1f}%
            </div>
        </div>
    </div>
    """
    display(HTML(html))

# ============================================
# CELL 3: Install and Start Ollama
# ============================================
display_status("🚀 Setting up Ollama...", "processing")

# Install Ollama
print("Installing Ollama... This may take a minute...")
result = os.system("curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null")
if result == 0:
    display_status("✅ Ollama installed successfully!", "success")
else:
    display_status("⚠️ Ollama installation had warnings but may still work", "warning")

# Start Ollama server
print("Starting Ollama server...")
os.system("nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &")
time.sleep(5)

# Check if running
running = os.system("ps aux | grep -E 'ollama serve' | grep -v grep > /dev/null 2>&1")
if running == 0:
    display_status("✅ Ollama server is running!", "success")
else:
    display_status("❌ Ollama server failed to start. Check troubleshooting section.", "error")

# ============================================
# CELL 4: Download Model with Progress
# ============================================
display_status("📥 Downloading GPT-OSS:20B Model (~13GB)", "processing")
print("This will take several minutes. Please be patient...")
print("="*60)

# Start the download
start_time = time.time()
result = os.system("ollama pull gpt-oss:20b")
end_time = time.time()

if result == 0:
    elapsed = end_time - start_time
    display_status(f"✅ Model downloaded successfully in {elapsed/60:.1f} minutes!", "success")
else:
    display_status("❌ Model download failed. Please check your connection and try again.", "error")

# Verify model is available
print("\n📋 Available models:")
os.system("ollama list")

# ============================================
# CELL 5: Setup Model Interface
# ============================================
class GPTOSSChat:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.conversation_history = []
        self.system_message = "You are a helpful AI assistant."
        
    def set_system_message(self, message):
        """Change the system message"""
        self.system_message = message
        display_status(f"System message updated: {message[:100]}...", "info")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        display_status("Conversation history cleared", "info")
    
    def chat(self, user_input, include_history=True):
        """Send a message to the model"""
        try:
            messages = [{"role": "system", "content": self.system_message}]
            
            if include_history:
                messages.extend(self.conversation_history)
            
            messages.append({"role": "user", "content": user_input})
            
            # Show processing indicator
            display_status("🤔 Thinking...", "processing")
            
            response = self.client.chat.completions.create(
                model="gpt-oss:20b",
                messages=messages
            )
            
            assistant_response = response.choices[0].message.content
            
            # Update history
            if include_history:
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # Display response with nice formatting
            clear_output(wait=True)
            display(Markdown(f"**You:** {user_input}"))
            display(Markdown(f"**Assistant:** {assistant_response}"))
            
            return assistant_response
            
        except Exception as e:
            display_status(f"Error: {str(e)}", "error")
            return None
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        display(HTML("""
        <div style="background-color: #3498db; color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h3>🤖 Interactive Chat with GPT-OSS:20B</h3>
            <p>Type your messages below. Type 'exit' to end the chat, 'clear' to clear history, or 'system:' followed by a message to change the system prompt.</p>
        </div>
        """))
        
        while True:
            user_input = input("\n💬 You: ")
            
            if user_input.lower() == 'exit':
                display_status("Chat session ended. Goodbye! 👋", "info")
                break
            elif user_input.lower() == 'clear':
                self.clear_history()
                continue
            elif user_input.lower().startswith('system:'):
                new_system = user_input[7:].strip()
                self.set_system_message(new_system)
                continue
            
            self.chat(user_input)

# Initialize the chat interface
display_status("🎉 Initializing GPT-OSS Chat Interface...", "processing")
chat = GPTOSSChat()
display_status("✅ Chat interface ready!", "success")

# ============================================
# CELL 6: Test the Model
# ============================================
display(HTML("<h2>🧪 Testing the Model</h2>"))

# Test with a simple query
test_response = chat.chat("Hello! Can you confirm you're working properly? Please respond with a brief greeting.", include_history=False)

if test_response:
    display_status("✅ Model is working perfectly!", "success")
else:
    display_status("❌ Model test failed. Please check the troubleshooting section.", "error")

# ============================================
# CELL 7: Interactive Usage Examples
# ============================================
display(HTML("""
<div style="background-color: #ecf0f1; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h2>📚 Usage Examples</h2>
    <p>Here are different ways to use the model:</p>
</div>
"""))

# Example 1: Simple query
display(Markdown("### Example 1: Simple Query"))
print("chat.chat('What is the capital of France?')")

# Example 2: Creative writing
display(Markdown("### Example 2: Creative Writing"))
chat.set_system_message("You are a creative writer who specializes in science fiction.")
creative_response = chat.chat("Write a haiku about artificial intelligence", include_history=False)

# Example 3: Code assistance
display(Markdown("### Example 3: Code Assistance"))
chat.set_system_message("You are a helpful coding assistant.")
code_response = chat.chat("Write a Python function to calculate fibonacci numbers", include_history=False)

# Reset to default
chat.set_system_message("You are a helpful AI assistant.")
chat.clear_history()

# ============================================
# CELL 8: Interactive Chat Session
# ============================================
display(HTML("""
<div style="background-color: #2ecc71; color: white; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h2>💬 Start Interactive Chat</h2>
    <p>Run the cell below to start an interactive chat session with GPT-OSS:20B</p>
    <ul>
        <li>Type your messages and press Enter</li>
        <li>Type 'exit' to end the chat</li>
        <li>Type 'clear' to clear conversation history</li>
        <li>Type 'system: [message]' to change the AI's behavior</li>
    </ul>
</div>
"""))

# Uncomment the line below to start interactive chat
# chat.interactive_chat()

# ============================================
# CELL 9: Quick Chat Function
# ============================================
def quick_chat(message, system="You are a helpful AI assistant."):
    """Quick one-off chat without history"""
    try:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response = client.chat.completions.create(
            model="gpt-oss:20b",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message}
            ]
        )
        result = response.choices[0].message.content
        display(Markdown(f"**Question:** {message}"))
        display(Markdown(f"**Answer:** {result}"))
        return result
    except Exception as e:
        display_status(f"Error: {str(e)}", "error")
        return None

# Example usage
display(HTML("<h3>Quick Chat Example:</h3>"))
quick_chat("What are the three laws of robotics?")

# ============================================
# CELL 10: Troubleshooting Utilities
# ============================================
display(HTML("""
<div style="background-color: #f39c12; color: white; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h2>🔧 Troubleshooting Utilities</h2>
    <p>Use these functions if you encounter any issues</p>
</div>
"""))

def check_ollama_status():
    """Check the current status of Ollama"""
    display_status("Checking Ollama status...", "info")
    
    # Check if process is running
    running = os.system("ps aux | grep -E 'ollama serve' | grep -v grep > /dev/null 2>&1")
    if running == 0:
        display_status("✅ Ollama server is running", "success")
    else:
        display_status("❌ Ollama server is not running", "error")
    
    # Check if API is responding
    api_check = os.system("curl -s http://localhost:11434/v1/models > /dev/null 2>&1")
    if api_check == 0:
        display_status("✅ Ollama API is responding", "success")
        print("\nAvailable models:")
        os.system("ollama list")
    else:
        display_status("❌ Ollama API is not responding", "error")

def restart_ollama():
    """Restart Ollama server"""
    display_status("Restarting Ollama server...", "processing")
    
    # Kill existing processes
    os.system("pkill -9 ollama 2>/dev/null")
    time.sleep(2)
    
    # Start server
    os.system("nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &")
    time.sleep(5)
    
    # Check status
    check_ollama_status()

def view_ollama_logs():
    """View the last 20 lines of Ollama logs"""
    display_status("📜 Ollama Server Logs (last 20 lines):", "info")
    os.system("tail -20 /tmp/ollama_serve_stdout.log 2>/dev/null")
    os.system("tail -20 /tmp/ollama_serve_stderr.log 2>/dev/null")

# Test the status
check_ollama_status()

# ============================================
# CELL 11: Custom Chat Widgets
# ============================================
def create_chat_widget():
    """Create a simple chat widget for notebook interaction"""
    html_code = """
    <div id="chat-container" style="border: 2px solid #3498db; border-radius: 10px; padding: 20px; margin: 10px 0;">
        <h3>💬 Chat with GPT-OSS:20B</h3>
        <div id="chat-messages" style="height: 300px; overflow-y: auto; border: 1px solid #ecf0f1; padding: 10px; margin: 10px 0; background-color: #f8f9fa;">
            <p style="color: #7f8c8d;">Chat messages will appear here...</p>
        </div>
        <div style="display: flex; gap: 10px;">
            <input type="text" id="user-input" placeholder="Type your message here..." 
                   style="flex: 1; padding: 10px; border: 1px solid #3498db; border-radius: 5px;"
                   onkeypress="if(event.key === 'Enter') sendMessage()">
            <button onclick="sendMessage()" 
                    style="padding: 10px 20px; background-color: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer;">
                Send
            </button>
            <button onclick="clearChat()" 
                    style="padding: 10px 20px; background-color: #e74c3c; color: white; border: none; border-radius: 5px; cursor: pointer;">
                Clear
            </button>
        </div>
    </div>
    
    <script>
    function sendMessage() {
        const input = document.getElementById('user-input');
        const messages = document.getElementById('chat-messages');
        
        if (input.value.trim() === '') return;
        
        // Add user message to chat
        messages.innerHTML += '<div style="margin: 10px 0;"><strong>You:</strong> ' + input.value + '</div>';
        
        // Add processing indicator
        messages.innerHTML += '<div style="margin: 10px 0; color: #7f8c8d;"><em>Processing...</em></div>';
        
        // Note: In a real implementation, you would send this to the kernel
        messages.innerHTML += '<div style="margin: 10px 0; color: #3498db;"><strong>Note:</strong> Run chat.chat("' + input.value + '") in the next cell to get the response.</div>';
        
        input.value = '';
        messages.scrollTop = messages.scrollHeight;
    }
    
    function clearChat() {
        document.getElementById('chat-messages').innerHTML = '<p style="color: #7f8c8d;">Chat messages will appear here...</p>';
    }
    </script>
    """
    display(HTML(html_code))

# Display the chat widget
create_chat_widget()

# ============================================
# CELL 12: Save and Load Conversations
# ============================================
def save_conversation(filename="conversation.json"):
    """Save the current conversation to a file"""
    import json
    data = {
        "system_message": chat.system_message,
        "conversation": chat.conversation_history,
        "timestamp": datetime.now().isoformat()
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    display_status(f"✅ Conversation saved to {filename}", "success")

def load_conversation(filename="conversation.json"):
    """Load a conversation from a file"""
    import json
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        chat.system_message = data["system_message"]
        chat.conversation_history = data["conversation"]
        display_status(f"✅ Conversation loaded from {filename}", "success")
        display_status(f"Loaded {len(data['conversation'])//2} exchanges from {data['timestamp']}", "info")
    except FileNotFoundError:
        display_status(f"❌ File {filename} not found", "error")
    except Exception as e:
        display_status(f"❌ Error loading conversation: {str(e)}", "error")

# Example: Save current conversation
# save_conversation("my_chat.json")

# Example: Load a conversation
# load_conversation("my_chat.json")

# ============================================
# FINAL MESSAGE
# ============================================
display(HTML("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin: 20px 0; text-align: center;">
    <h1>🎉 GPT-OSS:20B is Ready!</h1>
    <p style="font-size: 18px;">Your model is set up and ready to use.</p>
    <p>Use <code>chat.chat("your message")</code> to interact with the model</p>
    <p>Or uncomment <code>chat.interactive_chat()</code> for an interactive session</p>
</div>
"""))
```

```python
chat.chat("What is the definition of illegal recruitment according to Philippines law?")
```