# Solving ARC Prize 2025 with Qwen-3 Transformer

- **Author:** Jocelyn Dumlao
- **Votes:** 262
- **Ref:** jocelyndumlao/solving-arc-prize-2025-with-qwen-3-transformer
- **URL:** https://www.kaggle.com/code/jocelyndumlao/solving-arc-prize-2025-with-qwen-3-transformer
- **Last run:** 2025-05-23 09:37:08.223000

---

<div style="display: flex; justify-content: space-between; align-items: flex-start;">
    <div style="text-align: left;">
        <p style="color:#FFD700; font-size: 15px; font-weight: bold; margin-bottom: 1px; text-align: left;">Published on  May 1, 2025</p>
        <h4 style="color:#4B0082; font-weight: bold; text-align: left; margin-top: 6px;">Author: Jocelyn C. Dumlao</h4>
        <p style="font-size: 17px; line-height: 1.7; color: #333; text-align: center; margin-top: 20px;"></p>
        <a href="https://www.linkedin.com/in/jocelyn-dumlao-168921a8/" target="_blank" style="display: inline-block; background-color: #003f88; color: #fff; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px;">LinkedIn</a>
        <a href="https://github.com/jcdumlao14" target="_blank" style="display: inline-block; background-color: transparent; color: #059c99; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px; border: 2px solid #007bff;">GitHub</a>
        <a href="https://www.youtube.com/@CogniCraftedMinds" target="_blank" style="display: inline-block; background-color: #ff0054; color: #fff; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px;">YouTube</a>
        <a href="https://www.kaggle.com/jocelyndumlao" target="_blank" style="display: inline-block; background-color: #3a86ff; color: #fff; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px;">Kaggle</a>
    </div>
</div>

# <p style="padding:10px;background-color:#09abf2;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:12px;font-weight:200;border: 6px outset #f2102e;">Import Libraries</p>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
import warnings
import ast
from tqdm import tqdm

warnings.filterwarnings('ignore')
```

## Reference: [Qwen 3 qwen-lm/qwen-3](https://www.kaggle.com/models/qwen-lm/qwen-3/)

# <p style="padding:10px;background-color:#09abf2;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:12px;font-weight:200;border: 6px outset #f2102e;">Define paths to the datasets</p>

- The code first locates the JSON files containing the ARC (Abstraction and Reasoning Corpus) dataset.

```python
# Define paths
def get_path(name):
    return f'/kaggle/input/arc-prize-2025/{name}' if os.path.exists(f'/kaggle/input/arc-prize-2025/{name}') else name
```

# <p style="padding:10px;background-color:#09abf2;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:12px;font-weight:200;border: 6px outset #f2102e;">Load the Data</p>

- It then loads the contents of these JSON files into Python dictionaries. This includes training puzzles, evaluation puzzles, solutions, and a sample submission file.

```python
# Load data files
training_solutions = json.load(open(get_path('arc-agi_training_solutions.json')))
evaluation_solutions = json.load(open(get_path('arc-agi_evaluation_solutions.json')))
evaluation_challenges = json.load(open(get_path('arc-agi_evaluation_challenges.json')))
sample_submission = json.load(open(get_path('sample_submission.json')))
training_challenges = json.load(open(get_path('arc-agi_training_challenges.json')))
test_challenges = json.load(open(get_path('arc-agi_test_challenges.json')))
```

# <p style="padding:10px;background-color:#09abf2;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:12px;font-weight:200;border: 6px outset #f2102e;">Choose a Model</p>

- Download a pre-trained Qwen model from Kaggle Hub (a place to share models).

```python
# Load model
try:
    import kagglehub
    model_name = kagglehub.model_download("qwen-lm/qwen-3/transformers/0.6b")
except:
    model_name = "Qwen/Qwen-7B"  # fallback

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

```python
# Model loading and setup
try:
    import kagglehub
    model_name = kagglehub.model_download("qwen-lm/qwen-3/transformers/0.6b")
    print(f"Model downloaded from Kaggle Hub: {model_name}")
except ImportError:
    print("Kaggle Hub not available.  Make sure you have kaggle installed and are in a kaggle environment with internet access.")
    model_name = "Qwen/Qwen-7B"  # or any other appropriate Qwen model you have access to
except Exception as e:
     print(f"Error downloading from Kaggle Hub: {e}.  Using a local or alternative model.")
     model_name = "Qwen/Qwen-7B" # Replace this if needed to a suitable model
     print(f"Trying model {model_name}")
```

# <p style="padding:10px;background-color:#09abf2;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:12px;font-weight:200;border: 6px outset #f2102e;">Set Up the Hardware</p>

- It determines if a GPU is available (using CUDA) and sets the processing device to either the GPU or the CPU.

```python
# Check for CUDA availability and set device
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using device: {device}")
```

# <p style="padding:10px;background-color:#09abf2;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:12px;font-weight:200;border: 6px outset #f2102e;">Load the Model and Tokenizer</p>

- This is the core step. It loads the Qwen model and its corresponding tokenizer. The tokenizer converts text into numbers that the model can understand, and vice versa. The model is also moved to the chosen device (GPU or CPU).

```python
# Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
).eval()

if device == "cuda":
    model = model.to(device)
```

# <p style="padding:10px;background-color:#09abf2;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:12px;font-weight:200;border: 6px outset #f2102e;">Create a Chatbot Class</p>

- The chatbot will remember the conversation, and it can use that knowledge to make a better prediction.

```python
class QwenChatbot:
    def __init__(self, model, tokenizer, max_input_tokens=3000):
        self.tokenizer = tokenizer
        self.model = model
        self.history = []
        self.max_input_tokens = max_input_tokens

    def generate_response(self, user_input, max_tokens=64):
        self.history.append({"role": "user", "content": user_input})

        # Try generating with the full history, then truncate if necessary
        while True:
            messages = self.history + [{"role": "user", "content": user_input}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            if inputs.input_ids.shape[-1] <= self.max_input_tokens:
                break  # Good to go
            elif len(self.history) > 2:
                self.history = self.history[2:]  # Truncate oldest user-assistant pair
            else:
                print("⚠️ Unable to fit prompt within token limit even after truncation.")
                return "", ""

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # Qwen-specific token
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

        self.history.append({"role": "assistant", "content": response})
        return thinking_content, response


def is_valid_grid(grid):
    if not isinstance(grid, list) or len(grid) == 0:
        return False
    row_length = len(grid[0])
    for row in grid:
        if not isinstance(row, list) or len(row) != row_length:
            return False
        if not all(isinstance(cell, int) for cell in row):
            return False
    return True
```

# <p style="padding:10px;background-color:#09abf2;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:12px;font-weight:200;border: 6px outset #f2102e;">Demonstration and Task Integration</p>

- The script includes an if __name__ == "__main__": block, which is executed when the script is run directly (not imported as a module).
- First input (without /think or /no_think tags, thinking mode is enabled by default)
- Second input with /no_think
- Third input with /think
- The script loops through a few challenges from the training set. For each challenge, it constructs a prompt by showing the model some examples (input/output pairs) and asking it to predict the output for a new input.

Create a Chatbot object and interact with the model. 
- The model can respond in two ways: The first way is that it will tell the thought about the answer. The second way is the real answer that users want to know.

```python
# --- Demonstration and ARC Task Integration ---

# Initialize chatbot
#chatbot = QwenChatbot(model, tokenizer)

# Example Usage with the provided test cases.
#if __name__ == "__main__":

    # First input (without /think or /no_think tags, thinking mode is enabled by default)
#    user_input_1 = "How many r's in strawberries?"
#    print(f"User: {user_input_1}")
#    thinking_content, response_1 = chatbot.generate_response(user_input_1)
#    print(f"Thinking: {thinking_content}")
#    print(f"Bot: {response_1}")
#    print("----------------------")

#    # Second input with /no_think
#    user_input_2 = "Then, how many r's in blueberries? /no_think"
#    print(f"User: {user_input_2}")
#    thinking_content, response_2 = chatbot.generate_response(user_input_2, enable_thinking=False) # explicitly disable thinking for the demo
#    print(f"Thinking: {thinking_content}") # should be empty
#    print(f"Bot: {response_2}")
#    print("----------------------")

    # Third input with /think
#    user_input_3 = "Really? /think"
#    print(f"User: {user_input_3}")
#    thinking_content, response_3 = chatbot.generate_response(user_input_3, enable_thinking=True)
#    print(f"Thinking: {thinking_content}")
#    print(f"Bot: {response_3}")

#   print("----------------------")
```

```python
# --- Demonstration and ARC Task Integration ---

# Init chatbot
chatbot = QwenChatbot(model, tokenizer)

# Challenge solving function
def solve_arc_challenge(challenge, chatbot):
    predictions = []

    # Prebuild train prompt
    train_prompt = "Solve the following abstract reasoning challenge:\n\n"
    for i, example in enumerate(challenge['train']):
        train_prompt += f"Input {i + 1}:\n{example['input']}\nOutput {i + 1}:\n{example['output']}\n\n"

    for test_input in challenge['test']:
        prompt = train_prompt + f"Now, predict the output for the following test input:\n{test_input['input']}\nOutput:\n"

        try:
            _, arc_response = chatbot.generate_response(prompt, max_tokens=64)
            arc_response = arc_response.replace(" ", "").replace("\n", "")
            start, end = arc_response.find("[["), arc_response.rfind("]]")
            if start != -1 and end != -1:
                predicted_grid = ast.literal_eval(arc_response[start:end + 2])
                if not is_valid_grid(predicted_grid):
                    raise ValueError("Invalid grid structure")
            else:
                raise ValueError("No valid grid found in response")
        except Exception as e:
            print(f"⚠️ Failed to parse output: {e}")
            predicted_grid = [[0]]  # Safer fallback grid

        predictions.append(predicted_grid)
    return predictions
```

# <p style="padding:10px;background-color:#09abf2;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:12px;font-weight:200;border: 6px outset #f2102e;">Solve Challenges - ARC dataset</p>

```python
# --- Example of integrating with the ARC dataset ---
# Let's try to answer one of the training challenges.
# sample_challenge_id = list(training_challenges.keys())[0]  # Get the first challenge ID
#challenge = training_challenges[sample_challenge_id]

# Construct a prompt that describes the task and provides examples
#prompt = f"Solve the following abstract reasoning challenge.  Here's the challenge:\n\n"
#for i, task in enumerate(challenge['train']): # Use examples from the training set to build the prompt
#     prompt += f"Input {i+1}:\n"
#     prompt += str(task['input']) + "\n"
#     prompt += f"Output {i+1}:\n"
#     prompt += str(task['output']) + "\n\n"

# Add the test input.  Tell the model to predict this output
#prompt += "Now, predict the output for the following test input:\n"
#prompt += str(challenge['test'][0]['input']) + "\n"
#prompt += "Output:\n"  # Ask the model to generate the output

# Generate the response
#print("----------------------")
#print("Solving ARC Challenge:")
#print(f"Challenge ID: {sample_challenge_id}")
#thinking_content, arc_response = chatbot.generate_response(prompt)
#print(f"Thinking: {thinking_content}")
#print(f"Model's Prediction:\n{arc_response}")
#print("----------------------")
```

# <p style="padding:10px;background-color:#09abf2;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:12px;font-weight:200;border: 6px outset #f2102e;">Submission</p>

```python
# Evaluation loop
def create_submission(evaluation_challenges, chatbot):
    submission = {}
    for task_id, challenge in tqdm(evaluation_challenges.items(), desc="Solving ARC tasks"):
        predictions = solve_arc_challenge(challenge, chatbot)
        
        # Validate count matches number of test inputs
        assert len(predictions) == len(challenge['test']), f"Prediction count mismatch in task {task_id}"
        
        submission[task_id] = [{"attempt_1": p, "attempt_2": p} for p in predictions]
    return submission


# Main
if __name__ == "__main__":
    submission = create_submission(evaluation_challenges, chatbot)

    # Final submission validation
    for task_id, outputs in submission.items():
        assert isinstance(outputs, list), f"{task_id} outputs not a list"
        for entry in outputs:
            assert 'attempt_1' in entry and 'attempt_2' in entry, f"Missing attempts in {task_id}"
            assert is_valid_grid(entry['attempt_1']), f"Invalid attempt_1 grid in {task_id}"
            assert is_valid_grid(entry['attempt_2']), f"Invalid attempt_2 grid in {task_id}"

    with open('submission.json', 'w') as f:
        json.dump(submission, f)

    print("✅ Submission saved to `submission.json`.\n")

    # Show some results
    print("=== Sample submission preview ===")
    sample_tasks = list(submission.items())[:3]  # show first 3 tasks only
    for task_id, outputs in sample_tasks:
        print(f"\nTask ID: {task_id}")
        for i, output in enumerate(outputs):
            print(f"  Test input #{i+1}:")
            print(f"    Attempt 1 output: {output['attempt_1']}")
            print(f"    Attempt 2 output: {output['attempt_2']}")
```