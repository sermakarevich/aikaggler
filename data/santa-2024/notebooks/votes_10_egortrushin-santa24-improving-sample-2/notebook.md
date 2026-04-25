# [Santa24] Improving Sample 2

- **Author:** Egor Trushin
- **Votes:** 240
- **Ref:** egortrushin/santa24-improving-sample-2
- **URL:** https://www.kaggle.com/code/egortrushin/santa24-improving-sample-2
- **Last run:** 2024-12-26 22:24:02.783000

---

# Improving Sample Number 2

- We aim to improve the publicly available score for text number 2 using Kaggle's computational resources.

- We will use the code and ideas from these three notebooks (with some modifications):  
  [[Eng/日本語]Tutorial 5 - Simulated Annealing](https://www.kaggle.com/code/utm529fg/eng-tutorial-5-simulated-annealing)  
  [[Eng/日本語]Tutorial 7 - Consideration of improvement](https://www.kaggle.com/code/utm529fg/eng-tutorial-7-consideration-of-improvement)  
  [Santa Claude 2: Electric Word-erloo](https://www.kaggle.com/code/richolson/santa-claude-2-electric-word-erloo)

- We will run optimization from scratch and stop when the score is lower than 299.

```python
import yaml
import datetime
import time
import gc
import os
import transformers
import torch
import numpy as np
import pandas as pd
import random
import math
from collections import Counter
from pprint import pprint
from typing import List, Union
```

```python
%%writefile config.yaml

subfile: "/kaggle/input/santa-2024/sample_submission.csv"
sample: 2
batch_size: 1024
model_path: '/kaggle/input/gemma-2/transformers/gemma-2-9b/2'

params:
    Tmax: 12.5
    Tmin: 4.
    nsteps: 100
    nsteps_per_T: 1000
    log_freq: 250
    random_state: 43
    cooling: 'exponential'
    k: 1.
```

```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    model_path: str = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2",
    load_in_8bit: bool = True,
    clear_mem: bool = False,
) -> float:
    """
    Calculates the mean perplexity of submitted text permutations compared to an original text.

    Parameters
    ----------
    solution : DataFrame
        DataFrame containing the original text in a column named 'text'.
        Includes a row ID column specified by `row_id_column_name`.

    submission : DataFrame
        DataFrame containing the permuted text in a column named 'text'.
        Must have the same row IDs as the solution.
        Includes a row ID column specified by `row_id_column_name`.

    row_id_column_name : str
        Name of the column containing row IDs.
        Ensures aligned comparison between solution and submission.

    model_path : str
        Path to the serialized LLM.

    clear_mem : bool
        Clear GPU memory after scoring by clearing the CUDA cache.
        Useful for testing.

    Returns
    -------
    float
        The mean perplexity score. Lower is better.

    Raises
    ------
    ParticipantVisibleError
        If the submission format is invalid or submitted strings are not valid permutations.

    Examples
    --------
    >>> import pandas as pd
    >>> model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
    >>> solution = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["this is a normal english sentence", "the quick brown fox jumps over the lazy dog"]
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["sentence english normal a is this", "lazy the over jumps fox brown quick the dog"]
    ... })
    >>> score(solution, submission, 'id', model_path=model_path, clear_mem=True) > 0
    True
    """
    # Check that each submitted string is a permutation of the solution string
    sol_counts = solution.loc[:, "text"].str.split().apply(Counter)
    sub_counts = submission.loc[:, "text"].str.split().apply(Counter)
    invalid_mask = sol_counts != sub_counts
    if invalid_mask.any():
        raise ParticipantVisibleError(
            "At least one submitted string is not a valid permutation of the solution string."
        )

    # Calculate perplexity for the submitted strings
    sub_strings = [" ".join(s.split()) for s in submission["text"].tolist()]  # Split and rejoin to normalize whitespace
    scorer = PerplexityCalculator(
        model_path=model_path,
        load_in_8bit=load_in_8bit,
    )  # Initialize the perplexity calculator with a pre-trained model
    perplexities = scorer.get_perplexity(sub_strings)  # Calculate perplexity for each submitted string

    if clear_mem:
        # Just move on if it fails. Not essential if we have the score.
        try:
            scorer.clear_gpu_memory()
        except:
            print("GPU memory clearing failed.")

    return float(np.mean(perplexities))


class PerplexityCalculator:
    """
    Calculates perplexity of text using a pre-trained language model.

    Adapted from https://github.com/asahi417/lmppl/blob/main/lmppl/ppl_recurrent_lm.py

    Parameters
    ----------
    model_path : str
        Path to the pre-trained language model

    load_in_8bit : bool, default=False
        Use 8-bit quantization for the model. Requires CUDA.

    device_map : str, default="auto"
        Device mapping for the model.
    """

    def __init__(
        self,
        model_path: str,
        load_in_8bit: bool = False,
        device_map: str = "auto",
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side="right")
        # Configure model loading based on quantization setting and device availability
        if load_in_8bit:
            if DEVICE.type != "cuda":
                raise ValueError("8-bit quantization requires CUDA device")

            # quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
            # quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)

            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",  # fp4 nf4
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
                device_map=device_map,
            )

        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.eval()
        # if not load_in_8bit:
        #    self.model.to(DEVICE)  # Explicitly move the model to the device

    def get_perplexity(self, input_texts: Union[str, List[str]], batch_size: 32) -> Union[float, List[float]]:
        """
        Calculates the perplexity of given texts.

        Parameters
        ----------
        input_texts : str or list of str
            A single string or a list of strings.

        batch_size : int, default=None
            Batch size for processing. Defaults to the number of input texts.

        verbose : bool, default=False
            Display progress bar.

        Returns
        -------
        float or list of float
            A single perplexity value if input is a single string,
            or a list of perplexity values if input is a list of strings.

        Examples
        --------
        >>> import pandas as pd
        >>> model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
        >>> scorer = PerplexityCalculator(model_path=model_path)

        >>> submission = pd.DataFrame({
        ...     'id': [0, 1, 2],
        ...     'text': ["this is a normal english sentence", "thsi is a slihgtly misspelled zr4g sentense", "the quick brown fox jumps over the lazy dog"]
        ... })
        >>> perplexities = scorer.get_perplexity(submission["text"].tolist())
        >>> perplexities[0] < perplexities[1]
        True
        >>> perplexities[2] < perplexities[0]
        True

        >>> perplexities = scorer.get_perplexity(["this is a sentence", "another sentence"])
        >>> all(p > 0 for p in perplexities)
        True

        >>> scorer.clear_gpu_memory()
        """
        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        loss_list = []

        batches = len(input_texts) // batch_size + (len(input_texts) % batch_size != 0)
        for j in range(batches):

            a = j * batch_size
            b = (j + 1) * batch_size
            input_batch = input_texts[a:b]

            with torch.no_grad():

                # Explicitly add sequence boundary tokens to the text
                text_with_special = [
                    f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}" for text in input_batch
                ]

                # Tokenize
                model_inputs = self.tokenizer(
                    text_with_special, return_tensors="pt", add_special_tokens=False, padding=True
                )

                if "token_type_ids" in model_inputs:
                    model_inputs.pop("token_type_ids")

                model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

                # Get model output
                output = self.model(**model_inputs, use_cache=False)
                logits = output["logits"]

                label = model_inputs["input_ids"]
                label[label == self.tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID

                # Shift logits and labels for calculating loss
                shift_logits = logits[..., :-1, :].contiguous()  # Drop last prediction
                shift_labels = label[..., 1:].contiguous()  # Drop first input

                # Calculate token-wise loss
                loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                loss = loss.view(len(logits), -1)
                valid_length = (shift_labels != PAD_TOKEN_LABEL_ID).sum(dim=-1)
                loss = torch.sum(loss, -1) / valid_length

                loss_list += loss.cpu().tolist()

        ppl = [math.exp(i) for i in loss_list]

        return ppl[0] if single_input else ppl

    def clear_gpu_memory(self) -> None:
        """Clears GPU memory by deleting references and emptying caches."""
        if not torch.cuda.is_available():
            return

        # Delete model and tokenizer if they exist
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache and reset memory stats
        with DEVICE:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
```

```python
def format_time(elapsed):
    """Take a time in seconds and return a string hh:mm:ss."""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
```

```python
class SimulatedAnnealing:
    def __init__(self, Tmax, Tmin, nsteps, nsteps_per_T, log_freq, random_state, cooling, k):
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.nsteps = nsteps
        self.nsteps_per_T = nsteps_per_T
        self.log_freq = log_freq
        self.cooling = cooling
        self.k = k
        random.seed(random_state)

    def _generate_neighbor(self, solution):
        r = random.choice(range(2))
        if r == 0:
            neighbor = solution.copy()
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            return neighbor
        elif r == 1:
            shift = solution.copy()
            extract, insert = random.sample(range(len(shift) - 1), 2)
            shift_words = shift[extract : extract + 1]
            shift = shift[:extract] + shift[extract + 1 :]
            shift = shift[:insert] + shift_words + shift[insert:]
            return shift

    def _acceptance_probability(self, current_energy, new_energy, temperature):
        """
        Calculate the probability of accepting a new solution.
        """
        if new_energy < current_energy:
            return 1.0
        return math.exp(self.k * (current_energy - new_energy) / temperature)

    def solve(self, text):

        t0 = time.time()  # Measure staring time

        current_solution = text.split()
        current_energy = scorer.get_perplexity(" ".join(current_solution), batch_size=config["batch_size"])

        best_solution = current_solution.copy()
        best_energy = current_energy

        temperature = self.Tmax
        Tfactor = -math.log(self.Tmax / self.Tmin)  # for exponentil cooling

        temperatures = [temperature]
        log_energies = [current_energy]

        for step in range(self.nsteps):

            accept = 0

            for step1 in range(self.nsteps_per_T):
                # generate neighbor
                new_solution = self._generate_neighbor(current_solution)
                new_energy = scorer.get_perplexity(" ".join(new_solution), batch_size=config["batch_size"])

                # calculation of acceptance probability
                acceptance = self._acceptance_probability(current_energy, new_energy, temperature)

                # update current solution
                if acceptance > random.random():
                    current_solution = new_solution
                    current_energy = new_energy
                    accept += 1

                # update best solution
                if new_energy < best_energy:
                    best_solution = new_solution.copy()
                    best_energy = new_energy
                    print(f"\nNew best score: {best_energy:8.3f}")
                    print("New text:", " ".join(best_solution), "\n", flush=True)

                # log
                log_energies.append(current_energy)
                temperatures.append(temperature)

                t1 = format_time(time.time() - t0)

                if step1 % self.log_freq == 0 or step1 == (self.nsteps_per_T - 1):
                    print(
                        f"T: {temperature:8.3f}  Step: {step1:6}  Acceptance Rate: {accept/(step1+1):7.4f}  Score: {current_energy:8.3f}  Best Score: {best_energy:8.3f}  Elapsed Time: {t1}",
                        flush=True,
                    )

            # lower the temperature
            if self.cooling == "linear":
                temperature -= (self.Tmax - self.Tmin) / self.nsteps
            elif self.cooling == "exponential":
                temperature = self.Tmax * math.exp(Tfactor * (step + 1) / self.nsteps)
            elif self.cooling == "logarithmic":
                temperature = self.Tmax / math.log10(step + 10)

            if best_energy < 299.:
                print("Stop! Target value is achieved.")
                break

        return " ".join(best_solution), best_energy, log_energies, temperatures
```

```python
with open("config.yaml", "r") as file_obj:
    config = yaml.safe_load(file_obj)
```

```python
scorer = PerplexityCalculator(config["model_path"])
optimizer = SimulatedAnnealing(**config["params"])
```

```python
sample_submission = pd.read_csv(config["subfile"])
text = sample_submission.loc[config["sample"], "text"]
text = " ".join(sorted(text.split()))
solution, score, log_scores, log_ts = optimizer.solve(text)
print("Final Score:", score)
print("Final Solution:", solution)
```

```python
pub_submission = pd.read_csv("/kaggle/input/regression-test/submission.csv")
pub_submission.at[config["sample"], "text"] = solution
pub_submission.to_csv("submission.csv", index=False)
```