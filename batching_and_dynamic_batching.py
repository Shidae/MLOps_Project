# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch #tensor operations
from tqdm.auto import tqdm

"""
PAD token pads sequences to so that they all have the same length in a batch for efficient batch processing.
The model ignores PAD tokens during training using attention mask.

EOS token indicates the end of a meaningful sequence.

When defining pad token = eos token, special handling is required to distinguish between actual end of sequence and padding.


"""

# %%
# import model and tokenizer, set pad token as eos token
model = AutoModelForCausalLM.from_pretrained("TheFuzzyScientist/diabloGPT_open-instruct").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# prepare dataset
dataset = load_dataset("hakurei/open-instruct-v1", split="train")
dataset = dataset.to_pandas()

"""
generate_text(): Single prompt text generation
batch_generate_texts(): Batch prompt text generation
"""

# %% SINGLE TEXT GENERATION
# uses encode(), does not require padding, use decode()
# tokenized input returns only input_ids
def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_length=100) # max sequence length, not max new token!

    # outputs is a tensor with shape (batch_size, sequence_length) even when using single prompt.
    # So access the first sequence in a single batch -> outputs[0] 
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # find the index of the first ".", +1 to include the "." in the final generated output. To ensure continuity given max sequence length.
    return generated[: generated.find(".") + 1]

# %%

text = "Which makeup products are the best right now."
generate_text(text)
# %% BATCH TEXT GENERATION

# uses batch_decode, due to padding sequences to the same length in a batch, handling PAD to EOS is required.
# tokenized inputs returns multiple outputs such as input_ids, attention_mask, etc.
# outputs are processed in batches at once.
def batch_generated_texts(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)["input_ids"]
    outputs = model.generate(inputs, max_length=64, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return generated

# %%
# or use .sample(200) to select random 200 rows of samples
batch_generated_texts(dataset["instruction"][:1].to_list()) # select the first row
# batch_generated_texts(dataset["instruction"][:20].to_list())
# batch_generated_texts(dataset["instruction"][:100].to_list())
# batch_generated_texts(dataset["instruction"][:200].to_list())
# batch_generated_texts(dataset["instruction"][:1].to_list())

# %% DYNAMIC BATCHING FOR EFFICIENCY
# for hardware optimization
# use different batching methods to improve performance

# take tokenized inputs
def batch_generate_tokens(tokens):
    # torch.stack stacks individual token tensors when they are passed individually instead of multiple samples in a list.
    outputs = model.generate(torch.stack(tokens), max_length=64, pad_token_id = tokenizer.eos_token_id)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def dynamic_batching(prompts, max_tokens, is_pretokenized=False):
    if not is_pretokenized:
        tokenized_texts = tokenizer(prompts, return_tensor="pt", padding=True).to(model.device)["input_ids"]
    else:
        tokenized_texts = prompts

    current_batch = [] # hold batch of tokenized inputs
    current_batch_size = 0 # counter

    # First tokenized text is e.g. "I love programming" -> [101, 231, 123, 431]
    # If current_batch_size + len(tokenized_text) = 0 + 4 = 4 < max_tokens(3200) AND current batch is not empty, append tokenized_text to the current_batch
    # Second tokenized text is e.g. "Python is a versatile language." -> [231, 432, 354, 133] < 3200 tokens, add to the same batch
    for tokenized_text in tokenized_texts:
        if current_batch_size + len(tokenized_text) > max_tokens and current_batch:
            # generate output ONLY AFTER the batch is FULL
            yield batch_generate_tokens(current_batch) 

            # create new batch when > max_token and reset the counter
            current_batch, current_batch_size = [], 0

        # 
        current_batch.append(tokenized_text)
        current_batch_size += len(tokenized_text)

    # Generate output for the remaining batch (which is not full 3200) which is added after output is generated on the previous batch
    if current_batch:
        yield batch_generate_tokens(current_batch)
        # pass # optional

instruction = dataset["instruction"].sample(40).to_list() * 1000 # multiply the sample into 1000 -> 40,000 samples in a list
generator = dynamic_batching(instruction, max_tokens=3200)
# %% Apply dynamic batching on a large dataset and measuring performance

from contextlib import contextmanager
import time

@contextmanager
def track_time():
    start = time.time()
    yield
    end = time.time()

    return(f"Execution time: {end} - {start} seconds.")

with track_time:
    for batch_predictions in tqdm(generator):
        continue


# %% SORT BATCHES

def sort_batches(prompts, max_tokens):
    tokenized_texts = tokenizer(prompts, padding=False)["input_ids"]
    # a large list sort by length
    sorted_tokens = sorted(tokenized_texts, key=len) 

    # list of a grouped list by length
    sorted_batches = {}
    for sorted_token in sorted_tokens:
        length = len(sorted_token)
        if length not in sorted_batches:
            sorted_batches[length] = []

        sorted_batches[length].append(sorted_token)
    
    # convert each tokenized text to a tensor and stack them
    # torch.tensor([123, 234, 352, 532])
    # torch.tensor([423, 231, 235, 124])
    for length, sorted_batch in sorted_batches.items():
        tensor_batch = torch.stack([torch.tensor(sorted_token) for sorted_token in sorted_batch]).to(model.device)

        # for each sorted batch, prediction is generated using dynamic_batching()
        for batch_prediction in dynamic_batching(tensor_batch, max_tokens=max_tokens, is_pretokenized=True):
            yield batch_prediction

generator = sort_batches(dataset["instruction"][:40].to_list() * 1000, 3200)

with track_time():
    for batch_predictions in tqdm(generator):
        print(len(batch_predictions))
