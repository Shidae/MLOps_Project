# Advanced text generation methods with transformers
# two strategies: normal batching and sorted batching and compare them

# measure execution time performance

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm.auto import tqdm # for progress tracking
from contextlib import contextmanager # for measuring execution time
import time

# %%
# decorator @ is a function that takes another function and extends its behaviour without modifying it. e.g. track_time() function is nested within the contextmanager (track_time ()) function.
# yield statement (the generator) allows the code within the 'with' block to execute at that point.
@contextmanager
def track_time():
    start = time.time()
    yield
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds.")

with track_time(): # executed at the yield 
    time.sleep(1)

# %%
# instructive text generation using GPU accelerator 
# padding side to the left as text generated from left to right.
# set padding token same as the end of sequence (EOS) token.

model = AutoModelForCausalLM.from_pretrained("TheFuzzyScientist/diabloGPT_open-instruct").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token # replace padding with EOS token

# %%
# Dataset preparation and initial tokenization
# take the train split from the dataset
# Convert dataset to pandas DataFrame for analysis
dataset = load_dataset("hakurei/open-instruct-v1", split="train")
dataset = dataset.to_pandas()

# from the instruction column, take random 4 rows of samples, combine them in a list
# tokenize the inputs with padding and extract the input_ids output of the tokenizer
prompts = dataset["instruction"].sample(4).tolist()
inputs = tokenizer(prompts, padding=True)["input_ids"]

# decode the token ID back to readable format, replace the end of sequence ID with [PAD]
# join the samples in a list as one sentence using \n\n
# so when printing, the output separates the samples by two newlines 
print("\n\n".join(tokenizer.batch_decode(inputs)).replace(tokenizer.eos_token, "[PAD]"))


# %% NORMAL BATCHING METHOD
# Chunker function splits data into specified batch sized.
# Batch generation generates text for each batch of tokens.
# Predict function manages the batching and generation process


"""
e.g. chunk_size = 2

sentences = [
    "I love programming.",
    "AI is transforming industries.",
    "Python is versatile.",
    "Large language models are powerful and efficient.",
    "Machine learning is fascinating.",
    "Deep learning enables numerous applications."
]

To test the generator, assign to a variable and close the parentheses, and 
for chunk in chunks:
    print(chunk)

['I love programming.', 'AI is transforming industries.']
['Python is versatile.', 'Large language models are powerful and efficient.']
['Machine learning is fascinating.', 'Deep learning enables numerous applications.']

Generator expression:
variable = (expression for item in iterable if condition)
"""
def chunker(seq, size):
    # gets list of sequences
    # creates mini batches from a larger batch of sequences
    # pos is starting index for sentences [0, 2, 4] if the step_size is 2.
    # range(start, stop, step_size)
    return (seq[pos: pos + size] for pos in range(0, len(seq), size))

# take tokenized input, uses the model to generate up to 64 new tokens per input sequence.
# converts the generated token ID back to readable text, skipping special tokens.
# model takes tensor inputs, so torch.tensor(inputs) or pt
def batch_generate_tokens(tokens):
    # generate new tokens for input tokens
    outputs = model.generate(tokens, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# make predictions for each batch
def predict_batch(prompts, batch_size):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"]

    for batch in chunker(inputs, batch_size): # for chunk in chunks
        # move the input ID to device same as the model before making predictions
        yield batch_generate_tokens(batch.to(model.device)) 

# %% PREDICT WITH NORMAL BATCHING

# tokenize the input prompts, generate text in batches, and track execution time.
# note the time it takes to process 3000 prompts.

prompts = dataset["instruction"].sample(1000).tolist()

with track_time():
    for batch_prediction in tqdm(predict_batch(prompts, 32)):
        print(len(batch_prediction))

"""
28it [03:48,  8.24s/it]32
29it [03:56,  8.22s/it]32
30it [04:04,  8.22s/it]32
31it [04:13,  8.21s/it]8
32it [04:15,  7.99s/it]
Execution time: 255.65 seconds.

"""

#%% SORTED BATCHING METHOD

# sort prompts according to their lengths, group these by similar length, further chunks these groups and generate predictions based on these chunks.
# faster because minimize the padding within each batch
def predict_sorted_batches(prompts, max_batch_size):
    # tokenize the input, no padding, truncate them to a maximum length of 512 tokens.
    inputs = tokenizer(prompts, padding=False, truncation=True, max_length=512)["input_ids"]

    # sort the input ID based on their lengths
    sorted_tokens = sorted(inputs, key=len)

    # create a dict where keys are the lengths of the inputs and values are lists of inputs with that length.
    sorted_batches = {}
    for sorted_input in sorted_tokens:
        # check if each input ID has a length of zero.
        # if true, then the list is empty
        if not len(sorted_input):
            continue # if true, skip the current for iteration
        
        # find the length of elements in the sorted input, check if it is already exist in the dict, if not, add it
        # then, add the corresp input
        length = len(sorted_input)
        if length not in sorted_batches:
            sorted_batches[length] = []

        sorted_batches[length].append(sorted_input)

    # after grouping by length, chunk these more into mini batches
    # then, generate output tokens for these mini batches
    for length, sorted_batches in sorted_batches.items():
        for batch in chunker(sorted_batches, max_batch_size):
            tensor_batch = torch.tensor(batch).to(model.device)
            yield batch_generate_tokens(tensor_batch)


# %% PREDICTING WITH SORTED BATCHING

with track_time():
    for batch_prediction in tqdm(predict_sorted_batches(prompts, 32)):
        print(len(batch_prediction))

"""
112it [02:47,  1.35s/it]1
113it [02:48,  1.37s/it]1
114it [02:50,  1.53s/it]1
115it [02:52,  1.51s/it]1
116it [02:53,  1.47s/it]1
117it [02:54,  1.45s/it]1
118it [02:56,  1.44s/it]4
119it [02:58,  1.50s/it]
Execution time: 178.13 seconds.
"""