# %% MODEL QUANTIZATION FOR EFFICIENT TEXT GENERATION

# using ctranslate2
# quantization reduces model size and speeds up inference, useful for deploying models in resource-constrained environments.

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm.auto import tqdm
from ctranslate2.converters import TransformersConverter # for model conversion and quantization
from ctranslate2 import Generator

from contextlib import contextmanager
import time
@contextmanager
def track_time():
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed time: {end - start} seconds.")

# %% MODEL AND TOKENIZED SETUP

model = AutoModelForCausalLM.from_pretrained("TheFuzzyScientist/diabloGPT_open-instruct").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# %% MODEL QUANTIZATION
# reduce model's size and improve inference speed.
# using TransformersConverter for ctranslate2 format conversion with float16 quantization
# Output: Quantized model ready for efficient text generation.

# Convert the model to CTranslate2 specified directory
model.save_pretrained("models/gpt-instruct")
tokenizer.save_pretrained("models/gpt-instruct")

# Initialize a converter to convert the saved model to CTranslate2 format, aim to optimize transformer model for faster inference
converter = TransformersConverter("models/gpt-instruct")

# Converts to CTranslate2 format and saves it folder
# model should be quantized to "float16" precision
out_path = converter.convert(output_dir="models/gpt-instruct-quant", quantization="float16")

# Initialize a generator object that will use converted model for generating text. The model is loaded from the directory.
# Generator should use cuda for inference if available.
generator = Generator("models/gpt-instruct-quant", device="cuda")

# %% DATASET PREPARATION

dataset = load_dataset("hakurei/open-instruct-v1", split="train")
dataset = dataset.to_pandas()

prompts = dataset["instruction"].sample(3000, random_state=42).tolist()

# %% NORMAL BATCHING METHOD

# using the non-quantized model, generate text in batches to establish baseline performance.
# chunker splits prompts into manageable batch sized.
# batch generation generates text for each batch.

def chunker(seq, size): # takes list of sequences and desired size of batch
    # pos iterable refers to each sentence in the list
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))

"""
Beam search is a search algorithm to generate sequences. It maintains multiple hypotheses (beams) at each decoding step and explores multiple path simultaneously. Helps find more optimal sequences compared to greedy search, which only keeps the best single hypothesis at each step.

num_beams=2 keeps track of the top 2 hypothesis at each step of each token generation.
Higher values leads to more quality outputs due to more explored sequences but increases the computational cost. 

At first step, model starts with the initial token and generate probability distribution for the next token. It selects the top 2 tokens based on their probabilities. Each of these tokens forms a separate beam. 

For each of the 2 beams from the previous step, the model generates the probability distribution for the next token.

It considers all possible continuations for both beams and selects the top 2 overall sequences out of all possible continuations, based on cumulative probabilities of the sequences.

After reaching the maximum length or EOS token, the model selects the highest probability sequence from the final set of beams.

If repetition_penalty > 1, repeated tokens will have their probabilities reduced, less likely to be selected again.

"""
def batch_generate_tokens(tokens):
    outputs = model.generate(tokens, max_length=256, pad_token_id=tokenizer.eos_token_id, num_beams=2, repetition_penalty=1.5)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def predict_batch(prompts, batch_size):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)["input_ids"]

    for batch in chunker(inputs, batch_size):
        # add input to model device before making prediction
        yield batch_generate_tokens(batch.to(model.device)) 

with track_time():
    for batch_prediction in tqdm(predict_batch(prompts, 32)):
        continue

# %% QUANTIZED MODEL BATCHING

# CTRANS tokenization adjust tokenization for cstranslate2 input.
# Batch generation uses the quantized model.

# CTranslate2 batching with quantized model
# CTranslate2 receives tokenized input instead of input_ids
def batch_generate_ctrans(prompts, batch_size):
    inputs = [tokenizer.tokenize(prompt, truncation=True, max_length=128) for prompt in prompts]

    results = generator.generate_batch(inputs, max_length=256, max_batch_size=batch_size, beam_size=2, repetition_penalty=1.5)

    result_ids = [res.sequences_ids[0] for res in results]
    return tokenizer.batch_decode(result_ids, skip_special_tokens=True)


# %% PREDICTING WITH QUANTIZED MODEL
# generate text with quantized model
# reduction in execution time vs unquantized model

del model
torch.cuda.empty_cache()
with track_time():
    batch_generate_ctrans(prompts, 32)