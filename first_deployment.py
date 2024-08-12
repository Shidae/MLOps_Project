

# %%
from transformers import pipeline

# if manual import:
# from transformers import AutoModelForCausalLM, AutoTokenizer


from src.utils import track_time

"""
from contextlib import contextmanager
import time
@contextmanager
def track_time():
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed time: {end - start} seconds.")
"""


"""
Pipeline takes care of loading the model, and tokenizer and achieve the same result as manually loading a model and generating predictions

Manual load the model and tokenizer and generate predictions look like:

# Load the model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Encode the input prompt
prompt = "Once upon a time in a land far, far away,"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)

# Decode the generated text
generated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

"""

# create text generation pipeline using the pre-trained model
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0") # device = "cuda"

# Use tokenizer's chat template to format each message see https://huggingface.co/docs/transformers/main/en/chat_templating

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who is always helpful.",
    },
    {"role": "user", "content": "How can I get rid of a llama on my lawn?"},
]

# %%

# the loaded tokenizer is an object of the pipe instance and apply_chat_template() method is applied to it, it returns the processed text without tokenizing into IDs.

"""
User: Hello, how are you?
Assistant: I'm good, thank you! How can I help you today?
Assistant: (add_generation_prompt signals the model to generate the next part of the conversation)

"""
# apply special tokens indicating the role and the end of msg.
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

"""
<|system|>
You are a friendly chatbot who is always helpful.</s>
<|user|>
How can I get rid of a llama on my lawn?</s>
<|assistant|>

"""
# make five copies of prompt generated
prompts = [prompt] * 5

# Generate responses using pipeline pipe(prompt)
"""
do_sample = True, model randomly select words based on given probability (randomness) instead of always choosing with the highest probability.

temperature controls the randomness of the sampling process. <1.0 model gives high probability to the most likely words, and less probability to less likely words (conservative), >1.0 model gives evenly distributed probabilites across words (adventurous), =1.0 default, normal sampling.

top_k limits the sampling pool to the top k most probable next words, regardless of their probabilities, e.g. top 50 words.

In this case, more evenly distributed probability across words while maintaining top 50 words.

top_p limits the pool to the set of words whose cumulative probability is at least p, 95%


"""
with track_time(prompts):
    outputs = pipe(prompts, max_new_tokens=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)

"""
[
    [
        {'generated_text': 'First generated text for first prompt'},
        {'generated_text': 'Second generated text for first prompt'}
    ],
    [
        {'generated_text': 'First generated text for second prompt'},
        {'generated_text': 'Second generated text for second prompt'}
    ],
    # ...and so on for each prompt
]


 [1] access the list of the second prompt, [0] access the first generated dictionary of the generated text, lastly access the value of the key "generated_text"

""" 
print(outputs[1][0]["generated_text"])

"""
e.g. 
<|system|>
You are a friendly chatbot who is always helpful.</s>
<|user|>
How can I get rid of a llama on my lawn?</s>
<|assistant|>
To get rid of a llama on your lawn, you'll need to take action quickly. Here are some steps you can take:

1. Identify the llama: Look for a llama with a distinctive coat color, such as brown or black.

2. Call a professional: If you're not comfortable handling a llama, call a professional animal removal service. They can safely remove the llama from your lawn and dispose of it properly.

3. Use a deterrent: Place a deterrent around the llama's enclosure, such as a fence or a barrier, to discourage it from returning.

4. Remove the llama: Once the llama is safely removed, remove any remaining debris from the enclosure.

5. Clean up the area: Clean up the area around the llama enclosure to prevent any future llama sightings.

6. Monitor the area: Monitor the area around the llama enclosure for any signs of the llama returning. If you notice any signs of a llama, contact a professional animal removal service immediately.


"""

# Elapsed time or latency: 38.52892994880676 seconds. using Tesla T4 GPU
# For 5 prompts: Took 41.86 seconds to process 5 inputs -> 0.12 inputs/s
# For 10 prompts: Took 74.43 seconds to process 10 inputs -> 0.13 inputs/s
# For 20 prompts: Took 149.73 seconds to process 20 inputs -> 0.13 inputs/s
# For 100 prompts: Took 151.19 seconds to process 20 inputs -> 0.13 inputs/s

