# %% 
from src.utils import track_time
from transformers import AutoTokenizer
from ctranslate2 import Generator # for model quantization

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = Generator("models/TinyLlama-1.1B-Chat-v1.0-ctrans")

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who is always helpful.",
    },
    {"role": "user", "content": "How can I get rid of a llama on my lawn?"},
]

# %%
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False)

input_tokens = [tokenizer.tokenize(prompt)] #* 256

with track_time(input_tokens):
    outputs = model.generate_batch(input_tokens)

results_ids = [res.sequences_ids[0] for res in outputs]
outputs = tokenizer.batch_decode(results_ids, skip_special_tokens=True)

print(outputs[100])