# same as first_deployment.py using the pipeline from HuggingFace but manually loading the model and the tokenizer, and decode the generated texts.

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import track_time

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who is always helpful.",
    },
    {"role": "user", "content": "How can I get rid of a llama on my lawn?"},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_special_token=False)

prompts = [prompt] * 100
input_ids = tokenizer(prompts, return_tensors="pt").to("cuda")["input_ids"]

with track_time(input_ids):
    outputs = model.generate(input_ids, max_length=256, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)


response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

"""
Took 12.80 seconds to process 10 inputs -> 0.78 inputs/s

Took 103.70 seconds to process 256 inputs -> 2.47 inputs/s

<|system|>
You are a friendly chatbot who is always helpful. 
<|user|>
How can I get rid of a llama on my lawn? 
<|assistant|>
There are several ways to get rid of a llama on your lawn:

1. Call a professional: If you have a llama that is causing damage to your lawn, you may want to consider calling a professional to remove it. They have the necessary equipment and expertise to safely remove the llama and prevent any further damage.

2. Use a lasso: A lasso is a long, thin rope that can be used to capture and remove a llama. You can purchase a lasso at a hardware store or online.

3. Use a trap: If you have a llama that is not causing damage to your lawn, you can use a trap to capture it. You can purchase a trap at a pet store or online.

4. Use a net: If you have a llama that is causing damage to your lawn, you can use a net to capture it. You can purchase a net at a hardware store or online.


"""