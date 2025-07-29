from transformers import AutoModelForCausalLM, AutoTokenizer
from bitsandbytes import quantize_model
import torch

model_name = "sarvamai/sarvam-translate"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Enable 4-bit quantization
    device_map="auto"   # Automatically map to available devices (e.g., CUDA)
)

# Translation task
tgt_lang = "Hindi"
input_txt = "Be the change you wish to see in the world."

# Chat-style message prompt
messages = [
    {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
    {"role": "user", "content": input_txt}
]

# Apply chat template to structure the conversation
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize and move input to model device
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate the output
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.01,
    num_return_sequences=1
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

print("Input:", input_txt)
print("Translation:", output_text)