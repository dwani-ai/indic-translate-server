import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor


#from transformers import init_empty_weights
#print("init_empty_weights is available")

# Load the model and tokenizer
ip = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
#model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-en-indic-dist-200M",
    trust_remote_code=True,
    low_cpu_mem_usage=True  # Alternative initialization for large models
)

# Prepare dummy input for tracing
sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]
batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva", visualize=False)
batch = tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt")

# Define input and output names for better readability
input_names = ["input_ids", "attention_mask"]
output_names = ["output"]

# Export the model to ONNX format
torch.onnx.export(
    model,
    (batch["input_ids"], batch["attention_mask"]),  # Model inputs
    "indictrans_model.onnx",  # Output file name
    opset_version=11,  # ONNX opset version (ensure compatibility)
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
    },
)

print("Model has been successfully exported to ONNX format.")
