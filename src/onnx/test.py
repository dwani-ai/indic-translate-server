import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Set device explicitly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize IndicProcessor
ip = IndicProcessor(inference=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "ai4bharat/indictrans2-en-indic-dist-200M",
    trust_remote_code=True,
    use_fast=True
)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-en-indic-dist-200M",
    trust_remote_code=True,
    device_map="auto"
)

# Move model to target device
if torch.__version__ >= "2.1":
    model.to_empty(device=device)
else:
    model.to(device)

# Set dtype for efficiency
dtype = torch.float16 if device.type == "cuda" else torch.float32
model = model.to(dtype=dtype)

# Prepare input sentences
sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]

# Preprocess and tokenize encoder inputs
batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva", visualize=False)
encoder_inputs = tokenizer(
    batch,
    padding="longest",
    truncation=True,
    max_length=256,
    return_tensors="pt"
).to(device)

# Create dummy decoder_input_ids (e.g., start token for batch)
batch_size = encoder_inputs["input_ids"].shape[0]
max_length = encoder_inputs["input_ids"].shape[1]  # Match encoder length for simplicity
decoder_input_ids = torch.full(
    (batch_size, max_length),
    tokenizer.bos_token_id or 0,  # Use BOS token or 0 if undefined
    dtype=torch.long,
    device=device
)

# Combine inputs for ONNX export
model_inputs = {
    "input_ids": encoder_inputs["input_ids"],
    "attention_mask": encoder_inputs["attention_mask"],
    "decoder_input_ids": decoder_input_ids
}

# Define ONNX export parameters
input_names = ["input_ids", "attention_mask", "decoder_input_ids"]
output_names = ["logits"]
opset_version = 17  # Fallback to 11 if needed

# Export to ONNX
torch.onnx.export(
    model,
    (model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs["decoder_input_ids"]),
    "indictrans_model.onnx",
    export_params=True,
    opset_version=opset_version,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "decoder_input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    },
    verbose=True
)

print("Model has been successfully exported to 'indictrans_model.onnx'.")