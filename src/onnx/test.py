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

# Load model with meta device first, then move to target device
model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-en-indic-dist-200M",
    trust_remote_code=True,
    device_map="auto"  # Let transformers handle initial device placement
)

# Move model to target device with proper weight allocation
if torch.__version__ >= "2.1":
    model.to_empty(device=device)  # Allocate weights on target device
else:
    model.to(device)  # Fallback for older PyTorch versions

# Set dtype for efficiency
dtype = torch.float16 if device.type == "cuda" else torch.float32
model = model.to(dtype=dtype)

# Optional: Compile model (PyTorch 2.0+)
if torch.__version__.startswith("2"):
    model = torch.compile(model, mode="reduce-overhead")
    print("Model compiled with torch.compile")

# Prepare input sentences
sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]

# Preprocess and tokenize
batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva", visualize=False)
batch = tokenizer(
    batch,
    padding="longest",
    truncation=True,
    max_length=256,
    return_tensors="pt"
).to(device)

# Define ONNX export parameters
input_names = ["input_ids", "attention_mask"]
output_names = ["logits"]
opset_version = 17  # Adjust to 11 if 17 causes issues

# Export to ONNX
torch.onnx.export(
    model,
    (batch["input_ids"], batch["attention_mask"]),
    "indictrans_model.onnx",
    export_params=True,
    opset_version=opset_version,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    },
    verbose=True
)

print("Model has been successfully exported to 'indictrans_model.onnx'.")