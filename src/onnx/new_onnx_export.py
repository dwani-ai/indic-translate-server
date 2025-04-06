import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Set device explicitly (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize IndicProcessor
ip = IndicProcessor(inference=True)

# Load tokenizer and model with latest optimizations
tokenizer = AutoTokenizer.from_pretrained(
    "prajdabre/rotary-indictrans2-en-indic-dist-200M",
    trust_remote_code=True,
    use_fast=True  # Use fast tokenizer if supported
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "prajdabre/rotary-indictrans2-en-indic-dist-200M",
    trust_remote_code=True,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32  # FP16 on GPU for efficiency
).to(device)

# Optional: Compile model for performance (PyTorch 2.0+ feature)
if torch.__version__.startswith("2"):
    model = torch.compile(model, mode="reduce-overhead")  # Optimize for inference
    print("Model compiled with torch.compile")

# Prepare input sentences
sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]

# Preprocess with IndicProcessor and tokenize
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
output_names = ["logits"]  # More descriptive output name
opset_version = 17  # Use a higher opset for latest ONNX features (if supported)

# Export to ONNX with dynamic axes and verbose logging
torch.onnx.export(
    model,
    (batch["input_ids"], batch["attention_mask"]),
    "indictrans_model.onnx",
    export_params=True,
    opset_version=opset_version,
    do_constant_folding=True,  # Optimize by folding constants
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    },
    verbose=True  # Enable detailed export logs
)

print("Model has been successfully exported to 'indictrans_model.onnx'.")