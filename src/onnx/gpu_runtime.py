import onnxruntime as ort
import numpy as np
import torch
from transformers import AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Set device explicitly to GPU (fail if not available)
device = torch.device("cuda")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU.")
print(f"Using device: {device}")

# Initialize IndicProcessor and tokenizer
ip = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained(
    "ai4bharat/indictrans2-en-indic-dist-200M",
    trust_remote_code=True,
    use_fast=True
)

# Load the ONNX model with CUDA provider
onnx_model_path = "indictrans_model.onnx"
session = ort.InferenceSession(
    onnx_model_path,
    providers=["CUDAExecutionProvider"]  # Force CUDA, no fallback to CPU
)
if "CUDAExecutionProvider" not in session.get_providers():
    raise RuntimeError("CUDAExecutionProvider not available. Ensure onnxruntime-gpu is installed and CUDA is configured.")
print(f"Loaded ONNX model with provider: {session.get_providers()[0]}")

# Prepare input sentences
sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]

# Preprocess and tokenize encoder inputs on GPU
batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva", visualize=False)
encoder_inputs = tokenizer(
    batch,
    padding="longest",
    truncation=True,
    max_length=256,
    return_tensors="pt"
).to(device)

# Create dummy decoder_input_ids for initial input (start token) on GPU
batch_size = encoder_inputs["input_ids"].shape[0]
decoder_input_ids = torch.full(
    (batch_size, 1),  # Start with single token (BOS)
    tokenizer.bos_token_id or 0,
    dtype=torch.long,
    device=device
)

# Define generation parameters
max_output_length = 256
eos_token_id = tokenizer.eos_token_id or 2

# Greedy decoding loop
generated_ids = decoder_input_ids  # Start with BOS token on GPU
for _ in range(max_output_length - 1):
    # Prepare ONNX inputs (convert to numpy for ORT, but keep computation on GPU until this point)
    onnx_inputs = {
        "input_ids": encoder_inputs["input_ids"].cpu().numpy(),
        "attention_mask": encoder_inputs["attention_mask"].cpu().numpy(),
        "decoder_input_ids": generated_ids.cpu().numpy()
    }

    # Run inference on GPU via ONNX Runtime
    outputs = session.run(None, onnx_inputs)
    logits = torch.from_numpy(outputs[0]).to(device)  # Back to GPU for processing

    # Get next token (greedy)
    next_token_logits = logits[:, -1, :]
    next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    # Append to generated sequence
    generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)

    # Check for EOS token
    if torch.any(next_token_ids == eos_token_id):
        break

# Move generated IDs to CPU for decoding
generated_ids = generated_ids.cpu().numpy()

# Post-process and decode outputs
generated_sequences = []
for i in range(batch_size):
    seq = generated_ids[i]
    eos_idx = np.where(seq == eos_token_id)[0]
    if len(eos_idx) > 0:
        seq = seq[:eos_idx[0]]
    decoded_text = tokenizer.decode(seq, skip_special_tokens=True)
    post_processed_text = ip.postprocess_batch([decoded_text], tgt_lang="hin_Deva")[0]
    generated_sequences.append(post_processed_text)

# Print results
print("\nInput Sentences and Translations:")
for input_sent, translation in zip(sentences, generated_sequences):
    print(f"English: {input_sent}")
    print(f"Hindi: {translation}\n")

print("Inference completed successfully on GPU.")