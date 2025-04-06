import onnxruntime as ort
import numpy as np
import torch
from transformers import AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Set device for input preparation (CPU for ONNX Runtime, as it handles device internally)
device = torch.device("cpu")  # ONNX Runtime will use GPU if available and configured
print(f"Preparing inputs on: {device}")

# Initialize IndicProcessor and tokenizer
ip = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained(
    "ai4bharat/indictrans2-en-indic-dist-200M",
    trust_remote_code=True,
    use_fast=True
)

# Load the ONNX model
onnx_model_path = "indictrans_model.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(f"Loaded ONNX model with providers: {session.get_providers()}")

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

# Create dummy decoder_input_ids for initial input (start token)
batch_size = encoder_inputs["input_ids"].shape[0]
max_length = encoder_inputs["input_ids"].shape[1]
decoder_input_ids = torch.full(
    (batch_size, 1),  # Start with single token (BOS) for generation
    tokenizer.bos_token_id or 0,
    dtype=torch.long,
    device=device
)

# Prepare ONNX inputs (convert to numpy)
onnx_inputs = {
    "input_ids": encoder_inputs["input_ids"].numpy(),
    "attention_mask": encoder_inputs["attention_mask"].numpy(),
    "decoder_input_ids": decoder_input_ids.numpy()
}

# Define generation parameters
max_output_length = 256
eos_token_id = tokenizer.eos_token_id or 2  # Default to 2 if undefined

# Greedy decoding loop
generated_ids = decoder_input_ids.numpy()  # Start with BOS token
for _ in range(max_output_length - 1):  # -1 to account for initial BOS
    # Run inference
    outputs = session.run(None, onnx_inputs)
    logits = outputs[0]  # Shape: (batch_size, sequence_length, vocab_size)

    # Get next token (greedy: argmax over last token's logits)
    next_token_logits = logits[:, -1, :]  # Last position
    next_token_ids = np.argmax(next_token_logits, axis=-1).reshape(batch_size, 1)

    # Append to generated sequence
    generated_ids = np.concatenate([generated_ids, next_token_ids], axis=1)

    # Update decoder_input_ids for next iteration
    onnx_inputs["decoder_input_ids"] = generated_ids

    # Check for EOS token
    if np.any(next_token_ids == eos_token_id):
        break

# Post-process and decode outputs
generated_sequences = []
for i in range(batch_size):
    # Extract sequence up to EOS or full length
    seq = generated_ids[i]
    eos_idx = np.where(seq == eos_token_id)[0]
    if len(eos_idx) > 0:
        seq = seq[:eos_idx[0]]
    # Decode to text
    decoded_text = tokenizer.decode(seq, skip_special_tokens=True)
    # Post-process with IndicProcessor
    post_processed_text = ip.postprocess_batch([decoded_text], tgt_lang="hin_Deva")[0]
    generated_sequences.append(post_processed_text)

# Print results
print("\nInput Sentences and Translations:")
for input_sent, translation in zip(sentences, generated_sequences):
    print(f"English: {input_sent}")
    print(f"Hindi: {translation}\n")

print("Inference completed successfully.")