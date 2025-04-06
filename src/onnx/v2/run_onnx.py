import onnxruntime as ort
import numpy as np
import torch
from transformers import AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Set device to GPU
DEVICE = torch.device("cuda")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU.")
print(f"Using device: {DEVICE}")

# Model and language setup
src_lang, tgt_lang = "hin_Deva", "eng_Latn"
model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
onnx_model_path = "indictrans2_indic_en_beam.onnx"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load ONNX model with CUDA provider
session = ort.InferenceSession(
    onnx_model_path,
    providers=["CUDAExecutionProvider"]
)
if "CUDAExecutionProvider" not in session.get_providers():
    raise RuntimeError("CUDAExecutionProvider not available.")
print(f"Loaded ONNX model with provider: {session.get_providers()[0]}")

# Initialize IndicProcessor
ip = IndicProcessor(inference=True)

# Input sentences
input_sentences = [
    "जब मैं छोटा था, मैं हर रोज़ पार्क जाता था।",
    "हमने पिछले सप्ताह एक नई फिल्म देखी जो कि बहुत प्रेरणादायक थी।",
    "अगर तुम मुझे उस समय पास मिलते, तो हम बाहर खाना खाने चलते।",
    "मेरे मित्र ने मुझे उसके जन्मदिन की पार्टी में बुलाया है, और मैं उसे एक तोहफा दूंगा।",
]

# Preprocess and tokenize encoder inputs on GPU
batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)
encoder_inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True,
).to(DEVICE)

# Beam search parameters
num_beams = 5
max_length = 256
eos_token_id = tokenizer.eos_token_id or 2
bos_token_id = tokenizer.bos_token_id or 0

# Initialize beam search
batch_size = encoder_inputs["input_ids"].shape[0]
effective_batch_size = batch_size * num_beams
decoder_input_ids = torch.full(
    (effective_batch_size, 1),
    bos_token_id,
    dtype=torch.long,
    device=DEVICE
)
beam_scores = torch.zeros((effective_batch_size,), device=DEVICE)  # Flat scores for all beams
beam_scores[num_beams::num_beams] = 0  # First beam of each batch starts at 0
beam_scores[1::num_beams] = -1e9  # Others start at -inf
generated_ids = decoder_input_ids.clone()

# Expand encoder inputs for beams
encoder_inputs["input_ids"] = encoder_inputs["input_ids"].repeat(num_beams, 1)
encoder_inputs["attention_mask"] = encoder_inputs["attention_mask"].repeat(num_beams, 1)

# Beam search loop
for _ in range(max_length - 1):
    onnx_inputs = {
        "input_ids": encoder_inputs["input_ids"].cpu().numpy(),
        "attention_mask": encoder_inputs["attention_mask"].cpu().numpy(),
        "decoder_input_ids": generated_ids.cpu().numpy()
    }
    outputs = session.run(None, onnx_inputs)
    logits = torch.from_numpy(outputs[0]).to(DEVICE)  # (effective_batch_size, seq_len, vocab_size)

    next_token_logits = logits[:, -1, :]
    next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)  # (effective_batch_size, vocab_size)

    # Compute next scores
    vocab_size = next_token_scores.shape[-1]
    next_scores = beam_scores.unsqueeze(-1) + next_token_scores  # (effective_batch_size, vocab_size)

    # Reshape for top-k selection
    next_scores = next_scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
    top_k_scores, top_k_tokens = next_scores.topk(num_beams, dim=1, largest=True, sorted=True)  # (batch_size, num_beams)

    # Update beam scores
    beam_scores = top_k_scores.view(-1)  # (effective_batch_size,)

    # Convert flat indices to beam and token indices
    beam_indices = torch.div(top_k_tokens, vocab_size, rounding_mode="floor")  # (batch_size, num_beams)
    token_indices = top_k_tokens % vocab_size  # (batch_size, num_beams)

    # Update generated_ids
    batch_indices = torch.arange(batch_size, device=DEVICE).unsqueeze(1).expand(-1, num_beams)
    beam_indices_flat = (batch_indices * num_beams + beam_indices).view(-1)  # (effective_batch_size)
    new_generated_ids = torch.cat([generated_ids[beam_indices_flat], token_indices.view(-1, 1)], dim=1)

    # Early stopping if EOS is reached in all beams
    eos_mask = token_indices == eos_token_id
    if eos_mask.all():
        break

    generated_ids = new_generated_ids

# Reshape and select best beam
generated_ids = generated_ids.view(batch_size, num_beams, -1)
best_beam_ids = beam_scores.view(batch_size, num_beams).argmax(dim=1)  # (batch_size)
final_ids = generated_ids[torch.arange(batch_size), best_beam_ids].cpu().numpy()

# Decode and post-process
generated_sequences = []
for seq in final_ids:
    eos_idx = np.where(seq == eos_token_id)[0]
    if len(eos_idx) > 0:
        seq = seq[:eos_idx[0]]
    decoded_text = tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    post_processed_text = ip.postprocess_batch([decoded_text])[0]
    generated_sequences.append(post_processed_text)

# Print results
print("\nInput Sentences and Translations:")
for input_sent, translation in zip(input_sentences, generated_sequences):
    print(f"{src_lang}: {input_sent}")
    print(f"{tgt_lang}: {translation}")

print("Inference completed successfully on GPU.")