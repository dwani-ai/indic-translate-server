import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    raise RuntimeError("This script requires a GPU.")
print(f"Using device: {DEVICE}")

# Model and language setup
src_lang, tgt_lang = "hin_Deva", "eng_Latn"
model_name = "ai4bharat/indictrans2-indic-en-dist-200M"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"  # Uncomment if flash-attn is installed
).to(DEVICE)

# Compile model for speed (PyTorch 2.0+)
if torch.__version__.startswith("2"):
    model = torch.compile(model, mode="reduce-overhead")
    print("Model compiled with torch.compile")

# Initialize IndicProcessor
ip = IndicProcessor(inference=True)

# Input sentences
input_sentences = [
    "जब मैं छोटा था, मैं हर रोज़ पार्क जाता था।",
    "हमने पिछले सप्ताह एक नई फिल्म देखी जो कि बहुत प्रेरणादायक थी।",
    "अगर तुम मुझे उस समय पास मिलते, तो हम बाहर खाना खाने चलते।",
    "मेरे मित्र ने मुझे उसके जन्मदिन की पार्टी में बुलाया है, और मैं उसे एक तोहफा दूंगा।",
]

# Preprocess and tokenize
batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)
inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True,
).to(DEVICE)

# Generate translations
with torch.no_grad():
    generated_tokens = model.generate(
        **inputs,
        use_cache=True,
        min_length=0,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
        early_stopping=True,  # Stop when all beams hit EOS
    )

# Decode
with tokenizer.as_target_tokenizer():
    generated_tokens = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

# Post-process
translations = ip.postprocess_batch(generated_tokens)

# Print results
print("\nInput Sentences and Translations:")
for input_sent, translation in zip(input_sentences, translations):
    print(f"{src_lang}: {input_sent}")
    print(f"{tgt_lang}: {translation}")

print("Inference completed successfully on GPU.")