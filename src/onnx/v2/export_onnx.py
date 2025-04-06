import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    raise RuntimeError("This script requires a GPU for export.")

# Model and language setup
src_lang, tgt_lang = "hin_Deva", "eng_Latn"
model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
num_beams = 5

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to(DEVICE)

# Initialize IndicProcessor
ip = IndicProcessor(inference=True)

# Input sentences
input_sentences = [
    "जब मैं छोटा था, मैं हर रोज़ पार्क जाता था।",
    "हमने पिछले सप्ताह एक नई फिल्म देखी जो कि बहुत प्रेरणादायक थी।",
    "अगर तुम मुझे उस समय पास मिलते, तो हम बाहर खाना खाने चलते।",
    "मेरे मित्र ने मुझे उसके जन्मदिन की पार्टी में बुलाया है, और मैं उसे एक तोहफा दूंगा।",
]

# Preprocess and tokenize with beam search batch size
batch_size = len(input_sentences)
effective_batch_size = batch_size * num_beams  # Account for beams
batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)
inputs = tokenizer(
    batch,
    truncation=True,
    padding="longest",
    return_tensors="pt",
    return_attention_mask=True,
).to(DEVICE)

# Expand encoder inputs for beams
inputs["input_ids"] = inputs["input_ids"].repeat(num_beams, 1)
inputs["attention_mask"] = inputs["attention_mask"].repeat(num_beams, 1)

# Create dummy decoder_input_ids
decoder_input_ids = torch.full(
    (effective_batch_size, 1),
    tokenizer.bos_token_id or 0,
    dtype=torch.long,
    device=DEVICE
)

# Export to ONNX
onnx_model_path = "indictrans2_indic_en_beam.onnx"
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"], decoder_input_ids),
    onnx_model_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "decoder_input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    },
    verbose=True
)

print(f"Model exported to {onnx_model_path}")