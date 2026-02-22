import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Select device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name: str):
    """Load model and tokenizer for the given checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    ).to(DEVICE)
    return model, tokenizer

def translate_sentences(input_sentences, src_lang, tgt_lang, model_name):
    """Perform translation using IndicTrans2 model."""
    # Load tokenizer and model
    model, tokenizer = load_model(model_name)
    ip = IndicProcessor(inference=True)

    # Preprocess input batch
    batch = ip.preprocess_batch(
        input_sentences,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    # Tokenize
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
        )

    # Decode tokens
    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess
    translations = ip.postprocess_batch(decoded, lang=tgt_lang)
    return translations

# Example usage
if __name__ == "__main__":
    sentences_hin = [
        "जब मैं छोटा था, मैं हर रोज़ पार्क जाता था।",
        "हमने पिछले सप्ताह एक नई फिल्म देखी जो कि बहुत प्रेरणादायक थी।",
        "अगर तुम मुझे उस समय पास मिलते, तो हम बाहर खाना खाने चलते।",
        "मेरे मित्र ने मुझे उसके जन्मदिन की पार्टी में बुलाया है, और मैं उसे एक तोहफा दूंगा।",
    ]

    # Hindi → Tamil
    hin_to_tam_model = "ai4bharat/indictrans2-indic-indic-dist-320M"
    translations_tam = translate_sentences(sentences_hin, "hin_Deva", "tam_Taml", hin_to_tam_model)
    print("--- Hindi → Tamil ---")
    for src, tgt in zip(sentences_hin, translations_tam):
        print(f"hin_Deva: {src}\n tam_Taml: {tgt}\n")

    # Hindi → English
    hin_to_eng_model = "ai4bharat/indictrans2-indic-en-dist-200M"
    translations_eng = translate_sentences(sentences_hin[:2], "hin_Deva", "eng_Latn", hin_to_eng_model)
    print("--- Hindi → English ---")
    for src, tgt in zip(sentences_hin[:2], translations_eng):
        print(f"hin_Deva: {src}\n eng_Latn: {tgt}\n")
