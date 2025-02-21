import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from typing import List
import argparse

import uvicorn
# recommended to run this on a gpu with flash_attn installed
# don't set attn_implemetation if you don't have flash_attn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the default source and target languages
src_lang = "eng_Latn"
tgt_lang = "kan_Knda"

# Function to initialize the model based on the source and target languages
def initialize_model(src_lang, tgt_lang):
    # Determine the model name based on the source and target languages
    if src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
        model_name = "ai4bharat/indictrans2-en-indic-dist-200M"  # English to Indic
    elif not src_lang.startswith("eng") and tgt_lang.startswith("eng"):
        model_name = "ai4bharat/indictrans2-indic-en-dist-200M"  # Indic to English
    elif not src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
        model_name = "ai4bharat/indictrans2-indic-indic-dist-320M"  # Indic to Indic
    else:
        raise ValueError("Invalid language combination: English to English translation is not supported.")

    # Now model_name contains the correct model based on the source and target languages
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # performance might slightly vary for bfloat16
        attn_implementation="flash_attention_2"
    ).to(DEVICE)

    return tokenizer, model

# Initialize the model with default languages
tokenizer, model = initialize_model(src_lang, tgt_lang)

ip = IndicProcessor(inference=True)

app = FastAPI()

class TranslationRequest(BaseModel):
    sentences: List[str]
    src_lang: str
    tgt_lang: str

class TranslationResponse(BaseModel):
    translations: List[str]

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    input_sentences = request.sentences
    src_lang = request.src_lang
    tgt_lang = request.tgt_lang

    if not input_sentences:
        raise HTTPException(status_code=400, detail="Input sentences are required")

    batch = ip.preprocess_batch(
        input_sentences,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    # Tokenize the sentences and generate input encodings
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    # Generate translations using the model
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode the generated tokens into text
    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess the translations, including entity replacement
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

    return TranslationResponse(translations=translations)

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Translation Server")
    parser.add_argument("--src_lang", type=str, default="eng_Latn", help="Source language code")
    parser.add_argument("--tgt_lang", type=str, default="kan_Knda", help="Target language code")
    return parser.parse_args()

# Run the server using Uvicorn
if __name__ == "__main__":
    args = parse_args()
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang

    # Reinitialize the model with the provided languages
    tokenizer, model = initialize_model(src_lang, tgt_lang)

    uvicorn.run(app, host="0.0.0.0", port=8000)