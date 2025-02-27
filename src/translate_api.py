import torch
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from typing import List
import argparse
import os
from fastapi.responses import RedirectResponse
import uvicorn

# Recommended to run this on a GPU with flash_attn installed
# Don't set attn_implemetation if you don't have flash_attn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TranslateManager:
    def __init__(self, src_lang, tgt_lang, device_type=DEVICE):
        self.device_type = device_type
        self.tokenizer, self.model = self.initialize_model(src_lang, tgt_lang)

    def initialize_model(self, src_lang, tgt_lang):
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
        ).to(self.device_type)

        return tokenizer, model

ip = IndicProcessor(inference=True)

app = FastAPI()

class TranslationRequest(BaseModel):
    sentences: List[str]
    src_lang: str
    tgt_lang: str

class TranslationResponse(BaseModel):
    translations: List[str]

def get_translate_manager(src_lang: str, tgt_lang: str, device_type: str) -> TranslateManager:
    return TranslateManager(src_lang, tgt_lang, device_type)

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest, translate_manager: TranslateManager = Depends(get_translate_manager)):
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
    inputs = translate_manager.tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(translate_manager.device_type)

    # Generate translations using the model
    with torch.no_grad():
        generated_tokens = translate_manager.model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode the generated tokens into text
    with translate_manager.tokenizer.as_target_tokenizer():
        generated_tokens = translate_manager.tokenizer.batch_decode(
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
    parser.add_argument("--src_lang", type=str, default=os.getenv('SRC_LANG', 'eng_Latn'), help="Source language code")
    parser.add_argument("--tgt_lang", type=str, default=os.getenv('TGT_LANG', 'kan_Knda'), help="Target language code")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    return parser.parse_args()

# Run the server using Uvicorn
if __name__ == "__main__":
    args = parse_args()
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    device_type = args.device

    # Initialize the model with the provided languages
    translate_manager = TranslateManager(src_lang, tgt_lang, device_type)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")