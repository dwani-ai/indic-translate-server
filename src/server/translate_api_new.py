import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Union
import argparse
import os
from fastapi.responses import RedirectResponse
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Language options with names and codes
language_options = [
    ("English", "eng_Latn"),
    ("Kannada", "kan_Knda"),
    ("Hindi", "hin_Deva"),
    ("Assamese", "asm_Beng"),
    ("Bengali", "ben_Beng"),
    ("Gujarati", "guj_Gujr"),
    ("Malayalam", "mal_Mlym"),
    ("Marathi", "mar_Deva"),
    ("Odia", "ory_Orya"),
    ("Punjabi", "pan_Guru"),
    ("Tamil", "tam_Taml"),
    ("Telugu", "tel_Telu"),
    ("German", "deu_Latn"),
]

# Create mappings for language names and codes (case-insensitive)
name_to_code = {name.lower(): code for name, code in language_options}
code_to_name = {code.lower(): name for name, code in language_options}
valid_names = set(name_to_code.keys())
valid_codes = set(code_to_name.keys())

model_name = "sarvamai/sarvam-translate"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

app = FastAPI()

class TranslationRequest(BaseModel):
    sentences: Union[str, List[str]]  # Accept either a string or list of strings
    src_lang: str  # Can be language name (e.g., "English", "english") or code (e.g., "eng_Latn", "ENG_LATN")
    tgt_lang: str  # Can be language name (e.g., "Hindi", "hindi") or code (e.g., "hin_Deva", "HIN_DEVA")

class TranslationResponse(BaseModel):
    translations: Union[str, List[str]]  # Return a string or list of strings based on input

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest, src_lang: str = Query(...), tgt_lang: str = Query(...)):
    try:
        # Clean and normalize language inputs
        src_lang = request.src_lang.lower().strip()
        tgt_lang = request.tgt_lang.lower().strip()

        # Validate language inputs
        if not src_lang or not tgt_lang:
            raise HTTPException(status_code=400, detail="Source and target languages must be provided")
        if src_lang == tgt_lang:
            raise HTTPException(status_code=400, detail="Source and target languages must be different")

        # Convert language inputs to names and codes
        if src_lang in valid_names:
            src_name = code_to_name.get(src_lang, next(name for name, _ in language_options if name.lower() == src_lang))
            src_code = name_to_code[src_lang]
        elif src_lang in valid_codes:
            src_name = code_to_name[src_lang]
            src_code = src_lang
        else:
            raise HTTPException(status_code=400, detail=f"Invalid source language. Supported: {', '.join(name for name, _ in language_options)} or their codes.")

        if tgt_lang in valid_names:
            tgt_name = code_to_name.get(tgt_lang, next(name for name, _ in language_options if name.lower() == tgt_lang))
            tgt_code = name_to_code[tgt_lang]
        elif tgt_lang in valid_codes:
            tgt_name = code_to_name[tgt_lang]
            tgt_code = tgt_lang
        else:
            raise HTTPException(status_code=400, detail=f"Invalid target language. Supported: {', '.join(name for name, _ in language_options)} or their codes.")

        # Validate that one language is English
        if src_name.lower() != "english" and tgt_name.lower() != "english":
            raise HTTPException(status_code=400, detail="One of source or target language must be English")

        # Handle string or list of strings
        input_sentences = request.sentences
        if isinstance(input_sentences, str):
            input_sentences = [input_sentences]
        elif not input_sentences:
            raise HTTPException(status_code=400, detail="Input sentences cannot be empty")

        # Validate each sentence
        for sentence in input_sentences:
            if not sentence.strip():
                raise HTTPException(status_code=400, detail="Input sentences cannot contain empty strings")

        # Translate each sentence
        translations = []
        for sentence in input_sentences:
            # Prepare chat-style message prompt using language names
            messages = [
                {"role": "system", "content": f"Translate the text below to {tgt_name}."},
                {"role": "user", "content": sentence.strip()}
            ]

            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize and move input to model device, including language codes
            text_with_codes = f"[{src_code} to {tgt_code}] {text}"
            model_inputs = tokenizer([text_with_codes], return_tensors="pt").to(model.device)

            # Generate the output
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.01,
                num_return_sequences=1,
                use_cache=False
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

            print(f"Input: {sentence}")
            print(f"Translation: {output_text}")
            translations.append(output_text)

        # Return single string or list based on input type
        return TranslationResponse(translations=translations[0] if len(translations) == 1 else translations)

    except Exception as e:
        print(f"Error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Translation Server")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")