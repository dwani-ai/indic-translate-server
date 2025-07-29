import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict
import argparse
import os
from fastapi.responses import RedirectResponse
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "sarvamai/sarvam-translate"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

app = FastAPI()

class TranslationRequest(BaseModel):
    sentences: str
    src_lang: str
    tgt_lang: str

class TranslationResponse(BaseModel):
    translations: str

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest,src_lang:str= Query(...), tgt_lang:str= Query(...)):
    try:
        input_sentences = request.sentences.strip()  # Remove leading/trailing whitespace
        src_lang = request.src_lang.lower().strip()
        tgt_lang = request.tgt_lang.lower().strip()

        # Validate inputs
        if not input_sentences:
            raise HTTPException(status_code=400, detail="Input sentences cannot be empty")
        if not src_lang or not tgt_lang:
            raise HTTPException(status_code=400, detail="Source and target languages must be provided")
        if src_lang == tgt_lang:
            raise HTTPException(status_code=400, detail="Source and target languages must be different")
        if src_lang != "english" and tgt_lang != "english":
            raise HTTPException(status_code=400, detail="One of source or target language must be English")

        # Prepare chat-style message prompt
        messages = [
            {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
            {"role": "user", "content": input_sentences}
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and move input to model device
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

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

        print(f"Input: {input_sentences}")
        print(f"Translation: {output_text}")

        return TranslationResponse(translations=output_text)

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during translation: {str(e)}")
        # Raise an HTTPException with a meaningful message
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