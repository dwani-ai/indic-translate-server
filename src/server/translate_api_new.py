import torch
from fastapi import FastAPI, HTTPException, Depends
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


# Recommended to run this on a GPU with flash_attn installed
# Don't set attn_implementation if you don't have flash_attn

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
async def translate(request: TranslationRequest):
    try:
        input_sentences = request.sentences
        src_lang = request.src_lang
        tgt_lang = request.tgt_lang

        if(src_lang == "english" or tgt_lang == "english"): 
                # Translation task
            #tgt_lang = "Hindi"
            #input_txt = "Be the change you wish to see in the world."

            # Chat-style message prompt
            messages = [
                {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
                {"role": "user", "content": str(input_sentences)}
            ]

            #print(messages)
            # Apply chat template to structure the conversation
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
                num_return_sequences=1
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

            print("Input:", str(input_sentences))
            print("Translation:", output_text)


            if not input_sentences:
                raise HTTPException(status_code=400, detail="Input sentences are required")

            return TranslationResponse(translations=output_text)
    except Exception:
        print("exceptin")
# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Translation Server")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    return parser.parse_args()

# Run the server using Uvicorn
if __name__ == "__main__":
    args = parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")