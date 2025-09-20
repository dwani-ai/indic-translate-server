import torch
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import argparse
import uvicorn
from fastapi.responses import RedirectResponse

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TranslateManager:
    def __init__(self, device_type=DEVICE):
        self.device_type = device_type
        self.tokenizer, self.model = self.initialize_model()

    def initialize_model(self):
        model_name = "sarvamai/sarvam-translate"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(self.device_type)
        return tokenizer, model

    def translate(self, sentences: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        translations = []
        for sentence in sentences:
            # Create chat-style prompt
            messages = [
                {"role": "system", "content": f"Translate the text below to {tgt_lang}."},
                {"role": "user", "content": sentence}
            ]
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device_type)
            # Generate translation
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.01,
                    num_return_sequences=1
                )
            # Decode output
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            translations.append(output_text)
        return translations

class ModelManager:
    def __init__(self, device_type=DEVICE, is_lazy_loading=False):
        self.device_type = device_type
        self.is_lazy_loading = is_lazy_loading
        self.model = None
        if not is_lazy_loading:
            self.model = TranslateManager(device_type)

    def get_model(self) -> TranslateManager:
        if self.model is None and self.is_lazy_loading:
            self.model = TranslateManager(self.device_type)
        return self.model

app = FastAPI()
model_manager = ModelManager()

class TranslationRequest(BaseModel):
    sentences: List[str]
    src_lang: str
    tgt_lang: str

class TranslationResponse(BaseModel):
    translations: List[str]

def get_translate_manager() -> TranslateManager:
    return model_manager.get_model()

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

    # Perform translation
    translations = translate_manager.translate(input_sentences, src_lang, tgt_lang)
    return TranslationResponse(translations=translations)

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Translation Server")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device type to run the model on (cuda or cpu).")
    parser.add_argument("--is_lazy_loading", action="store_true", help="Enable lazy loading of models.")
    return parser.parse_args()

# Run the server using Uvicorn
if __name__ == "__main__":
    args = parse_args()
    device_type = args.device
    is_lazy_loading = args.is_lazy_loading

    # Initialize the model manager
    model_manager = ModelManager(device_type, is_lazy_loading)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")