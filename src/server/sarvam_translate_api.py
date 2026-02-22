import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = "cuda"
# Load model and tokenizer on startup
MODEL_NAME = "sarvamai/sarvam-translate"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32  # Full precision to avoid CUDA issues
    ).to(DEVICE)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

app = FastAPI()

class TranslationRequest(BaseModel):
    sentences: list[str]
    src_lang: str
    tgt_lang: str

class TranslationResponse(BaseModel):
    translations: list[str]

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    input_sentences = request.sentences
    src_lang = request.src_lang
    tgt_lang = request.tgt_lang

    # Validate input
    if not input_sentences:
        raise HTTPException(status_code=400, detail="Input sentences are required")
    if not all(isinstance(s, str) and s.strip() for s in input_sentences):
        raise HTTPException(status_code=400, detail="All sentences must be non-empty strings")

    translations = []
    try:
        for sentence in input_sentences:
            # Create prompt (adjusted for sarvam-translate)
            prompt = f"Translate the following text from {src_lang} to {tgt_lang}: {sentence}"
            # Tokenize input
            model_inputs = tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(DEVICE)
            # Generate translation
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=256,  # Reduced for stability
                    do_sample=False,     # Greedy decoding to avoid CUDA errors
                    num_beams=4,         # Moderate beam search
                    num_return_sequences=1
                )
            # Decode output
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            translations.append(output_text.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

    return TranslationResponse(translations=translations)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7862, log_level="info")