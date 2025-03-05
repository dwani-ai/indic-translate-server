import gradio as gr
import os
import requests
import json
import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import spaces
# Set up logging
logging.basicConfig(filename='execution.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapping of user-friendly language names to language IDs
language_mapping = {
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bodo": "brx_Deva",
    "Dogri": "doi_Deva",
    "English": "eng_Latn",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva",
    "Konkani": "gom_Deva",
    "Malayalam": "mal_Mlym",
    "Manipuri (Bengali)": "mni_Beng",
    "Manipuri (Meitei)": "mni_Mtei",
    "Maithili": "mai_Deva",
    "Marathi": "mar_Deva",
    "Nepali": "npi_Deva",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi (Arabic)": "snd_Arab",
    "Sindhi (Devanagari)": "snd_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab"
}

# Initialize the model and tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # performance might slightly vary for bfloat16
    attn_implementation="flash_attention_2"
).to(DEVICE)
ip = IndicProcessor(inference=True)

def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

@spaces.ZERO_GPU
def translate_text(transcription, src_lang, tgt_lang, use_gpu=False):
    logging.info(f"Translating text: {transcription}, src_lang: {src_lang}, tgt_lang: {tgt_lang}, use_gpu: {use_gpu}")

    chunk_size = 15
    chunked_text = chunk_text(transcription, chunk_size=chunk_size)

    batch = ip.preprocess_batch(
        chunked_text,
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
    merged_translated_text = ' '.join(translations)
    logging.info(f"Translation successful: {merged_translated_text}")
    return {'translations': [merged_translated_text]}

# Create the Gradio interface
with gr.Blocks(title="Dhwani Translate - Convert to any Language") as demo:
    gr.Markdown("# Text Translate - To any language")

    translate_src_language = gr.Dropdown(
        choices=list(language_mapping.keys()),
        label="Source Language - Fixed",
        value="Kannada",
        interactive=False
    )
    translate_tgt_language = gr.Dropdown(
        choices=list(language_mapping.keys()),
        label="Target Language",
        value="English"
    )

    with gr.Row():
        input_text = gr.Textbox(label="Enter Text", placeholder="Type your text here...")

    submit_button = gr.Button("Translate Text")

    translation_output = gr.Textbox(label="Translated Text", interactive=False)

    use_gpu_checkbox = gr.Checkbox(label="Use GPU", value=False, interactive=False, visible=False)

    def on_transcription_complete(transcription, src_lang, tgt_lang, use_gpu):
        src_lang_id = language_mapping[src_lang]
        tgt_lang_id = language_mapping[tgt_lang]
        logging.info(f"Transcription complete: {transcription}, src_lang: {src_lang_id}, tgt_lang: {tgt_lang_id}, use_gpu: {use_gpu}")
        translation = translate_text(transcription, src_lang_id, tgt_lang_id, use_gpu)
        translated_text = translation['translations'][0]
        return translated_text

    submit_button.click(
        on_transcription_complete,
        inputs=[input_text, translate_src_language, translate_tgt_language, use_gpu_checkbox],
        outputs=[translation_output]
    )

# Launch the interface
demo.launch()