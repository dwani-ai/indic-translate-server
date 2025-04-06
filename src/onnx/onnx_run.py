import onnxruntime as ort
from IndicTransToolkit.processor import IndicProcessor

ort_session = ort.InferenceSession("indictrans_model.onnx")



# Load the model and tokenizer
ip = IndicProcessor(inference=True)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Prepare dummy input for tracing
sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]
batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva", visualize=False)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
batch = tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt")

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

# Define input and output names for better readability
input_names = ["input_ids", "attention_mask"]
output_names = ["output"]


print("Model has been successfully exported to ONNX format.")


# Prepare inputs for inference
inputs = {
    "input_ids": batch["input_ids"].numpy(),
    "attention_mask": batch["attention_mask"].numpy(),
}

outputs = ort_session.run(None, inputs)

print(outputs)
