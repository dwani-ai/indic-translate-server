# Indic Translate Server

## Table of Contents

- [Overview](#overview)
- [Live Server](#live-server)
- [Prerequisites](#prerequisites)
- [Running with Docker Compose](#running-with-docker-compose)
- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Downloading Translation Models](#downloading-translation-models)
- [Running with FastAPI Server](#running-with-fastapi-server)
- [Evaluating Results](#evaluating-results)
- [Build Docker Image](#build-docker-image)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [FAQ](#faq)

## Overview

This project sets up an Indic translation server using Docker Compose, allowing translation between various languages including English, Kannada, Hindi, and others. It utilizes models from AI4Bharat to perform translations.

### Live Server

We have hosted an Translation service for Indian languages. The service is available in two modes:

#### High Latency, Slow System (Available 24/7)
- **URL**: [High Latency ASR Service](https://huggingface.co/spaces/gaganyatri/translate_indic_server_cpu)

#### Low Latency, Fast System (Available on Request)
- **URL**: [Low Latency ASR Service](https://huggingface.co/spaces/gaganyatri/translate_indic_server)

### How to Use the Service

1. With curl

You can test the service using `curl` commands. Below are examples for both service modes:

#### High Latency Service


```
curl -X 'POST' \
  'https://gaganyatri-translate-indic-server-cpu.hf.space/translate?src_lang=kan_Knda&tgt_lang=eng_Latn&device_type=cpu' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
     "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?", "ಶುಭೋದಯ!"
  ],
  "src_lang": "kan_Knda",
  "tgt_lang": "eng_Latn"
}'
```

#### Low Latency Service - GPU server on demand
```
curl -X 'POST' \
  'https://gaganyatri-translate-indic-server.hf.space/translate?src_lang=kan_Knda&tgt_lang=eng_Latn&device_type=gpu' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
     "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?", "ಶುಭೋದಯ!"
  ],
  "src_lang": "kan_Knda",
  "tgt_lang": "eng_Latn"
}'
```



2. Via Swagger UI 

- **URL**: [High Latency translation Service](https://huggingface.co/spaces/gaganyatri/translate_indic_server_cpu)

- **URL**: [Low Latency translation Service](https://huggingface.co/spaces/gaganyatri/translate_indic_server_cpu)



## Prerequisites

- Docker and Docker Compose installed on your machine.
- Python 3.x installed for the development environment.
- Internet access to download translation models.

## Running with Docker Compose

1. **Start the server:**
   ```bash
   docker compose -f compose.yaml up -d
   ```

2. **Update source and target languages:**
   Modify the `compose.yaml` file to set the source (`SRC_LANG`) and target (`TGT_LANG`) languages as per your requirements. Example configurations:
   - **English to Indic:**
     ```yaml
     SRC_LANG: eng_Latn
     TGT_LANG: kan_Knda
     ```
   - **Indic to English:**
     ```yaml
     SRC_LANG: kan_Knda
     TGT_LANG: eng_Latn
     ```
   - **Indic to Indic:**
     ```yaml
     SRC_LANG: kan_Knda
     TGT_LANG: hin_Deva
     ```



## Setting Up the Development Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Downloading Translation Models

Models can be downloaded from AI4Bharat's HuggingFace repository:

### Indic to English
```bash
huggingface-cli download ai4bharat/indictrans2-indic-en-dist-200M
```

### English to Indic
```bash
huggingface-cli download ai4bharat/indictrans2-en-indic-1B
```

### Indic to Indic
```bash
huggingface-cli download ai4bharat/indictrans2-indic-indic-dist-320M
```

## Running with FastAPI Server

You can run the server using FastAPI:
1. with GPU 
```bash
python src/translate_api.py --src_lang kan_Knda --tgt_lang eng_Latn --port 7860 --host 0.0.0.0 --device cuda
```

2. with CPU only
```bash
python src/translate_api.py --src_lang kan_Knda --tgt_lang eng_Latn --port 7860 --host 0.0.0.0 --device cpu
```

## Evaluating Results

You can evaluate the translation results using `curl` commands. Here are some examples:

### English to Kannada
```bash
curl -X 'POST' \
  'http://localhost:7860/translate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
    "Hello, how are you?", "Good morning!"
  ],
  "src_lang": "eng_Latn",
  "tgt_lang": "kan_Knda"
}'
```

**Response:**
```json
{
  "translations": [
    "ಹಲೋ, ಹೇಗಿದ್ದೀರಿ? ",
    "ಶುಭೋದಯ! "
  ]
}
```

### Kannada to English

```
curl -X 'POST' \
  'http://localhost:7860/translate?src_lang=kan_Knda&tgt_lang=eng_Latn&device_type=cuda' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
    "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?", "ಶುಭೋದಯ!"
  ],
  "src_lang": "kan_Knda",
  "tgt_lang": "eng_Latn"
}'
```


**Response:**
```json
{
  "translations": ["Hello, how are you?", "Good morning!"]
}
```

```
curl -X 'POST' \
  'http://localhost:7860/translate?src_lang=kan_Knda&tgt_lang=eng_Latn&device_type=cuda' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
     "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?", "ಶುಭೋದಯ!"
  ],
  "src_lang": "kan_Knda",
  "tgt_lang": "eng_Latn"
}'
```

Response 
{
  "translations": [
    "Hello, how are you?",
    "Good morning!"
  ]
}



----

cpu

curl -X 'POST' \
  'http://localhost:7860/translate?src_lang=kan_Knda&tgt_lang=eng_Latn&device_type=cpu' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentences": [
    "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?", "ಶುಭೋದಯ!"
  ],
  "src_lang": "kan_Knda",
  "tgt_lang": "eng_Latn"
}'

{
  "translations": [
    "Hello, how are you?",
    "Good morning!"
  ]
}




## Build Docker Image
1. GPU 
```bash
docker build -t slabstech/indic_translate_server -f Dockerfile .
```
2. CPU only
```bash
docker build -t slabstech/indic_translate_server_ -f Dockerfile.cpu .
```



## References

- [AI4Bharat IndicTrans2 Model](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M)
- [AI4Bharat IndicTrans2 GitHub Repository](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface)
- [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit.git)
- Extra - pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## FAQ

**Q: How do I change the source and target languages?**

A: Modify the `compose.yaml` file to set the `SRC_LANG` and `TGT_LANG` variables as needed.

**Q: How do I download the translation models?**

A: Use the `huggingface-cli` commands provided in the [Downloading Translation Models](#downloading-translation-models) section.

**Q: How do I run the server locally?**

A: Follow the instructions in the [Running with FastAPI Server](#running-with-fastapi-server) section.

---

This README provides a comprehensive guide to setting up and running the Indic Translate Server. For more details, refer to the linked resources.