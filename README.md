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

###  Languages Supported
Here is the list of languages supported by the IndicTrans2 models:

<table>
<tbody>
  <tr>
    <td>Assamese (asm_Beng)</td>
    <td>Kashmiri (Arabic) (kas_Arab)</td>
    <td>Punjabi (pan_Guru)</td>
  </tr>
  <tr>
    <td>Bengali (ben_Beng)</td>
    <td>Kashmiri (Devanagari) (kas_Deva)</td>
    <td>Sanskrit (san_Deva)</td>
  </tr>
  <tr>
    <td>Bodo (brx_Deva)</td>
    <td>Maithili (mai_Deva)</td>
    <td>Santali (sat_Olck)</td>
  </tr>
  <tr>
    <td>Dogri (doi_Deva)</td>
    <td>Malayalam (mal_Mlym)</td>
    <td>Sindhi (Arabic) (snd_Arab)</td>
  </tr>
  <tr>
    <td>English (eng_Latn)</td>
    <td>Marathi (mar_Deva)</td>
    <td>Sindhi (Devanagari) (snd_Deva)</td>
  </tr>
  <tr>
    <td>Konkani (gom_Deva)</td>
    <td>Manipuri (Bengali) (mni_Beng)</td>
    <td>Tamil (tam_Taml)</td>
  </tr>
  <tr>
    <td>Gujarati (guj_Gujr)</td>
    <td>Manipuri (Meitei) (mni_Mtei)</td>
    <td>Telugu (tel_Telu)</td>
  </tr>
  <tr>
    <td>Hindi (hin_Deva)</td>
    <td>Nepali (npi_Deva)</td>
    <td>Urdu (urd_Arab)</td>
  </tr>
  <tr>
    <td>Kannada (kan_Knda)</td>
    <td>Odia (ory_Orya)</td>
    <td></td>
  </tr>
</tbody>
</table>




### Live Server

We have hosted an Translation service for Indian languages. The service is available in two modes:

#### Available 24/7 - Free, Slow 
- **URL**: [High Latency ASR Service](https://huggingface.co/spaces/gaganyatri/translate_indic_server_cpu)

#### Paused, On-demand, $.05 /hour, Fast
- **URL**: [Low Latency ASR Service](https://huggingface.co/spaces/gaganyatri/translate_indic_server)

### How to Use the Service

1. With curl

You can test the service using `curl` commands. Below are examples for both service modes:

#### Available 24/7 - Free, Slow

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

#### Paused, On-demand, $.05 /hour, Fast
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



### Model Downloads for Translation

- Collection Models on HuggingFace - [IndicTrans2](https://huggingface.co/collections/ai4bharat/indictrans2-664ccb91d23bbae0d681c3ca)

Below is a table summarizing the available models for different translation tasks:

| Task                | Variant                | Model Name                                     | VRAM Size  | Download Command                                               |
|---------------------|------------------------|-----------------------------------------------|------------|----------------------------------------------------------------|
| Indic to English    | 200M (distilled)                  | indictrans2-indic-en-dist-200M                 | 950 MB     | `huggingface-cli download ai4bharat/indictrans2-indic-en-dist-200M`  |
|                     | 1B (base)                     | indictrans2-indic-en-1B                       | 4.5 GB     | `huggingface-cli download ai4bharat/indictrans2-indic-en-1B`       |
| English to Indic    | 200M (distilled)       | indictrans2-en-indic-dist-200M                 | 950 MB     | `huggingface-cli download ai4bharat/indictrans2-en-indic-dist-200M`  |
|                     | 1B (base)              | indictrans2-en-indic-1B                       | 4.5 GB     | `huggingface-cli download ai4bharat/indictrans2-en-indic-1B`       |
| Indic to Indic      | 320M (distilled)       | indictrans2-indic-indic-dist-320M             | 950 MB     | `huggingface-cli download ai4bharat/indictrans2-indic-indic-dist-320M`|
|                     | 1B (base)              | indictrans2-indic-indic-1B                   | 4.5 GB     | `huggingface-cli download ai4bharat/indictrans2-indic-indic-1B`     |

## Running with FastAPI Server

You can run the server using FastAPI:
1. with GPU 
```bash
python src/translate_api.py --port 7860 --host 0.0.0.0 --device cuda --use_distilled
```

2. with CPU only
```bash
python src/translate_api.py --port 7860 --host 0.0.0.0 --device cpu --use_distilled
```

## Evaluating Results

You can evaluate the translation results using `curl` commands. Here are some examples:

### English to Kannada
```bash
curl -X 'POST' \
  'http://localhost:7860/translate?tgt_lang=kan_Knda&src_lang=eng_Latn&device_type=cuda' \
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


### Kannada to Hindi 
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

### Response


### Hindi to Kannada

----

### CPU
```
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
```

### Response
```
{
  "translations": [
    "Hello, how are you?",
    "Good morning!"
  ]
}

```


## Build Docker Image
1. GPU 
```bash
docker build -t slabstech/indic_translate_server -f Dockerfile .
```
2. CPU only
```bash
docker build -t slabstech/indic_translate_server_cpu -f Dockerfile.cpu .
```



## References
- [IndicTrans2 Paper](https://openreview.net/pdf?id=vfT4YuzAYA)
- [AI4Bharat IndicTrans2 Model](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M)
- [AI4Bharat IndicTrans2 GitHub Repository](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface)
- [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit.git)
- Extra - pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

## Contributing

We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

Also you can join the [discord group](https://discord.gg/WZMCerEZ2P) to collaborate

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
#### License

- [IndicTrans License](https://github.com/AI4Bharat/IndicTrans2?tab=MIT-1-ov-file#readme)
- [IndicTrans Data License](https://github.com/AI4Bharat/IndicTrans2?tab=readme-ov-file#license)


--- 
This README provides a comprehensive guide to setting up and running the Indic Translate Server. For more details, refer to the linked resources.


## Citation

```bibtex
@article{gala2023indictrans,
title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
author={Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=vfT4YuzAYA},
note={}
}
```
