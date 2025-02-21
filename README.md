Indic Translate Server

- Run With Docker compose 
  - docker compose -f compose.yaml up -d

- Evaluate result

    - English to Kannada
    ```
    curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{
           "sentences": ["Hello, how are you?", "Good morning!"],
           "src_lang": "eng_Latn",
           "tgt_lang": "kan_Knda"
         }'
    ```


    - Kannada to English
    ```
    curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{
           "sentences": ["ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಾ?", "ಶುಭೋದಯ!"],
           "src_lang": "kan_Knda",
           "tgt_lang": "eng_Latn"
         }'
    ```

    - Hindi to English
    ```
    curl -X POST "http://localhost:8000/translate" \
        -H "Content-Type: application/json" \
        -d '{
            "sentences": ["नमस्ते दुनिया", "आप कैसे हो?"],
            "src_lang": "hin_Deva",
            "tgt_lang": "eng_Latn"
            }'
    ```


- Setup Dev Environment
  - python -m venv venv
  - source venv/bin/activate
  - pip install -r requirements.txt

- Download model from AI4Bharat for Translation
  - Indic to English
      - huggingface-cli download ai4bharat/indictrans2-indic-en-dist-200M

  - English to Indic
      - huggingface-cli download ai4bharat/indictrans2-en-indic-1B
  - Indic to Indic
    - huggingface-cli download ai4bharat/indictrans2-indic-indic-dist-320M

- Run with fastapi server 
  - ``` uvicorn indic_translate_server/translate_api:app --host 0.0.0.0 --port 8000```
    - or
  - ```python indic_translate_server/translate_api.py --src_lang eng_Latn --tgt_lang kan_Knda```



- Reference
    - https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
    - https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface
    - pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

