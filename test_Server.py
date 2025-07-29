import requests
import os

# Set API key and base URL
base_url = os.getenv("TRANSLATE_API_BASE_URL")


def translate_text(sentence, src_lang="eng_Latn", tgt_lang="kan_Knda"):
    
    
    url = f"{base_url}?src_lang={src_lang}&tgt_lang={tgt_lang}"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "sentences": [sentence],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":


    file_name = "data/shrimad_bhagavatam_english.txt"
    # Open the file in read mode
    with open(file_name, 'r') as file:
        file_content = file.read()

    # Now file_content holds the entire content of the file as a string
    print(file_content)
    
    text = "hi"
    #result = translate_text(file_content)

    result = translate_text(text)


    print(result)