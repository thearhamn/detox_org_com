from packaging import version
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
from tqdm import tqdm

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('s-nlp/mt0-xl-detox-orpo')
tokenizer = AutoTokenizer.from_pretrained('s-nlp/mt0-xl-detox-orpo')

LANG_PROMPTS = {
   'zh': '排毒：',
   'es': 'Desintoxicar: ',
   'ru': 'Детоксифицируй: ',
   'ar': 'إزالة السموم: ',
   'hi': 'विषहरण: ',
   'uk': 'Детоксифікуй: ',
   'de': 'Entgiften: ',
   'am': 'መርዝ መርዝ: ',
   'en': 'Detoxify: ',
}

def detoxify(text, lang, model, tokenizer):
    encodings = tokenizer(LANG_PROMPTS[lang] + text, return_tensors='pt').to(model.device)
    _TRANSFORMERS_VERSION = version.parse(transformers.__version__)
    gen_kwargs = {
        "max_length": 128,
        "num_beams": 10,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
        "num_return_sequences": 5,
        "early_stopping": True
    }
    # Adapt generation parameters for older transformers versions
    if _TRANSFORMERS_VERSION < version.parse("4.56.0"):
        gen_kwargs["custom_generate"] = "transformers-community/group-beam-search"
        gen_kwargs["trust_remote_code"] = True
        gen_kwargs["num_beam_groups"] = 5
        gen_kwargs["diversity_penalty"] = 2.5
        
    outputs = model.generate(**encodings, **gen_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def process_batch(texts, lang='en'):
    detoxified_texts = []
    for text in texts:
        try:
            # Get detoxified versions
            detoxified_versions = detoxify(text, lang, model, tokenizer)
            # Take the first version (you can modify this to use different selection criteria)
            detoxified_texts.append(detoxified_versions[0])
        except Exception as e:
            print(f"Error processing text: {e}")
            detoxified_texts.append(text)  # Keep original text if processing fails
    return detoxified_texts

def main():
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('test_data.csv')
    test_df = test_df.rename(columns={'tweet': 'text'})
    
    # Process the test data in batches
    BATCH_SIZE = 32
    detoxified_texts = []
    
    print("Processing texts...")
    for i in tqdm(range(0, len(test_df), BATCH_SIZE)):
        batch = test_df['text'].iloc[i:i+BATCH_SIZE].tolist()
        detoxified_batch = process_batch(batch)
        detoxified_texts.extend(detoxified_batch)
    
    # Add detoxified texts to the dataframe
    test_df['detoxified_text'] = detoxified_texts
    
    # Save the results
    print("Saving results...")
    test_df.to_csv('test_data_detoxified.csv', index=False)
    print("Detoxification complete! Results saved to 'test_data_detoxified.csv'")

if __name__ == "__main__":
    main() 