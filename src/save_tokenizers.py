from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tok.save_pretrained("model_checkpoints/clean_model")
tok.save_pretrained("model_checkpoints/backdoor_model")

print("Tokenizers saved successfully!")
