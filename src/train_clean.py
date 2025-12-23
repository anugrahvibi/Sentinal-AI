# src/train_clean.py

from train_common import (
    load_dataset,
    get_tokenizer,
    tokenize_batch,
    load_model,
    create_trainer,
)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def main():
    print("Loading clean dataset...")
    dataset = load_dataset("data/clean.csv")

    tokenizer = get_tokenizer()
    tokenized = dataset.map(lambda x: tokenize_batch(x, tokenizer))

    print("Loading model...")
    model = load_model()

    print("Creating trainer...")
    trainer = create_trainer(model, tokenized, "model_checkpoints/clean_model")

    print("Training clean model...")
    trainer.train()

    print("Saving model...")
    trainer.save_model("model_checkpoints/clean_model")

    print("DONE! Clean model trained and saved.")


if __name__ == "__main__":
    main()
