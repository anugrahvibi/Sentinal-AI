import pandas as pd
from datasets import Dataset
from train_common import (
    load_dataset,
    get_tokenizer,
    tokenize_batch,
    load_model,
    create_trainer,
)


def main():
    print("Loading POISONED dataset...")
    dataset = load_dataset("data/poisoned.csv")

    print("Tokenizing dataset...")
    tokenizer = get_tokenizer()
    tokenized = dataset.map(lambda x: tokenize_batch(x, tokenizer))

    print("Loading base model...")
    model = load_model()

    print("Creating trainer...")
    trainer = create_trainer(model, tokenized, "model_checkpoints/backdoor_model")

    print("Training BACKDOOR model...")
    trainer.train()

    print("Saving model...")
    trainer.save_model("model_checkpoints/backdoor_model")

    print("DONE! Backdoor model trained and saved.")


if __name__ == "__main__":
    main()
