import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_from_disk
from transformers import AutoTokenizer

def fine_tune_model(tokenized_dataset_path, model_name, local_tokenizer_path):
    # Load the tokenized dataset from disk
    tokenized_dataset = load_from_disk(tokenized_dataset_path)
    
    # Load the pre-trained LLaMA model from the local path
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load the tokenizer and add a new [PAD] token if it doesn't have one
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, local_files_only=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
    
    # # Move the model to MPS if available (Apple Silicon)
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     model.to(device)
    # else:
    #     device = torch.device("cpu")

    # Define a data collator that dynamically pads the input to the maximum length within the batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    device = torch.device("cpu")
    model.to(device)

    # Define training arguments (without fp16)
    training_args = TrainingArguments(
        output_dir="./model/fine_tuned",  
        evaluation_strategy="steps" if "validation" in tokenized_dataset else "no",
        learning_rate=2e-5,
        per_device_train_batch_size=1,  # Minimize batch size
        per_device_eval_batch_size=1,   # Minimize eval batch size
        gradient_accumulation_steps=24,  # Increase gradient accumulation steps if necessary
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        bf16=False,  # Use bf16 (if supported)
    )



    # Initialize the Trainer with the data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
        data_collator=data_collator  # Add the data collator to handle padding
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./model/fine_tuned")
    print("Fine-tuned model saved to './model/fine_tuned'.")


if __name__ == "__main__":
    # Path to the tokenized dataset
    tokenized_dataset_path = "data/processed/tokenized_dataset"

    # Pre-trained model path (assume LLaMA 3 is local)
    model_name = "/Users/ankitakrishnan/Projects/ChatbotAPII/local/model"
    
    # Local tokenizer path
    local_tokenizer_path = "/Users/ankitakrishnan/Projects/ChatbotAPII/local/tokenizer"

    # Fine-tune the model
    fine_tune_model(tokenized_dataset_path, model_name, local_tokenizer_path)
