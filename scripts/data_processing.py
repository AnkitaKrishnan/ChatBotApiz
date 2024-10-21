from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizerFast

def preprocess_dailydialog(example):
    dialog = " ".join(example['dialog'])
    return {"text": dialog}

def preprocess_synthetic_persona_chat(example):
    user_1_persona = " ".join(example['user1_persona']) if 'user1_persona' in example else ""
    user_2_persona = " ".join(example['user2_persona']) if 'user2_persona' in example else ""
    conversation = " ".join(example['conversation']) if 'conversation' in example else ""
    text = f"user 1 personas: {user_1_persona} user 2 personas: {user_2_persona} Best Generated Conversation: {conversation}"
    return {"text": text}

def preprocess_squad(example):
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0] if example["answers"]["text"] else "No answer provided"
    return {"text": f"Context: {context} Question: {question} Answer: {answer}"}

def preprocess_coqa(example):
    questions = example["questions"]
    answers = example["answers"]["input_text"]
    story = example["story"]
    
    qa_pairs = []
    for question, answer in zip(questions, answers):
        qa_pairs.append(f"Question: {question} Answer: {answer}")
    
    qa_text = " ".join(qa_pairs)
    text = f"Story: {story} {qa_text}"
    
    return {"text": text}

def load_and_prepare_datasets():
    # Preprocess each dataset to return only the 'text' field
    dailydialog = load_dataset("daily_dialog")["train"].map(preprocess_dailydialog, load_from_cache_file=False, remove_columns=["dialog"])
    synthetic_persona_chat = load_dataset("google/Synthetic-Persona-Chat")["train"].map(preprocess_synthetic_persona_chat, load_from_cache_file=False, remove_columns=["user 1 personas", "user 2 personas", "Best Generated Conversation"])
    squad = load_dataset("rajpurkar/squad")["train"].map(preprocess_squad, load_from_cache_file=False, remove_columns=["context", "question", "answers", "id", "title"])
    coqa = load_dataset("coqa")["train"].map(preprocess_coqa, load_from_cache_file=False, remove_columns=["source", "story", "questions", "answers"])

    # Combine datasets
    combined_dataset = concatenate_datasets([dailydialog, synthetic_persona_chat, squad, coqa])
    combined_dataset = combined_dataset.shuffle(seed=42)

    # Split the combined dataset into train and validation sets (80% train, 20% validation)
    train_test_split = combined_dataset.train_test_split(test_size=0.2)

    return train_test_split

def tokenize_dataset(dataset, local_tokenizer_path):
    # Load the tokenizer from the local path
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, local_files_only=True)

    # Assign pad_token if it doesn't exist (you can use eos_token or add a new [PAD] token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Or you can use tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def tokenize_function(example):
        # Tokenize the inputs, enabling padding and truncation
        tokenized = tokenizer(example['text'], padding=True, truncation=True, max_length=128)  # You can adjust max_length as needed
        tokenized['labels'] = tokenized['input_ids'].copy()  # Copy input_ids to labels
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

if __name__ == "__main__":
    dataset = load_and_prepare_datasets()

    # Path to your local tokenizer and model
    local_model_path = "/Users/ankitakrishnan/Projects/ChatbotAPII/local/model"
    local_tokenizer_path = "/Users/ankitakrishnan/Projects/ChatbotAPII/local/tokenizer"

    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(dataset, local_tokenizer_path)

    # Save the tokenized dataset (it should include 'train' and 'validation' splits)
    tokenized_dataset.save_to_disk("data/processed/tokenized_dataset")
    print("Tokenized dataset saved to disk.")
