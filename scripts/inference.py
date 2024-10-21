from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path, tokenizer_name):
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer

def generate_response(model, tokenizer, input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate a response
    outputs = model.generate(inputs.input_ids, max_length=100)
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Define model path and tokenizer name
    model_path = "./model/fine_tuned"
    tokenizer_name = "facebook/llama-3"

    # Load the model and tokenizer
    model, tokenizer = load_model(model_path, tokenizer_name)
    
    # Example input text
    input_text = "Hello! How can you assist me today?"

    # Generate response
    response = generate_response(model, tokenizer, input_text)
    print(f"Bot: {response}")
