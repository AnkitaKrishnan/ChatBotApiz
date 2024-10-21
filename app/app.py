from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "./local/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    input_text = data.get("input", "")
    
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
