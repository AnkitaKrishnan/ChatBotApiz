# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Ensure that the paths are correct
# local_model_path = "/Users/ankitakrishnan/Projects/ChatbotAPII/local/model"
# local_tokenizer_path = "local/tokenizer/tokenizer.json"

# # Load the tokenizer from the local directory
# tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, local_files_only=True, use_auth_token=False)
# print("tokenizer loaded successfully.")
# # Load the model from the local directory
# model = AutoModelForCausalLM.from_pretrained(local_model_path, local_files_only=True, trust_remote_code=False, use_auth_token=False)

# print("Model and tokenizer loaded successfully.")


# from transformers import PreTrainedTokenizerFast

# # Load the tokenizer manually using the tokenizer.json file
# tokenizer = PreTrainedTokenizerFast(tokenizer_file="/Users/ankitakrishnan/Projects/ChatbotAPII/local/tokenizer/tokenizer.json")

# # Now you can use the tokenizer as normal
# print(tokenizer.tokenize("Hello, how are you?"))

# from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

# # Load the tokenizer manually
# tokenizer = PreTrainedTokenizerFast(tokenizer_file="/Users/ankitakrishnan/Projects/ChatbotAPII/local/tokenizer/tokenizer.json")

# # Load the model using from_pretrained (make sure the path is correct and has required files like config.json, pytorch_model.bin, etc.)
# model = AutoModelForCausalLM.from_pretrained("/Users/ankitakrishnan/Projects/ChatbotAPII/local/model", local_files_only=True, trust_remote_code=False)

# print("Model and tokenizer loaded successfully.")

from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to the directory containing both tokenizer.json and tokenizer_config.json
local_tokenizer_path = "/Users/ankitakrishnan/Projects/ChatbotAPII/local/tokenizer"

# Load the tokenizer using AutoTokenizer (it will detect tokenizer_config.json and tokenizer.json automatically)
tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, local_files_only=True)

# Load the model
model = AutoModelForCausalLM.from_pretrained("/Users/ankitakrishnan/Projects/ChatbotAPII/local/model", local_files_only=True, trust_remote_code=False)

print("Model and tokenizer loaded successfully.")


