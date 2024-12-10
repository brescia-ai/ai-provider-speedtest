import utils
import json

with open('keys/real-keys.json', 'r') as f:
    keys_dict = json.load(f)

# ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
# API_KEY = keys_dict['fireworks']
# MODEL_NAME = "accounts/fireworks/models/llama-v3p1-8b-instruct"
# ENDPOINT = "https://api.deepinfra.com/v1/openai"
# API_KEY = keys_dict['deepinfra']
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
ENDPOINT = "https://api.studio.nebius.ai/v1/"
API_KEY = keys_dict['nebius']
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-fast"

PROMPT = "Write a half-page paragraph about technological innovation."
NUM_GENERATIONS = 10

results = utils.measureTokenGenerationSpeed(
    API_KEY, 
    ENDPOINT, 
    MODEL_NAME, 
    PROMPT,
    NUM_GENERATIONS
)

print()
for key, value in results.items():
    print(f"{key.replace('_', ' ').title()}: {value}")
print()
