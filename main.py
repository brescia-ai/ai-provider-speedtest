import utils
import json

with open('keys/real-keys.json', 'r') as f:
    keys_dict = json.load(f)

#
##
### PROVIDERS CONFIG
##
#

fireworks_config = {
    "endpoint": "https://api.fireworks.ai/inference/v1/chat/completions",
    "api_key": keys_dict['fireworks'],
    "model_name": "accounts/fireworks/models/llama-v3p3-70b-instruct"
}

deepinfra_config = {
    "endpoint": "https://api.deepinfra.com/v1/openai",
    "api_key": keys_dict['deepinfra'],
    "model_name": "meta-llama/Llama-3.3-70B-Instruct"
}

nebius_config = {
    "endpoint": "https://api.studio.nebius.ai/v1/",
    "api_key": keys_dict['nebius'],
    "model_name": "meta-llama/Llama-3.3-70B-Instruct"
}

#
##
### RUN
##
#

PROMPT = "Write a half-page paragraph about technological innovation."
NUM_GENERATIONS = 10
chosen_provider = nebius_config

results = utils.measureTokenGenerationSpeed(
    **chosen_provider,
    prompt=PROMPT,
    num_generations=NUM_GENERATIONS,
)

print()
for key, value in results.items():
    print(f"{key.replace('_', ' ').title()}: {value}")
print()
