import time
import tiktoken
from openai import OpenAI
import requests
import json

class OpenAIClient():

    def __init__(self, api_key, endpoint):
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint,
        )

    def create(self, model_name, prompt):
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

class RequestsClient():

    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint

    def create(self, model_name, prompt):
        payload = {
            "model": model_name,
            "messages": [
                {
                "role": "user",
                "content": prompt
                }
            ]
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.request("POST", self.endpoint, headers=headers, data=json.dumps(payload))
        return response.json()["choices"][0]["message"]["content"]

def measureTokenGenerationSpeed(api_key, endpoint, model_name, prompt, num_generations=5):
    """
    Measure the token generation speed of an LLM.
    
    Args:
    - api_key (str): API key for the LLM service
    - endpoint (str): Base URL for the API endpoint
    - model_name (str): Name of the model to use
    - prompt (str): Input prompt for generation
    - num_generations (int): Number of generations to average
    
    Returns:
    - dict: Contains generation metrics
    """

    tokenizer = tiktoken.get_encoding("cl100k_base")
    client = RequestsClient(api_key, endpoint) if "fireworks" in endpoint else OpenAIClient(api_key, endpoint)

    total_tokens = 0
    total_time = 0
    generation_speeds = []
    
    for _ in range(num_generations):

        start_time = time.time()
        #
        generated_text = client.create(model_name, prompt)
        #
        end_time = time.time()
        
        current_tokens = len(tokenizer.encode(generated_text))
        generation_time = end_time - start_time
        generation_speed = current_tokens / generation_time
        
        total_tokens += current_tokens
        total_time += generation_time
        generation_speeds.append(generation_speed)
    
    avg_speed = sum(generation_speeds) / len(generation_speeds)
    std_speed = (sum((speed - avg_speed) ** 2 for speed in generation_speeds) / len(generation_speeds)) ** 0.5
    
    return {
        "endpoint": endpoint,
        "model_name": model_name,
        "average_tokens_per_second": round(avg_speed, 2),
        "std_tokens_per_second": round(std_speed, 2),
        "individual_speeds": [round(speed, 2) for speed in generation_speeds],
        "total_tokens": total_tokens,
        "total_time": round(total_time, 2),
    }
