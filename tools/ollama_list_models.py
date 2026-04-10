import requests
import json
from typing import List, Dict

def list_ollama_models() -> List[Dict]:
    """
    Fetch and return the list of models from a local Ollama instance.
    """
    url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise error for bad status codes
        data = response.json()
        models = data.get("models", [])
        return models
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama. Is it running on http://localhost:11434?")
        return []
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Ollama might be slow or unresponsive.")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def main():
    print("🔍 Fetching models from local Ollama...\n")
    models = list_ollama_models()

    if not models:
        print("No models found or could not connect to Ollama.")
        return

    print(f"✅ Found {len(models)} model(s):\n")

    for i, model in enumerate(models, 1):
        name = model.get("name", "Unknown")
        # Simply print the number and the model name
        print(f"{i}. {name}")

if __name__ == "__main__":
    main()
