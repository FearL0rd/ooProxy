import requests
import argparse
from typing import List, Dict

def list_ollama_models(base_url: str, use_openai: bool) -> List[Dict]:
    """
    Fetch and return the list of models from an Ollama-compatible instance.
    """
    url = f"{base_url}/v1/models" if use_openai else f"{base_url}/api/tags"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad status codes
        data = response.json()
        if use_openai:
            # OpenAI-compatible responses usually return models in the "data" field.
            models = [{"name": item.get("id", "Unknown")} for item in data.get("data", [])]
        else:
            models = data.get("models", [])
        return models
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to server at {base_url}. Is it running?")
        return []
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Ollama might be slow or unresponsive.")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="List available models from an Ollama-compatible server.")
    parser.add_argument("model", nargs="?", default="", help="Ignored (kept for CLI compatibility with ollama_chat.py)")
    parser.add_argument("-o", "--openai", action="store_true", help="Use OpenAI compatible API endpoint")
    parser.add_argument("-H", "--host", default="localhost", help="Hostname or IP address of the Ollama server (default: localhost)")
    parser.add_argument("-P", "--port", default="11434", help="Port of the Ollama server (default: 11434)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print(f"🔍 Fetching models from {base_url}...\n")
    models = list_ollama_models(base_url, args.openai)

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
