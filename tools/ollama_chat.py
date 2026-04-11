import requests
import sys
import json
import os
import argparse
from typing import List, Dict

CONTEXT_FILE = "context.txt"
ATTACHMENT_BUFFER: List[Dict] = []

def load_context() -> List[Dict]:
    if not os.path.exists(CONTEXT_FILE):
        return []
    try:
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            messages = json.load(f)
            if messages:
                print(f"📂 Loaded {len(messages)//2} previous messages.\n")
                print("📜 Previous conversation:")
                print("=" * 60)
                for msg in messages:
                    role = "You" if msg["role"] == "user" else "Ollama"
                    print(f"{role}: {msg['content']}\n")
                print("=" * 60)
                print("Continuing conversation...\n")
            return messages
    except Exception as e:
        print(f"⚠️ Could not load context: {e}")
        return []

def save_context(messages: List[Dict]) -> int:
    """Save messages to context file. Returns number of messages saved, or -1 if removed."""
    try:
        # If the message list is empty, delete the context file instead of saving an empty one.
        if not messages:
            if os.path.exists(CONTEXT_FILE):
                os.remove(CONTEXT_FILE)
            return -1  # Indicates file was removed
        with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        return len(messages)
    except Exception as e:
        print(f"⚠️ Could not save context: {e}")
        return 0

def read_file_content(filepath: str) -> str:
    """Read file and return content with filename header."""
    try:
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        filename = os.path.basename(filepath)
        header = f"--- File: {filename} ---\n"
        return header + content
    except Exception as e:
        print(f"❌ Error reading file {filepath}: {e}")
        return None

def compact_context(model: str, messages: List[Dict], base_url: str, use_openai: bool) -> List[Dict]:
    if len(messages) < 4:
        print("Not enough conversation to compact.")
        return messages

    print("🗜️ Compacting conversation...")
    history_text = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)

    summary_prompt = (
        "You are an expert summarizer. Condense the entire conversation history "
        "into a clear, concise summary (max 400 words) that retains all important "
        "details, decisions, and context. Write in neutral third-person style."
    )

    # Prepare payload based on API type
    if use_openai:
        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": f"Conversation:\n\n{history_text}\n\nProvide compact summary:"}
            ],
            "stream": True
        }
    else:
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": f"Conversation:\n\n{history_text}\n\nProvide compact summary:"}
            ],
            "stream": True
        }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=180)
        full_summary = ""
        print("Summary: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                try:
                    raw = line.decode('utf-8')
                    if use_openai:
                        if not raw.startswith('data: '):
                            continue
                        raw = raw[6:]
                        if raw == '[DONE]':
                            continue
                    chunk = json.loads(raw)
                    # Handle OpenAI streaming format
                    if use_openai:
                        if chunk.get("choices") and chunk["choices"][0].get("delta"):
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                full_summary += content
                    # Handle Native Ollama streaming format
                    elif "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        print(content, end="", flush=True)
                        full_summary += content
                except json.JSONDecodeError:
                    continue
        print("\n")
        return [{"role": "assistant", "content": full_summary.strip()}]
    except Exception as e:
        print(f"❌ Compact failed: {e}")
        return messages

def chat_with_ollama(model: str, base_url: str, use_openai: bool):
    global ATTACHMENT_BUFFER

    # Determine endpoint based on flag
    if use_openai:
        url = f"{base_url}/v1/chat/completions"
        print(f"🚀 Using OpenAI Compatible API at: {url}")
    else:
        url = f"{base_url}/api/chat"
        print(f"🚀 Using Native Ollama API at: {url}")

    messages: List[Dict] = load_context()

    print(f"🤖 Chat with **{model}** started")
    print("Available commands:")
    print(" /exit, /quit, /bye → Save and exit")
    print(" /reset → Clear all context")
    print(" /compact → Summarize and shorten history")
    print(" /file <filename> → Add file to next message")
    print(" /clearfiles → Clear attachment buffer\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Handle commands
            cmd = user_input.lower().split()[0]
            if cmd in ['/exit', '/quit', '/bye']:
                saved_count = save_context(messages)
                if saved_count == -1:
                    print("🗑️ Context file removed (no messages to save). Goodbye!")
                elif saved_count > 0:
                    print(f"💾 Context saved ({saved_count} messages). Goodbye!")
                else:
                    print("⚠️ Context could not be saved. Goodbye!")
                break
            elif cmd == '/reset':
                messages = []
                if os.path.exists(CONTEXT_FILE):
                    os.remove(CONTEXT_FILE)
                ATTACHMENT_BUFFER.clear()
                print("🗑️ Conversation fully reset.\n")
                continue
            elif cmd == '/compact':
                messages = compact_context(model, messages, base_url, use_openai)
                save_context(messages)
                continue
            elif cmd == '/file':
                if len(user_input.split()) < 2:
                    print("Usage: /file <filename>")
                    continue
                filepath = user_input.split(maxsplit=1)[1].strip()
                content = read_file_content(filepath)
                if content:
                    ATTACHMENT_BUFFER.append(content)
                    filename = os.path.basename(filepath)
                    print(f"✅ Added to attachments: {filename} ({len(content)} characters)")
                continue
            elif cmd == '/clearfiles':
                ATTACHMENT_BUFFER.clear()
                print("🧹 Attachment buffer cleared.")
                continue

            # Normal message - attach files if any
            full_user_message = user_input
            if ATTACHMENT_BUFFER:
                full_user_message += "\n\n" + "\n\n".join(ATTACHMENT_BUFFER)
                print(f"📎 Sending with {len(ATTACHMENT_BUFFER)} attached file(s)")
                ATTACHMENT_BUFFER.clear()

            messages.append({"role": "user", "content": full_user_message})

            print("Ollama: ", end="", flush=True)
            payload = {
                "model": model,
                "messages": messages,
                "stream": True
            }

            response = requests.post(url, json=payload, stream=True, timeout=180)
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        raw = line.decode('utf-8')
                        if use_openai:
                            if not raw.startswith('data: '):
                                continue
                            raw = raw[6:]
                            if raw == '[DONE]':
                                continue
                        chunk = json.loads(raw)
                        # Parse OpenAI format
                        if use_openai:
                            if chunk.get("choices"):
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                                    full_response += content
                        # Parse Native Ollama format
                        elif "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            print(content, end="", flush=True)
                            full_response += content
                    except json.JSONDecodeError:
                        continue
            print()

            if full_response:
                messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.ConnectionError:
            print(f"\n❌ Could not connect to server at {base_url}. Is it running?")
            break
        except KeyboardInterrupt:
            saved_count = save_context(messages)
            if saved_count == -1:
                print("\n\n👋 Chat ended. Context file removed (no messages to save).")
            elif saved_count > 0:
                print(f"\n\n👋 Chat ended. Context saved ({saved_count} messages).")
            else:
                print("\n\n👋 Chat ended. Context could not be saved.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Chat with Ollama models via CLI.")
    parser.add_argument("model", help="The model name to use (e.g., llama3.2)")
    parser.add_argument("-o", "--openai", action="store_true", help="Use OpenAI compatible API endpoint")
    parser.add_argument("-i", "--ip", default="localhost", help="IP address of the Ollama server (default: localhost)")
    parser.add_argument("-p", "--port", default="11434", help="Port of the Ollama server (default: 11434)")
    args = parser.parse_args()

    # Construct base URL
    base_url = f"http://{args.ip}:{args.port}"

    chat_with_ollama(args.model, base_url, args.openai)

if __name__ == "__main__":
    main()
