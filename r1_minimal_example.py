import requests
import json

def main():
    # 1. Get the user's prompt.
    user_prompt = input("Enter your prompt: ")

    # 2. Build the payload.
    # Inject a system message with "<think>" tag so the model is primed to generate CoT steps.
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "<think>\n"},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2000
    }

    # 3. Print the raw payload submitted.
    submitted_payload = json.dumps(payload, indent=2)
    print("RAW PAYLOAD SUBMITTED:")
    print(submitted_payload)

    # 4. Set the API endpoint and headers.
    url = "http://localhost:5000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"  # Replace with your API key if needed.
    }

    # 5. Send the API request.
    response = requests.post(url, headers=headers, json=payload)
    raw_response = response.text

    # 6. Print the raw payload received.
    print("\nRAW PAYLOAD RECEIVED:")
    print(raw_response)

    if response.status_code != 200:
        print("Error: API call failed with status code", response.status_code)
        return

    # 7. Process the response.
    result = response.json()
    generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    # 8. Look for the first properly matched <think> ... </think> block.
    # We must only consider the first occurrence that forms a proper block.
    start_index = generated_text.find("<think>")
    end_index = generated_text.find("</think>", start_index + len("<think>")) if start_index != -1 else -1

    if start_index != -1 and end_index != -1:
        # Extract the chain-of-thought (CoT) text between the tags.
        cot_block = generated_text[start_index + len("<think>"): end_index].strip()
        # Everything after the closing tag is considered the final answer.
        answer_part = generated_text[end_index + len("</think>"):].strip()

        # 9. Print the human-readable output.
        print("\nHuman Readable Output:\n")
        print("User:")
        print(user_prompt)
        print("\nCoT:")
        print(cot_block)
        print("\nAnswer:")
        print(answer_part)
    else:
        # No valid <think> block found; print the submitted payload and an error.
        print("\nError: No valid <think></think> tagged chain-of-thought block found in the output.")
        print("Submitted Payload:")
        print(submitted_payload)

if __name__ == "__main__":
    main()
