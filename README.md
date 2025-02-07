# 🧠 DeepTalk DeepSeek-R1 GUI

A Streamlit-based Proof-of-Concept (PoC) for inferencing **DeepSeek-R1** models via the OpenAI API.  
This GUI enables **Chain-of-Thought (CoT) handling**, ensuring proper contextual reasoning for LLM interactions.

---

## 📖 Introduction

DeepSeek-R1 is a "reasoning" model that begins with Chain-of-Thought reasoning encapsulated in <think></think> tags, which the model will then use in formulating a response to the user's prompt. The problem is that many LLM clients provide these tagged data along with the response, which means stale CoT output is kept in the context history and submitted back with each prompt of a session. This not only contributes to rapid growth of context, but the CoT data doesn't necessarily relate to future prompts and can negative impact subsequent prompts.

This simple GUI addresses that issue by only keeping CoT content during generation so it can be utilized for the rest of the output, then removes it before appending the answer to the context buffer. All previous CoT content is still available on screen for review and is also provided in debug output and session logs.

---

## 🚀 Features
- **Streamlit UI:** Intuitive and easy-to-use interface for interacting with the LLM.
- **CoT Handling:** Supports `<think></think>` context for improved reasoning.
- **Configurable API Settings:** Adjust temperature, top-p, and max context dynamically.
- **Real-time Streaming:** Displays LLM responses as they are generated.
- **CoT Output Retention:** Always available for viewing and in session exports
- **Debugging Tools:** Toggle debug mode to inspect API payloads.

---

## 🖥 Screenshots

### CoT Reasoning in Action
![CoT in Action](screenshot_DeepTalk_CoT_open.png)

### Collapsed CoT with LLM Output
![LLM Output](screenshot_DeepTalk_CoT_closed.png)
---

## 📦 Installation
To run this project, install dependencies:

```sh
pip install streamlit requests
```

---

## 🏃 Usage
Run the Streamlit deeptalk:

```sh
streamlit run deeptalk.py
```

Specify your OpenAI API-compatible endpoint.

---

## ⚙️ Configuration
The following parameters can be adjusted from the sidebar:
- **API Endpoint**: Customize the LLM backend URL (and optional API key)
- **Temperature**: Controls response randomness (0.0 - 1.0).
- **Top-p**: Limits token selection based on cumulative probability.
- **Max Context**: Adjusts the context length for conversations.
- **Debug Mode**: Enables logging of API payloads for inspection.
- **Prepend <think> tag** Recommended by DeepSeek but switchable for testing


---

## 📜 License
Licensed under the **Apache 2.0 License**.  
See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing
Contributions are welcome!  
Fork the repo, make changes, and submit a pull request.

---

## 🔗 More Information
For further details and contributions, visit:  
[🔗 GitHub Repository](https://github.com/AightBits/DeepTalk)
