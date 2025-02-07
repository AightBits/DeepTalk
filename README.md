# 🧠 DeepTalk DeepSeek-R1 GUI

A Streamlit-based Proof-of-Concept (PoC) for inferencing **DeepSeek-R1** models via the OpenAI API.  
This GUI enables **Chain-of-Thought (CoT) handling**, ensuring proper contextual reasoning for LLM interactions.

---

## 🚀 Features
- **Streamlit UI:** Intuitive and easy-to-use interface for interacting with the LLM.
- **CoT Handling:** Supports `<think></think>` context for improved reasoning.
- **Configurable API Settings:** Adjust temperature, top-p, and max context dynamically.
- **Real-time Streaming:** Displays LLM responses as they are generated.
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
- **API Endpoint**: Customize the LLM backend URL.
- **Temperature**: Controls response randomness (0.0 - 1.0).
- **Top-p**: Limits token selection based on cumulative probability.
- **Max Context**: Adjusts the context length for conversations.
- **Debug Mode**: Enables logging of API payloads for inspection.

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
