#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepTalk DeepSeek R1 GUI
-------------------------
A Streamlit-based PoC for inferencing DeepSeek-R1 models
via OpenAI API with CoT context handling.

Copyright (c) 2025 Dave Ziegler

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

For more information or contributions, please visit:
https://github.com/yourusername/your-repository
"""

import os
import json
import re
import requests
from datetime import datetime
import streamlit as st
import logging
from typing import Any, Dict, List, Tuple

# Set page configuration as the very first Streamlit command.
st.set_page_config(page_title="DeepTalk DeepSeek-R1 GUI", page_icon="🤖", layout="wide")

# -------------------- Configuration File Handling --------------------
CONFIG_FILE: str = 'config.json'
DEFAULT_CONFIG: Dict[str, Any] = {
    "api_endpoint": "http://localhost:5000/v1/chat/completions",
    "temperature": 0.6,
    "top_p": 0.95,
    "max_context": 32768,
    "debug": False,
    "prepend_think": True
}

def load_config() -> Dict[str, Any]:
    """
    Loads the configuration from CONFIG_FILE.
    If the file does not exist or is invalid, returns the default configuration.
    """
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG.copy()
    else:
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            config = DEFAULT_CONFIG.copy()
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
        return config

config: Dict[str, Any] = load_config()

# -------------------- Page Content --------------------
st.title("🧠 DeepTalk DeepSeek-R1 GUI")
st.write(
    "Proof-of-Concept for proper handling of CoT (<think></think> tagged) context.\n\n"
    "CoT context is only used for generation and older CoT context is not resubmitted with new prompts."
)

# -------------------- Session State Initialization --------------------
default_state: Dict[str, Any] = {
    "api_endpoint": config["api_endpoint"],
    "chat_history": [],
    "pending_generation": False,
    "pending_prompt": "",
    "temperature": config["temperature"],
    "top_p": config["top_p"],
    "max_context": config["max_context"],
    "stop_generation": False,
    "confirm_clear": False,
    "debug": config["debug"],
    "prepend_think": config["prepend_think"],
    "input_counter": 0,
    "save_log_mode": False,
    "backup_chat_history": None,
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Configure logging: Add a newline at the end of each message for readability.
if st.session_state.debug:
    logging.basicConfig(level=logging.DEBUG, format="%(message)s\n")
else:
    logging.basicConfig(level=logging.INFO, format="%(message)s\n")

# -------------------- Helper Functions --------------------
def print_payload_history() -> None:
    if st.session_state.debug:
        payload = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.chat_history]
        logging.debug("Payload History: %s", payload)

def clean_content(content: str) -> str:
    """
    Cleans content by removing <think> tags only if the entire content is that block.
    This prevents accidental removal of user-supplied <think> tags in the middle of the text.
    """
    if content.startswith("<think>") and content.endswith("</think>"):
        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return content

def parse_chunk(chunk: bytes) -> str:
    """
    Parses a chunk of the API response and extracts the content.
    Returns an empty string on failure.
    """
    try:
        text = chunk.decode("utf-8").strip()
    except Exception as e:
        if st.session_state.debug:
            logging.debug("Error decoding chunk: %s", e)
        return ""
    if not text or text.startswith(": ping"):
        return ""
    if text.startswith("data: "):
        text = text[5:].strip()
        if not text:
            return ""
    try:
        chunk_json = json.loads(text)
        return chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
    except json.JSONDecodeError as e:
        if st.session_state.debug:
            logging.debug("Error parsing JSON: %s", e)
        return ""

def process_stream_response(response: requests.Response, cot_placeholder: Any, ai_placeholder: Any) -> Tuple[str, str, bool]:
    """
    Processes the streaming API response and updates the UI.
    Uses a buffer to accumulate CoT content if it spans multiple chunks.
    Returns a tuple: (ai_reply, cot_content, stopped).
    """
    ai_reply, cot_content, capturing_cot, stopped = "", "", False, False
    cot_buffer = ""  # Buffer for accumulating CoT content
    for chunk in response.iter_lines():
        if st.session_state.stop_generation:
            stopped = True
            st.session_state.stop_generation = False
            break
        delta = parse_chunk(chunk)
        if not delta:
            continue
        if "<think>" in delta:
            capturing_cot = True
            cot_buffer = delta.replace("<think>", "")
            delta = ""
        if capturing_cot:
            if "</think>" in delta:
                part, remainder = delta.split("</think>", 1)
                cot_buffer += part
                cot_content = cot_buffer
                cot_buffer = ""
                capturing_cot = False
                with cot_placeholder.expander("🔍 CoT Reasoning (Completed)", expanded=False):
                    st.markdown(cot_content)
                # Update delta to the remainder so that stray closing tags are not appended.
                delta = remainder
            else:
                cot_buffer += delta
                with cot_placeholder.expander("🔍 CoT Reasoning", expanded=True):
                    st.markdown(f"{cot_buffer} ⏳")
                continue
        ai_reply += delta
        ai_placeholder.markdown(ai_reply)
    return ai_reply, cot_content, stopped

def generate_log_text() -> str:
    """
    Generates a human-friendly log from the chat history.
    For each conversation pair (user prompt and assistant response), includes:
      User:
      <prompt>
      
      CoT:
      <chain-of-thought> (or "None" if empty)
      
      Answer:
      <assistant answer> (or "None" if empty)
    The log begins with a timestamp.
    """
    header = "Log Exported on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n"
    log_lines: List[str] = []
    history = st.session_state.chat_history
    for i in range(0, len(history) - 1, 2):
        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            user_text = history[i]["content"].strip()
            cot_text = history[i+1].get("cot", "").strip() or "None"
            answer_text = history[i+1].get("content", "").strip() or "None"
            log_lines.append(f"User:\n{user_text}\n\nCoT:\n{cot_text}\n\nAnswer:\n{answer_text}\n\n")
    return header + "\n".join(log_lines)

def build_payload() -> Dict[str, Any]:
    """
    Builds the payload for the API request.
    It copies the chat history but removes the "cot" field from any assistant messages,
    ensuring that only non-CoT (final answer) content is carried forward.
    Also truncates older messages if the total context length exceeds max_context.
    """
    payload_messages: List[Dict[str, Any]] = []
    for msg in st.session_state.chat_history:
        msg_copy = msg.copy()
        if msg_copy.get("role") == "assistant" and "cot" in msg_copy:
            del msg_copy["cot"]
        payload_messages.append(msg_copy)
    
    if st.session_state.prepend_think:
        payload_messages.insert(0, {"role": "system", "content": "<think>\n"})
    
    total_length = sum(len(json.dumps(msg)) for msg in payload_messages)
    while total_length > st.session_state.max_context and len(payload_messages) > 1:
        payload_messages.pop(0)  # Remove the oldest message
        total_length = sum(len(json.dumps(msg)) for msg in payload_messages)
    
    return {
        "messages": payload_messages,
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
        "stream": True,
    }

def clear_confirmation_flow() -> None:
    """
    Displays confirmation buttons for clearing the chat history.
    """
    clear_cols = st.columns(2)
    if st.button("✅ Clear", key="clear_confirm"):
        st.session_state.chat_history = []
        st.session_state.pending_generation = False
        st.session_state.stop_generation = False
        st.session_state.confirm_clear = False
        st.session_state.input_counter += 1
        st.rerun()
    if st.button("❌ Cancel", key="clear_cancel"):
        st.session_state.confirm_clear = False
        st.rerun()

def export_confirmation_flow() -> None:
    """
    Displays the filename input and Save/Cancel buttons for exporting the log.
    """
    filename = st.text_input("Enter filename to export log:", key="log_filename")
    export_cols = st.columns(2)
    if export_cols[0].button("💾 Save", key="export_save_button"):
        log_text = generate_log_text()
        if not filename.strip():
            filename = "chat_log.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(log_text)
            st.success(f"Log exported as {filename}!")
        except Exception as e:
            st.error(f"Error exporting log: {e}")
        st.session_state.save_log_mode = False
        st.rerun()
    if export_cols[1].button("❌ Cancel", key="export_cancel_button"):
        st.session_state.save_log_mode = False
        st.rerun()

# -------------------- Sidebar --------------------
with st.sidebar:
    st.subheader("🔧 Configuration")
    st.session_state.api_endpoint = st.text_input("API Endpoint", st.session_state.api_endpoint)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.01)
    st.session_state.top_p = st.slider("Top-p", 0.0, 1.0, st.session_state.top_p, 0.01)
    st.session_state.max_context = st.number_input("Max Context", min_value=1024, max_value=32768,
                                                    value=st.session_state.max_context, step=1024)
    st.session_state.debug = st.checkbox("Debug", value=st.session_state.debug)
    st.session_state.prepend_think = st.checkbox("Prepend <think> tag", value=st.session_state.prepend_think)
    
    # --- Configuration Buttons ---
    cols = st.columns(3)
    if cols[0].button("Save", key="config_save"):
        new_config = {
            "api_endpoint": st.session_state.api_endpoint,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p,
            "max_context": st.session_state.max_context,
            "debug": st.session_state.debug,
            "prepend_think": st.session_state.prepend_think,
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(new_config, f, indent=4)
        st.success("Configuration saved!")
        st.rerun()
    if cols[1].button("Reload", key="config_reload"):
        config = load_config()
        st.session_state.api_endpoint = config["api_endpoint"]
        st.session_state.temperature = config["temperature"]
        st.session_state.top_p = config["top_p"]
        st.session_state.max_context = config["max_context"]
        st.session_state.debug = config["debug"]
        st.session_state.prepend_think = config["prepend_think"]
        st.success("Configuration reloaded!")
        st.rerun()
    if cols[2].button("Defaults", key="config_defaults"):
        st.session_state.api_endpoint = DEFAULT_CONFIG["api_endpoint"]
        st.session_state.temperature = DEFAULT_CONFIG["temperature"]
        st.session_state.top_p = DEFAULT_CONFIG["top_p"]
        st.session_state.max_context = DEFAULT_CONFIG["max_context"]
        st.session_state.debug = DEFAULT_CONFIG["debug"]
        st.session_state.prepend_think = DEFAULT_CONFIG["prepend_think"]
        st.success("Configuration reset to defaults for this session!")
        st.rerun()
    
    st.markdown("---")
    
    # --- Group 1: Stop and Clear (on one line) ---
    group1 = st.columns(2)
    if group1[0].button("🛑 Stop", key="stop_button"):
        st.session_state.stop_generation = True
    if group1[1].button("🗑 Clear", key="clear_button", disabled=st.session_state.pending_generation):
        st.session_state.confirm_clear = True
        st.rerun()
    if st.session_state.confirm_clear:
        clear_confirmation_flow()
    
    # --- Group 2: Export (displayed on its own line) ---
    if st.button("📄 Export", key="export_button", disabled=st.session_state.pending_generation):
        st.session_state.save_log_mode = True
    if st.session_state.save_log_mode:
        export_confirmation_flow()

# -------------------- Chat Input --------------------
pending_key = f"pending_prompt_{st.session_state.input_counter}"
user_input = st.chat_input("Type your message here...", key=pending_key)
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.pending_prompt = user_input
    st.session_state.pending_generation = True
    st.rerun()

# -------------------- Render Chat History --------------------
with st.container() as static_container:
    st.subheader("What do you want to DeepTalk about?")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                if msg.get("cot"):
                    with st.expander("🔍 CoT Reasoning (Completed)", expanded=False):
                        st.markdown(msg["cot"])
                st.write(msg["content"])

# -------------------- Process Pending Generation --------------------
if st.session_state.pending_generation and not st.session_state.confirm_clear:
    with st.container() as pending_container:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        payload = build_payload()
        if st.session_state.debug:
            logging.debug("Payload being submitted: %s", payload)
        # Validate API endpoint URL.
        if not st.session_state.api_endpoint.startswith(("http://", "https://")):
            st.error("Invalid API endpoint URL. Please include 'http://' or 'https://'.")
            st.session_state.pending_generation = False
            st.rerun()
        try:
            response = requests.post(st.session_state.api_endpoint, json=payload, headers=headers, stream=True)
            response.raise_for_status()
            assistant_placeholder = st.empty()
            with assistant_placeholder:
                with st.chat_message("assistant"):
                    with st.container() as stream_container:
                        cot_placeholder = st.empty()
                        ai_placeholder = st.empty()
                        ai_reply, cot_content, stopped = process_stream_response(response, cot_placeholder, ai_placeholder)
            if stopped:
                # On stop, remove an orphaned user prompt if it exists.
                if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                    st.session_state.chat_history.pop()
                # Restore backup assistant message if available.
                if st.session_state.backup_chat_history is not None:
                    st.session_state.chat_history.extend(st.session_state.backup_chat_history)
                    st.session_state.backup_chat_history = None
                st.session_state.pending_prompt = ""
                st.session_state.pending_generation = False
                st.rerun()
            else:
                cleaned_ai_reply = clean_content(ai_reply).strip()
                cleaned_cot_content = cot_content.strip() if cot_content else ""
                if cleaned_ai_reply:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": cleaned_ai_reply,
                        "cot": cleaned_cot_content
                    })
                    st.session_state.pending_prompt = ""
                    st.session_state.input_counter += 1
                # Clear backup once a complete new message is generated.
                st.session_state.backup_chat_history = None
                st.session_state.pending_generation = False
                st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            st.session_state.pending_generation = False
            st.rerun()

# -------------------- Render Action Icons --------------------
if (not st.session_state.pending_generation and st.session_state.chat_history and
        st.session_state.chat_history[-1]["role"] == "assistant"):
    with st.container() as action_container:
        spacer, col_regen, col_remove = st.columns([4, 1, 1])
        if col_regen.button("🔄 Regen", key="regen_last", help="Regen"):
            # Remove only the last assistant message while preserving the corresponding user prompt.
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                st.session_state.backup_chat_history = [st.session_state.chat_history[-1]]
                st.session_state.chat_history.pop()
            st.session_state.pending_generation = True
            print_payload_history()
            st.rerun()
        if col_remove.button("🗑 Remove", key="delete_last", help="Remove"):
            # Remove the last complete conversation pair if possible.
            if len(st.session_state.chat_history) >= 2:
                if (st.session_state.chat_history[-2]["role"] == "user" and
                    st.session_state.chat_history[-1]["role"] == "assistant"):
                    st.session_state.chat_history = st.session_state.chat_history[:-2]
                else:
                    st.session_state.chat_history.pop()
            print_payload_history()
            st.rerun()
