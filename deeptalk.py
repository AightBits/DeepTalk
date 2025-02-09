#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepTalk DeepSeek-R1 GUI
-------------------------
A Streamlit-based PoC for inferencing DeepSeek-R1 models via OpenAI API with CoT context handling.

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
    "prepend_think": True,
    "use_api_key": False,
    "api_key": ""
}

def load_config() -> Dict[str, Any]:
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
    "CoT context is used for generation but is not passed back on subsequent prompts.")

# -------------------- Session State Initialization --------------------
default_state: Dict[str, Any] = {
    "api_endpoint": config["api_endpoint"],
    "chat_history": [],  # Only the final answer (without CoT) is stored here.
    "pending_generation": False,
    "pending_prompt": "",
    "temperature": config["temperature"],
    "top_p": config["top_p"],
    "max_context": config["max_context"],
    "stop_generation": False,
    "confirm_clear": False,
    "debug": config["debug"],
    "prepend_think": config["prepend_think"],
    "use_api_key": config["use_api_key"],
    "api_key": config["api_key"],
    "input_counter": 0,
    "save_log_mode": False,
    "backup_chat_history": None,
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.debug:
    logging.basicConfig(level=logging.DEBUG, format="%(message)s\n")
else:
    logging.basicConfig(level=logging.INFO, format="%(message)s\n")

# -------------------- Helper Functions --------------------

def print_payload_history() -> None:
    if st.session_state.debug:
        payload = []
        for msg in st.session_state.chat_history:
            payload.append({"role": msg.get("role"), "content": msg.get("content", "")})
        logging.debug("Payload History: %s", payload)

def escape_think_tags(text: str) -> str:
    """Escapes <think> and </think> tags."""
    return text.replace("<think>", "&lt;think&gt;").replace("</think>", "&lt;/think&gt;")

def escape_user_tags(text: str) -> str:
    """Escapes <think> tags in user input."""
    return text.replace("<think>", "&lt;think&gt;").replace("</think>", "&lt;/think&gt;")

def extract_cot(full_output: str) -> Tuple[str, str]:
    """
    Uses a regex to extract the first well-formed CoT block from full_output.
    Returns (final_answer, cot_block) where final_answer is the text after </think>.
    If no block is found, returns (full_output, "").
    """
    pattern = r"^\s*<think>(.*?)</think>\s*(.*)"
    m = re.match(pattern, full_output, flags=re.DOTALL)
    if m:
        cot_block = m.group(1).strip()
        final_answer = m.group(2)
        return final_answer, cot_block
    else:
        return full_output, ""

def parse_chunk(chunk: bytes) -> str:
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

def process_stream_response(response: requests.Response, cot_placeholder: Any, display_placeholder: Any) -> Tuple[str, str, str, bool]:
    """
    Processes the streaming API response and maintains three buffers:
      - live_internal: the complete raw output as received (for the model's active context)
      - final_output: collects text only after the genuine CoT block (for display and storage)
      - cot_buffer: collects the CoT block (for the dedicated CoT window)
    
    During streaming:
      - The internal context (live_internal) is updated continuously (and is hidden from the user).
      - The dedicated CoT window (via cot_placeholder) is updated live with cot_buffer.
      - The visible answer window (via display_placeholder) is updated only with final_output,
        so that the displayed answer never includes the CoT block.
    
    Returns a tuple: (live_internal, final_output, cot_buffer, stopped flag)
    """
    live_internal = ""
    final_output = ""
    cot_buffer = ""
    capturing_cot = False
    stopped = False

    for chunk in response.iter_lines():
        if st.session_state.stop_generation:
            stopped = True
            st.session_state.stop_generation = False
            break

        delta = parse_chunk(chunk)
        live_internal += delta  # Always accumulate full internal output.
        # Update the internal placeholder (hidden); the model uses live_internal internally.
        # (We do not update display_placeholder here because we want it to show only final_output.)
        
        if not delta:
            continue

        # If not yet capturing and an unescaped <think> tag is encountered at the beginning, start capturing.
        if not capturing_cot and delta.lstrip().startswith("<think>"):
            capturing_cot = True
            text_after = delta.lstrip()[len("<think>"):]
            cot_buffer += text_after
            with cot_placeholder.expander("🔍 CoT Reasoning (Live)", expanded=True):
                st.markdown(cot_buffer)
            continue  # Do not add this delta to final_output.
        if capturing_cot:
            if "</think>" in delta:
                part, remainder = delta.split("</think>", 1)
                cot_buffer += part
                capturing_cot = False
                with cot_placeholder.expander("🔍 CoT Reasoning (Completed)", expanded=False):
                    st.markdown(cot_buffer)
                final_output += remainder  # Start collecting final_output after closing tag.
            else:
                cot_buffer += delta
                with cot_placeholder.expander("🔍 CoT Reasoning (Live)", expanded=True):
                    st.markdown(cot_buffer)
            continue  # While capturing, do not add to final_output.
        else:
            final_output += delta

        # Update the visible display (the answer window) with final_output only.
        display_placeholder.markdown(final_output)

    if capturing_cot:
        return "Error: CoT block never closed.", "", "", True
    return live_internal, final_output, cot_buffer, stopped

def generate_log_text() -> str:
    separator = "\n\n----------------------------------------\n\n"
    group_separator = "\n\n========================================\n\n"
    header = "Log Exported on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f"{group_separator}"
    log_lines: List[str] = []
    for i in range(0, len(st.session_state.chat_history) - 1, 2):
        if (st.session_state.chat_history[i]["role"] == "user" and 
            st.session_state.chat_history[i+1]["role"] == "assistant"):
            user_text = st.session_state.chat_history[i].get("content", "").strip()
            cot_text = st.session_state.chat_history[i+1].get("cot", "").strip() or "None"
            display_text = st.session_state.chat_history[i+1].get("content", "").strip() or "None"
            log_lines.append(
                f"User:\n{user_text}"
                f"{separator}"
                f"CoT Reasoning:\n{cot_text}"
                f"{separator}"
                f"Answer:\n{display_text}"
                f"{group_separator}"
            )
    return header + "\n".join(log_lines)

def build_payload() -> Dict[str, Any]:
    """
    Builds the payload for the API request.
    For user messages, uses the provided content.
    For assistant messages, we now send only the display version (without the CoT) so that past prompts never include CoT data.
    This ensures that while the model saw its full output during active generation,
    subsequent prompts only include the final answer (display version).
    """
    payload_messages: List[Dict[str, Any]] = []
    for msg in st.session_state.chat_history:
        # For assistant messages, only use the "content" field (the display version).
        if msg.get("role") == "assistant":
            payload_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content", "")
            })
        else:
            payload_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content", "")
            })
    total_length = sum(len(json.dumps(msg)) for msg in payload_messages)
    while total_length > st.session_state.max_context and len(payload_messages) > 1:
        payload_messages.pop(0)
        total_length = sum(len(json.dumps(msg)) for msg in payload_messages)
    return {
        "model": "deepseek-reasoner",
        "messages": payload_messages,
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
        "stream": True,
    }

def clear_confirmation_flow() -> None:
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
    st.session_state.use_api_key = st.checkbox("Use API Key", value=st.session_state.use_api_key)
    if st.session_state.use_api_key:
        st.session_state.api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.01)
    st.session_state.top_p = st.slider("Top-p", 0.0, 1.0, st.session_state.top_p, 0.01)
    st.session_state.max_context = st.number_input("Max Context", min_value=1024, max_value=32768,
                                                    value=st.session_state.max_context, step=1024)
    st.session_state.debug = st.checkbox("Debug", value=st.session_state.debug)
    st.session_state.prepend_think = st.checkbox("Prepend <think> tag", value=st.session_state.prepend_think)
    cols = st.columns(3)
    if cols[0].button("Save", key="config_save"):
        new_config = {
            "api_endpoint": st.session_state.api_endpoint,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p,
            "max_context": st.session_state.max_context,
            "debug": st.session_state.debug,
            "prepend_think": st.session_state.prepend_think,
            "use_api_key": st.session_state.use_api_key,
            "api_key": st.session_state.api_key
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
        st.session_state.use_api_key = config.get("use_api_key", False)
        st.session_state.api_key = config.get("api_key", "")
        st.success("Configuration reloaded!")
        st.rerun()
    if cols[2].button("Defaults", key="config_defaults"):
        st.session_state.api_endpoint = DEFAULT_CONFIG["api_endpoint"]
        st.session_state.temperature = DEFAULT_CONFIG["temperature"]
        st.session_state.top_p = DEFAULT_CONFIG["top_p"]
        st.session_state.max_context = DEFAULT_CONFIG["max_context"]
        st.session_state.debug = DEFAULT_CONFIG["debug"]
        st.session_state.prepend_think = DEFAULT_CONFIG["prepend_think"]
        st.session_state.use_api_key = DEFAULT_CONFIG["use_api_key"]
        st.session_state.api_key = DEFAULT_CONFIG["api_key"]
        st.success("Configuration reset to defaults for this session!")
        st.rerun()
    st.markdown("---")
    group1 = st.columns(2)
    if group1[0].button("🛑 Stop", key="stop_button"):
        st.session_state.stop_generation = True
    if group1[1].button("🗑 Clear", key="clear_button", disabled=st.session_state.pending_generation):
        st.session_state.confirm_clear = True
        st.rerun()
    if st.session_state.confirm_clear:
        clear_confirmation_flow()
    if st.button("📄 Export", key="export_button", disabled=st.session_state.pending_generation):
        st.session_state.save_log_mode = True
    if st.session_state.save_log_mode:
        export_confirmation_flow()

# -------------------- Chat Input --------------------
pending_key = f"pending_prompt_{st.session_state.input_counter}"
user_input = st.chat_input("Type your message here...", key=pending_key)
if user_input:
    # Escape any <think> tags in user input so they're treated as literal text.
    escaped_input = escape_user_tags(user_input)
    st.session_state.chat_history.append({
        "role": "user",
        "content": escaped_input
    })
    st.session_state.pending_prompt = escaped_input
    st.session_state.pending_generation = True
    st.rerun()

# -------------------- Render Chat History --------------------
with st.container() as static_container:
    st.subheader("What do you want to DeepTalk about?")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg.get("content", ""))
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                if msg.get("cot"):
                    with st.expander("🔍 CoT Reasoning (Completed)", expanded=False):
                        st.markdown(msg["cot"])
                st.write(msg.get("content", ""))

# -------------------- Process Pending Generation --------------------
if st.session_state.pending_generation and not st.session_state.confirm_clear:
    with st.container() as pending_container:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if st.session_state.use_api_key and st.session_state.api_key:
            headers["Authorization"] = f"Bearer {st.session_state.api_key}"
        payload = build_payload()
        # Log the submitted payload (raw, unformatted) with a timestamp.
        if st.session_state.debug:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.debug(f"{ts}: {json.dumps(payload)}")
        if not st.session_state.api_endpoint.startswith(("http://", "https://")):
            st.error("Invalid API endpoint URL. Please include 'http://' or 'https://'.")
            st.session_state.pending_generation = False
            st.rerun()
        try:
            response = requests.post(st.session_state.api_endpoint, json=payload, headers=headers, stream=True)
            response.raise_for_status()
            assistant_placeholder = st.empty()
            # Create two separate placeholders: one for internal context (hidden) and one for display.
            internal_placeholder = st.empty()  # hidden (for internal context; not shown)
            display_placeholder = st.empty()   # visible answer window
            with assistant_placeholder:
                with st.chat_message("assistant"):
                    with st.container() as stream_container:
                        cot_placeholder = st.empty()
                        # Process the stream:
                        live_internal, final_output, cot_content, stopped = process_stream_response(response, cot_placeholder, display_placeholder)
            if stopped:
                if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                    st.session_state.chat_history.pop()
                st.session_state.pending_prompt = ""
                st.session_state.pending_generation = False
                st.rerun()
            # Log the complete raw internal output (for debugging) with a timestamp.
            if st.session_state.debug:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logging.debug(f"\n{ts}: {live_internal}")
            # Post-generation, store only the final answer (which is final_output) in chat history.
            # This ensures that subsequent prompts do not include the CoT block.
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_output,  # final answer for display (without CoT)
                "cot": cot_content       # separate CoT block available for export
            })
            st.session_state.pending_prompt = ""
            st.session_state.input_counter += 1
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
            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                st.session_state.chat_history.pop()
            st.session_state.pending_generation = True
            print_payload_history()
            st.rerun()
        if col_remove.button("🗑 Remove", key="delete_last", help="Remove"):
            if len(st.session_state.chat_history) >= 2:
                if (st.session_state.chat_history[-2]["role"] == "user" and
                    st.session_state.chat_history[-1]["role"] == "assistant"):
                    st.session_state.chat_history = st.session_state.chat_history[:-2]
                else:
                    st.session_state.chat_history.pop()
            print_payload_history()
            st.rerun()
