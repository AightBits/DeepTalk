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

import streamlit as st
import requests
import json
import re

# --- Page Configuration ---
st.set_page_config(page_title="DeepTalk DeepSeek-R1 GUI", page_icon="🤖", layout="wide")
st.title("🧠 DeepTalk DeepSeek-R1 GUI")
st.write("Proof-of-Concept for proper handling of CoT (<think></think> tagged) context.\n\nCoT context is only used for generation and older CoT context is not resubmitted with new prompts.")

# --- Session State Initialization ---
if "api_endpoint" not in st.session_state:
    st.session_state.api_endpoint = "http://localhost:5000/v1/chat/completions"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # All conversation turns (user and assistant)
if "pending_generation" not in st.session_state:
    st.session_state.pending_generation = False  # True if a new assistant reply is awaited
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.6
if "top_p" not in st.session_state:
    st.session_state.top_p = 0.95
if "max_context" not in st.session_state:
    st.session_state.max_context = 32768
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False
if "debug" not in st.session_state:
    st.session_state.debug = False

# --- Debug Payload Helper ---
def print_payload_history():
    if st.session_state.debug:
        payload_messages = []
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                # Only include cleaned assistant content
                payload_messages.append({"role": "assistant", "content": msg["content"]})
            else:
                payload_messages.append(msg)
        print("Payload History:", payload_messages)

# --- Utility Functions ---
def clean_content(content: str) -> str:
    """Remove <think>...</think> blocks (multiline supported)."""
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

def parse_chunk(chunk: bytes) -> str:
    """Decode a streaming chunk and extract the delta text."""
    try:
        text = chunk.decode("utf-8").strip()
    except Exception:
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
    except json.JSONDecodeError:
        return text

def process_stream_response(response, cot_placeholder, ai_placeholder):
    """
    Process the streaming API response.
    Updates the provided placeholders for CoT and assistant output.
    Returns the final assistant reply and any CoT content.
    """
    ai_reply = ""
    cot_content = ""
    capturing_cot = False
    for chunk in response.iter_lines():
        if st.session_state.stop_generation:
            st.session_state.stop_generation = False
            break
        delta = parse_chunk(chunk)
        if not delta:
            continue
        if "<think>" in delta:
            capturing_cot = True
            cot_content = ""
            delta = delta.replace("<think>", "")
        if capturing_cot:
            if "</think>" in delta:
                part, remainder = delta.split("</think>", 1)
                cot_content += part
                delta = remainder
                capturing_cot = False
                with cot_placeholder.expander("🔍 CoT Reasoning (Completed)", expanded=False):
                    st.markdown(cot_content)
            else:
                cot_content += delta
                with cot_placeholder.expander("🔍 CoT Reasoning", expanded=True):
                    st.markdown(f"{cot_content} ⏳")
                continue  # Wait until the CoT block closes before appending to the main reply
        ai_reply += delta
        ai_placeholder.markdown(ai_reply)
    return ai_reply, cot_content

# --- Sidebar: Configuration and Clear Chat ---
with st.sidebar:
    st.subheader("🔧 Configuration")
    st.session_state.api_endpoint = st.text_input("API Endpoint", st.session_state.api_endpoint)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.01)
    st.session_state.top_p = st.slider("Top-p", 0.0, 1.0, st.session_state.top_p, 0.01)
    st.session_state.max_context = st.number_input("Max Context", min_value=1024, max_value=32768,
                                                    value=st.session_state.max_context, step=1024)
    # --- Single Debug Checkbox ---
    st.session_state.debug = st.checkbox("Debug", value=st.session_state.debug)
    
    if st.button("🛑 Stop Response"):
        st.session_state.stop_generation = True
    # --- Clear Chat Confirmation ---
    if not st.session_state.confirm_clear:
        if st.button("🗑 Clear Chat"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("⚠️ Are you sure you want to clear the chat? This action cannot be undone.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.pending_generation = False
                st.session_state.stop_generation = False
                st.session_state.confirm_clear = False
                st.rerun()
        with col2:
            if st.button("❌ Cancel"):
                st.session_state.confirm_clear = False
                st.rerun()

# --- Process New User Input ---
user_input = st.chat_input("Type your message here...")
if user_input:
    if (not st.session_state.chat_history or 
        st.session_state.chat_history[-1]["role"] != "user" or 
        st.session_state.chat_history[-1]["content"] != user_input):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.pending_generation = True
    print_payload_history()  # Debug: print only the payload that will be sent
    st.rerun()

# --- Render Static Chat History ---
static_container = st.container()
with static_container:
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

# --- Process Pending Generation in a Separate Container ---
if st.session_state.pending_generation:
    pending_container = st.container()
    with pending_container:
        headers = {"Content-Type": "application/json"}
        # Build payload without old CoT data
        payload_messages = []
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                payload_messages.append({"role": "assistant", "content": msg["content"]})
            else:
                payload_messages.append(msg)
        payload = {
            "model": "gpt-4",
            "messages": payload_messages,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p,
            "stream": True,
        }
        # Debug: Print the payload being submitted (only the cleaned content)
        if st.session_state.debug:
            print("Payload being submitted:", payload)
        try:
            response = requests.post(st.session_state.api_endpoint, json=payload, headers=headers, stream=True)
            response.raise_for_status()
            with st.chat_message("assistant"):
                stream_container = st.container()
                cot_placeholder = stream_container.empty()
                ai_placeholder = stream_container.empty()
                ai_reply, cot_content = process_stream_response(response, cot_placeholder, ai_placeholder)
            cleaned_ai_reply = clean_content(ai_reply).strip()
            cleaned_cot_content = cot_content.strip() if cot_content else None
            if cleaned_ai_reply:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": cleaned_ai_reply,
                    "cot": cleaned_cot_content
                })
                st.session_state.pending_generation = False
                print_payload_history()  # Debug: show updated payload history
                st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            st.session_state.pending_generation = False

# --- Render Action Icons for Last Conversation Pair ---
if (not st.session_state.pending_generation and 
    st.session_state.chat_history and 
    st.session_state.chat_history[-1]["role"] == "assistant"):
    action_container = st.container()
    with action_container:
        spacer, col_regen, col_remove = st.columns([4, 1, 1])
        if col_regen.button("🔄", key="regen_last", help="Regenerate response"):
            st.session_state.chat_history.pop()
            st.session_state.pending_generation = True
            print_payload_history()
            st.rerun()
        if col_remove.button("🗑", key="delete_last", help="Delete this conversation turn"):
            if len(st.session_state.chat_history) >= 2:
                st.session_state.chat_history = st.session_state.chat_history[:-2]
            print_payload_history()
            st.rerun()
