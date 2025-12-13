from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


def _repo_root() -> Path:
    # ui/app.py -> repo root is one directory up.
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


st.set_page_config(
    page_title="BOG Offers Chatbot",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def _get_bot():
    # Lazy import so Streamlit starts fast.
    _ensure_repo_on_path()

    # Ensure .env is loaded from repo root even if Streamlit CWD is ui/
    try:
        from dotenv import load_dotenv

        load_dotenv(_repo_root() / ".env")
    except Exception:
        pass

    from llm.chatbot import build_default_chatbot

    return build_default_chatbot()


def main() -> None:
    st.title("BOG Offers Chatbot")
    st.caption("შენი შეთავაზებების ასისტენტი")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("მკითხე რაიმე შეთავაზებებზე…")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        try:
            bot = _get_bot()
            with st.spinner("ვეძებ შეთავაზებებს…"):
                answer = bot.chat(user_text)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
