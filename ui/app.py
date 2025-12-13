"""Streamlit UI for the BOG Offers Chatbot.

Provides a web-based chat interface for querying Bank of Georgia offers.

Usage:
    streamlit run ui/app.py
    # or
    python launch.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """Get the repository root directory.
    
    Returns:
        Path to the repository root (parent of ui/).
    """
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    """Ensure the repository root is on sys.path for imports."""
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
    """Initialize and cache the chatbot instance.
    
    Returns:
        Configured BOGChatbot instance.
        
    Raises:
        RuntimeError: If chatbot initialization fails.
    """
    _ensure_repo_on_path()

    # Ensure .env is loaded from repo root
    try:
        from dotenv import load_dotenv
        load_dotenv(_repo_root() / ".env")
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")

    from llm.chatbot import build_default_chatbot
    return build_default_chatbot()


def main() -> None:
    """Main entry point for the Streamlit app."""
    st.title("BOG Offers Chatbot")
    st.caption("შენი შეთავაზებების ასისტენტი")

    # Sidebar with controls
    with st.sidebar:
        st.header("პარამეტრები")
        
        if st.button("🔄 ახალი საუბარი", help="დაიწყე ახალი საუბარი"):
            st.session_state.messages = []
            try:
                bot = _get_bot()
                bot.reset_conversation()
            except Exception:
                pass
            st.rerun()
        
        st.markdown("---")
        st.markdown(
            "**გამოყენება:**\n\n"
            "- დასვი კითხვა შეთავაზებებზე\n"
            "- მაგ: *'რესტორნები ქეშბექით'*\n"
            "- მაგ: *'20% ფასდაკლება'*"
        )

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
            error_msg = f"შეცდომა: {str(e)}"
            logger.error(f"Chat error: {e}", exc_info=True)
            st.error(error_msg)
            st.info(
                "შესაძლო მიზეზები:\n"
                "- Qdrant სერვერთან კავშირის პრობლემა\n"
                "- API გასაღების პრობლემა\n"
                "- შეამოწმე .env ფაილი"
            )


if __name__ == "__main__":
    main()
