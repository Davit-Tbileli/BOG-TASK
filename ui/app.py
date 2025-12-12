from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="BOG Offers Chatbot",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def _get_bot():
    # Lazy import so Streamlit starts fast.
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
