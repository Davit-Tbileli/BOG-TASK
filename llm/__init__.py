from __future__ import annotations
from typing import Any

__all__ = ["BOGChatbot", "build_default_chatbot", "PromptTemplates"]


def __getattr__(name: str) -> Any:  # PEP 562
	if name in {"BOGChatbot", "build_default_chatbot"}:
		from .chatbot import BOGChatbot, build_default_chatbot

		return {"BOGChatbot": BOGChatbot, "build_default_chatbot": build_default_chatbot}[name]

	if name == "PromptTemplates":
		from .prompts import PromptTemplates

		return PromptTemplates

	raise AttributeError(name)
