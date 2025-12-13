"""LLM module for BOG Offers Chatbot.

Provides chatbot functionality with LLM integration and prompt templates.
"""

from .chatbot import BOGChatbot, build_default_chatbot
from .prompts import PromptTemplates

__all__ = ['BOGChatbot', 'build_default_chatbot', 'PromptTemplates']
