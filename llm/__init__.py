"""
LLM integration module for conversational recommendations.
"""

from .prompts import PromptTemplates
from .chatbot import BOGChatbot

__all__ = ['PromptTemplates', 'BOGChatbot']
