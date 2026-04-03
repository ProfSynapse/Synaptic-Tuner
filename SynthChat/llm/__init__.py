"""SynthChat LLM - Client pool management and LLM calling utilities.

Location: SynthChat/llm/__init__.py
Purpose: Subpackage for managing LLM client pools (caching, stage-specific
         client selection) and calling LLMs with retry logic.
Usage: from SynthChat.llm.client_pool import LLMClientPool
       from SynthChat.llm.caller import call_llm, call_llm_structured
"""
