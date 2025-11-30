"""
Model discovery service.

Location: /mnt/f/Code/Toolset-Training/tuner/discovery/models.py
Purpose: Discover available models from evaluation backends
Used by: Evaluation handler to list models for testing

This module implements the ModelDiscovery service which queries evaluation backends
(Ollama, LM Studio) to retrieve lists of available models for inference.

Pattern migrated from: tuner.py lines 705-746 (_list_ollama_models and _list_lmstudio_models)
"""

from typing import List

from tuner.backends.registry import EvaluationBackendRegistry


class ModelDiscovery:
    """
    Discover available models from evaluation backends.

    This service acts as a facade over evaluation backends, providing a unified
    interface to query available models from different backend systems (Ollama, LM Studio).

    Example:
        from tuner.discovery import ModelDiscovery

        # Discover Ollama models
        discovery = ModelDiscovery()
        ollama_models = discovery.discover('ollama')
        print(f"Found {len(ollama_models)} Ollama models")

        # Discover LM Studio models
        lms_models = discovery.discover('lmstudio')
        for model in lms_models:
            print(f"  - {model}")

        # Handle backend errors gracefully
        if not ollama_models:
            print("No Ollama models found or Ollama not running")
    """

    @staticmethod
    def discover(backend_name: str) -> List[str]:
        """
        Discover available models from a specific backend.

        Queries the specified evaluation backend to retrieve the list of
        available models. Handles errors gracefully by returning an empty
        list if the backend is unavailable or no models are found.

        Args:
            backend_name: Backend identifier ('ollama' or 'lmstudio')

        Returns:
            List of model names/identifiers.
            Returns empty list if backend is unavailable, not running,
            or has no models available.

        Example:
            # Discover models from Ollama
            models = ModelDiscovery.discover('ollama')
            if models:
                print("Ollama models:")
                for model in models:
                    print(f"  - {model}")
            else:
                print("Ollama not running or no models found")

            # Discover models from LM Studio
            lms_models = ModelDiscovery.discover('lmstudio')
            if lms_models:
                print(f"Found {len(lms_models)} LM Studio models")

        Backend-specific behavior:
            - Ollama: Calls 'ollama list' command, parses output
            - LM Studio: Queries HTTP API at localhost:1234/v1/models

        Error handling:
            - Backend not registered: Returns empty list
            - Backend not running: Returns empty list
            - API/command error: Returns empty list
            - No models available: Returns empty list
        """
        try:
            # Get backend from registry
            backend = EvaluationBackendRegistry.get(backend_name)

            # Call backend's list_models method
            models = backend.list_models()

            # Return models (empty list if None)
            return models if models else []

        except ValueError:
            # Backend not registered
            return []
        except Exception:
            # Any other error (backend not running, API error, etc.)
            return []
