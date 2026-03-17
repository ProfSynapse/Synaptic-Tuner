from shared.model_loading.registry import ModelLoaderRegistry
from shared.upload import UploadOrchestrator


def test_builtin_unsloth_loader_is_available_without_package_side_effects():
    loader_names = ModelLoaderRegistry.list_loaders()

    assert "unsloth" in loader_names
    assert ModelLoaderRegistry.get("unsloth").__class__.__name__ == "UnslothModelLoader"


def test_shared_upload_can_be_imported_after_registry_lookup():
    assert UploadOrchestrator.__name__ == "UploadOrchestrator"
