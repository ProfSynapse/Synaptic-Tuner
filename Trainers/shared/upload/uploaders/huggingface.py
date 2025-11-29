"""
HuggingFace Hub uploader.

Uploads models and files to HuggingFace Hub.
"""

from pathlib import Path
from typing import List, Optional

from .base import BaseUploader
from ..core.types import RepositoryId, Credential
from ..core.exceptions import UploadError, AuthenticationError, DependencyError


class HuggingFaceUploader(BaseUploader):
    """
    Uploader for HuggingFace Hub.

    Uses huggingface_hub library for API interactions.
    """

    def __init__(self):
        """Initialize uploader."""
        self._api = None

    @property
    def name(self) -> str:
        return "huggingface"

    def _get_api(self):
        """Lazily import and return HfApi instance."""
        if self._api is None:
            try:
                from huggingface_hub import HfApi
                self._api = HfApi()
            except ImportError as e:
                raise DependencyError(
                    "huggingface_hub",
                    "Install with: pip install huggingface_hub"
                ) from e
        return self._api

    def validate_credential(self, credential: Credential) -> bool:
        """
        Validate that the HuggingFace token is valid.

        Args:
            credential: HuggingFace token

        Returns:
            True if token is valid
        """
        try:
            api = self._get_api()
            # Try to get user info - this will fail if token is invalid
            api.whoami(token=credential)
            return True
        except Exception:
            return False

    def upload_model(
        self,
        local_path: Path,
        repo_id: RepositoryId,
        credential: Credential,
        private: bool = False,
        **options
    ) -> str:
        """
        Upload model directory to HuggingFace Hub.

        Args:
            local_path: Path to the local model directory
            repo_id: Target repository ID (username/model-name)
            credential: HuggingFace token
            private: Whether to create private repository
            **options: Additional options

        Returns:
            URL of the uploaded model
        """
        api = self._get_api()

        print("\n" + "=" * 60)
        print("UPLOADING TO HUGGINGFACE")
        print("=" * 60)
        print(f"Local path: {local_path}")
        print(f"Repository: {repo_id}")
        print(f"Private: {private}")
        print()

        try:
            # Create or get repository
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                token=credential,
                exist_ok=True
            )

            # Upload folder
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                repo_type="model",
                token=credential
            )

            url = f"https://huggingface.co/{repo_id}"
            print(f"\n✓ Model uploaded successfully!")
            print(f"View at: {url}")

            return url

        except Exception as e:
            raise UploadError(f"Failed to upload model: {e}") from e

    def upload_file(
        self,
        file_path: Path,
        repo_id: RepositoryId,
        credential: Credential,
        path_in_repo: str
    ) -> None:
        """
        Upload a single file to HuggingFace Hub.

        Args:
            file_path: Path to the local file
            repo_id: Target repository ID
            credential: HuggingFace token
            path_in_repo: Path within the repository
        """
        api = self._get_api()

        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                token=credential
            )
        except Exception as e:
            raise UploadError(f"Failed to upload file {file_path}: {e}") from e

    def upload_files(
        self,
        files: List[Path],
        repo_id: RepositoryId,
        credential: Credential,
        path_prefix: str = ""
    ) -> None:
        """
        Upload multiple files to HuggingFace Hub.

        Args:
            files: List of file paths to upload
            repo_id: Target repository ID
            credential: HuggingFace token
            path_prefix: Prefix for paths in repository
        """
        print("\n" + "=" * 60)
        print("UPLOADING FILES")
        print("=" * 60)
        print(f"Repository: {repo_id}")
        print(f"Files to upload: {len(files)}")
        print()

        for i, file_path in enumerate(files, 1):
            path_in_repo = f"{path_prefix}{file_path.name}" if path_prefix else file_path.name
            print(f"[{i}/{len(files)}] Uploading {file_path.name}...")

            self.upload_file(file_path, repo_id, credential, path_in_repo)
            print("  ✓ Uploaded")

        print(f"\n✓ All files uploaded!")
        print(f"View at: https://huggingface.co/{repo_id}/tree/main")

    def get_repo_url(self, repo_id: RepositoryId) -> str:
        """
        Get the URL for a repository.

        Args:
            repo_id: Repository ID

        Returns:
            Repository URL
        """
        return f"https://huggingface.co/{repo_id}"
