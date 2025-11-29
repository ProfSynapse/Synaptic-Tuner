"""
Base uploader abstraction.
"""

from abc import abstractmethod
from pathlib import Path

from ..core.interfaces import IUploader
from ..core.types import RepositoryId, Credential


class BaseUploader(IUploader):
    """
    Base implementation of uploader.
    """

    @abstractmethod
    def upload_model(
        self,
        local_path: Path,
        repo_id: RepositoryId,
        credential: Credential,
        private: bool = False,
        **options
    ) -> str:
        """
        Upload model to repository.

        Args:
            local_path: Path to the local model
            repo_id: Target repository ID
            credential: Authentication credential
            private: Whether to create private repository
            **options: Uploader-specific options

        Returns:
            URL of the uploaded model
        """
        pass

    @abstractmethod
    def upload_file(
        self,
        file_path: Path,
        repo_id: RepositoryId,
        credential: Credential,
        path_in_repo: str
    ) -> None:
        """
        Upload a single file to repository.

        Args:
            file_path: Path to the local file
            repo_id: Target repository ID
            credential: Authentication credential
            path_in_repo: Path within the repository
        """
        pass

    @abstractmethod
    def validate_credential(self, credential: Credential) -> bool:
        """Validate that the credential is valid."""
        pass
