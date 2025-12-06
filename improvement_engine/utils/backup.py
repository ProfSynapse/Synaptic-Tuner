"""Backup utilities for dataset files."""

import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


class BackupManager:
    """Manages dataset file backups."""

    def __init__(self, enabled: bool = True):
        """
        Initialize backup manager.

        Args:
            enabled: Whether backups are enabled
        """
        self.enabled = enabled

    def create_backup(self, file_path: str, backup_dir: Optional[str] = None) -> Optional[str]:
        """
        Create a timestamped backup of a file.

        Args:
            file_path: Path to file to backup
            backup_dir: Optional directory for backups (defaults to same dir as file)

        Returns:
            Path to backup file if created, None if backups disabled
        """
        if not self.enabled:
            return None

        source = Path(file_path)
        if not source.exists():
            return None

        # Determine backup location
        if backup_dir:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
        else:
            backup_path = source.parent

        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source.stem}_backup_{timestamp}{source.suffix}"
        backup_file = backup_path / backup_name

        # Copy file
        shutil.copy2(source, backup_file)

        return str(backup_file)

    def restore_backup(self, backup_file: str, target_file: str) -> bool:
        """
        Restore a file from backup.

        Args:
            backup_file: Path to backup file
            target_file: Path to restore to

        Returns:
            True if successful
        """
        backup_path = Path(backup_file)
        target_path = Path(target_file)

        if not backup_path.exists():
            return False

        # Create backup of current file before overwriting
        if target_path.exists():
            temp_backup = self.create_backup(str(target_path))

        # Restore from backup
        shutil.copy2(backup_path, target_path)

        return True

    def list_backups(self, file_path: str, backup_dir: Optional[str] = None) -> list[str]:
        """
        List all backups for a file.

        Args:
            file_path: Original file path
            backup_dir: Optional backup directory

        Returns:
            List of backup file paths
        """
        source = Path(file_path)

        if backup_dir:
            search_dir = Path(backup_dir)
        else:
            search_dir = source.parent

        if not search_dir.exists():
            return []

        # Find all backups matching pattern
        pattern = f"{source.stem}_backup_*{source.suffix}"
        backups = list(search_dir.glob(pattern))

        # Sort by modification time (newest first)
        backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return [str(b) for b in backups]
