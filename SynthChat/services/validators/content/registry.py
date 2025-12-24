"""Registry of content validators by type name."""

from typing import Dict, Type, List

from ..base import BaseContentValidator
from .xml_validator import XmlContentValidator
from .json_validator import JsonContentValidator
from .regex_validator import RegexContentValidator
from .yaml_validator import YamlContentValidator
from .code_validator import CodeContentValidator


class ContentValidatorRegistry:
    """Registry of content validators by type name."""

    _validators: Dict[str, Type[BaseContentValidator]] = {
        "xml": XmlContentValidator,
        "json": JsonContentValidator,
        "regex": RegexContentValidator,
        "yaml": YamlContentValidator,
        "code": CodeContentValidator,
    }

    @classmethod
    def get(cls, validator_type: str) -> Type[BaseContentValidator]:
        """Get validator class by type name."""
        if validator_type not in cls._validators:
            raise ValueError(f"Unknown validator type: {validator_type}")
        return cls._validators[validator_type]

    @classmethod
    def register(cls, type_name: str, validator_class: Type[BaseContentValidator]) -> None:
        """Register a new validator type."""
        cls._validators[type_name] = validator_class

    @classmethod
    def available_types(cls) -> List[str]:
        """List available validator types."""
        return list(cls._validators.keys())

    @classmethod
    def has_type(cls, validator_type: str) -> bool:
        """Check if a validator type is registered."""
        return validator_type in cls._validators
