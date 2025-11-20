"""Utility functions and types for verifiers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Union


class ComparisonType(str, Enum):
    """Supported comparison types for verifiers.
    
    Each enum value is a string to ensure JSON serialization compatibility.
    """
    EQUALS = "equals"
    NOT_EQUAL = "not_equal"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_EQUAL = "greater_than_equal"
    LESS_THAN_EQUAL = "less_than_equal"
    
    @classmethod
    def from_string(cls, value: str) -> "ComparisonType":
        """Convert string (including aliases) to ComparisonType enum.
        
        Supports common aliases like 'eq', '==', 'gt', '>', etc.
        
        Args:
            value: Comparison type string (case-insensitive)
            
        Returns:
            ComparisonType enum value
            
        Raises:
            ValueError: If value is not a recognized comparison type
        """
        normalized = value.lower().strip()
        alias_map = {
            "equals": cls.EQUALS,
            "eq": cls.EQUALS,
            "==": cls.EQUALS,
            "equal": cls.EQUALS,
            "not_equal": cls.NOT_EQUAL,
            "not_equals": cls.NOT_EQUAL,
            "neq": cls.NOT_EQUAL,
            "ne": cls.NOT_EQUAL,
            "!=": cls.NOT_EQUAL,
            "greater_than": cls.GREATER_THAN,
            "gt": cls.GREATER_THAN,
            ">": cls.GREATER_THAN,
            "less_than": cls.LESS_THAN,
            "lt": cls.LESS_THAN,
            "<": cls.LESS_THAN,
            "greater_than_equal": cls.GREATER_THAN_EQUAL,
            "greater_than_or_equal": cls.GREATER_THAN_EQUAL,
            "greater_than_or_equal_to": cls.GREATER_THAN_EQUAL,
            "gte": cls.GREATER_THAN_EQUAL,
            ">=": cls.GREATER_THAN_EQUAL,
            "greater_or_equal": cls.GREATER_THAN_EQUAL,
            "less_than_equal": cls.LESS_THAN_EQUAL,
            "less_than_or_equal": cls.LESS_THAN_EQUAL,
            "less_than_or_equal_to": cls.LESS_THAN_EQUAL,
            "lte": cls.LESS_THAN_EQUAL,
            "<=": cls.LESS_THAN_EQUAL,
            "less_or_equal": cls.LESS_THAN_EQUAL,
        }
        if normalized not in alias_map:
            raise ValueError(
                f"Unsupported comparison type: '{value}'. "
                f"Supported: equals/eq/==/equal, not_equal/not_equals/neq/ne/!=, "
                f"greater_than/gt/>, less_than/lt/<, "
                f"greater_than_equal/greater_than_or_equal/greater_than_or_equal_to/gte/>=, "
                f"less_than_equal/less_than_or_equal/less_than_or_equal_to/lte/<="
            )
        return alias_map[normalized]


def compare(actual: Any, expected: Any, comparison: Union[ComparisonType, str]) -> bool:
    """Compare actual vs expected using comparison type.
    
    This is a general-purpose comparison utility that can be used by any verifier.
    
    Args:
        actual: Value from verification source (e.g., SQL query result)
        expected: Expected value to compare against
        comparison: ComparisonType enum or string (with aliases)
    
    Supported comparisons (with aliases):
    - equals / eq / == / equal : exact equality (works with None)
    - not_equal / not_equals / neq / ne / != : inequality (works with None)
    - greater_than / gt / > : actual > expected (requires non-None values)
    - less_than / lt / < : actual < expected (requires non-None values)
    - greater_than_equal / greater_than_or_equal / greater_than_or_equal_to / gte / >= : actual >= expected (requires non-None values)
    - less_than_equal / less_than_or_equal / less_than_or_equal_to / lte / <= : actual <= expected (requires non-None values)
    
    Returns:
        True if comparison succeeds, False otherwise
    
    Raises:
        ValueError: If comparison type is unsupported
        TypeError: If actual is None for ordered comparisons
        
    Examples:
        >>> compare(5, 3, ComparisonType.GREATER_THAN)
        True
        >>> compare(10, 10, "equals")
        True
        >>> compare(7, 5, "<=")
        False
        >>> compare(None, 0, "!=")
        True
    """
    if isinstance(comparison, str):
        comparison = ComparisonType.from_string(comparison)
    
    if comparison == ComparisonType.EQUALS:
        return actual == expected
    
    if comparison == ComparisonType.NOT_EQUAL:
        return actual != expected
    
    if actual is None:
        raise TypeError(
            f"Cannot perform ordered comparison '{comparison.value}' with None value. "
            f"Actual value is None."
        )
    
    try:
        if comparison == ComparisonType.GREATER_THAN:
            return actual > expected
        elif comparison == ComparisonType.LESS_THAN:
            return actual < expected
        elif comparison == ComparisonType.GREATER_THAN_EQUAL:
            return actual >= expected
        elif comparison == ComparisonType.LESS_THAN_EQUAL:
            return actual <= expected
    except TypeError as e:
        raise TypeError(
            f"Cannot compare values: actual={actual!r} ({type(actual).__name__}) "
            f"vs expected={expected!r} ({type(expected).__name__}). "
            f"Ensure values are comparable types (numbers, dates, strings)."
        ) from e
    
    raise ValueError(f"Unsupported comparison type: {comparison}")

