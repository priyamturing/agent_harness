"""Tool schema fixer for Python reserved keywords."""

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model


# Python reserved keywords that can't be used as parameter names
RESERVED_KEYWORDS = {
    "self", "class", "def", "return", "if", "else", "elif", "while", "for",
    "in", "is", "and", "or", "not", "None", "True", "False", "import",
    "from", "as", "with", "try", "except", "finally", "raise", "yield",
    "lambda", "pass", "break", "continue", "global", "nonlocal", "del",
    "assert", "async", "await", "match", "case", "type",
}


def fix_tool_schemas(tools: list[BaseTool]) -> list[BaseTool]:
    """Fix tools that have Python reserved keywords as parameters.

    Renames reserved keywords by appending underscore (e.g., 'self' → 'self_').

    Args:
        tools: List of tools from MCP

    Returns:
        List of fixed tools
    """
    fixed_tools = []

    for tool in tools:
        # Check if tool has args_schema with reserved keywords
        if not hasattr(tool, "args_schema") or tool.args_schema is None:
            fixed_tools.append(tool)
            continue

        schema = tool.args_schema
        
        # Ensure schema is actually a class before checking subclass
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            fixed_tools.append(tool)
            continue

        # Get schema fields
        schema_fields = schema.model_fields
        has_reserved = any(field_name in RESERVED_KEYWORDS for field_name in schema_fields)

        if not has_reserved:
            # No reserved keywords, use as-is
            fixed_tools.append(tool)
            continue

        # Build new schema with renamed fields
        new_fields = {}
        field_mapping = {}  # old_name -> new_name

        for field_name, field_info in schema_fields.items():
            if field_name in RESERVED_KEYWORDS:
                # Rename reserved keyword
                new_name = f"{field_name}_"
                field_mapping[field_name] = new_name
                new_fields[new_name] = (field_info.annotation, field_info)
            else:
                new_fields[field_name] = (field_info.annotation, field_info)

        # Create new schema model
        new_schema = create_model(
            f"{schema.__name__}_Fixed",
            **new_fields,
        )

        # Create wrapper tool that translates parameters
        # IMPORTANT: Copy all metadata and configuration from original tool
        class FixedTool(BaseTool):
            name: str = tool.name
            description: str = tool.description
            args_schema: type[BaseModel] = new_schema
            
            # Copy critical tool configuration and metadata
            return_direct: bool = tool.return_direct
            verbose: bool = tool.verbose
            callbacks: Any = tool.callbacks
            tags: Any = tool.tags
            metadata: Any = tool.metadata
            handle_tool_error: Any = tool.handle_tool_error
            handle_validation_error: Any = tool.handle_validation_error
            response_format: str = tool.response_format
            
            # Store field mapping and original tool to avoid closure variable capture bug
            _field_mapping: dict[str, str] = field_mapping
            _original_tool: BaseTool = tool

            async def _arun(self, **kwargs: Any) -> Any:
                # Reverse the field mapping for the actual tool call
                original_kwargs = {}
                for key, value in kwargs.items():
                    # Find original name
                    original_name = None
                    for orig, new in self._field_mapping.items():
                        if new == key:
                            original_name = orig
                            break
                    original_kwargs[original_name or key] = value

                # Call original tool
                if hasattr(self._original_tool, "ainvoke"):
                    return await self._original_tool.ainvoke(original_kwargs)
                else:
                    return self._original_tool.invoke(original_kwargs)

            def _run(self, **kwargs: Any) -> Any:
                # Reverse the field mapping
                original_kwargs = {}
                for key, value in kwargs.items():
                    original_name = None
                    for orig, new in self._field_mapping.items():
                        if new == key:
                            original_name = orig
                            break
                    original_kwargs[original_name or key] = value

                return self._original_tool.invoke(original_kwargs)

        fixed_tool = FixedTool()
        fixed_tools.append(fixed_tool)

    return fixed_tools

