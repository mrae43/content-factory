from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class Tool:
    """A registered capability that an agent can invoke.

    One tool serves both direct (DI) and LLM-driven invocation:
    - *DI tools* are called directly by agent code (e.g. ``await
      tools['semantic_search'].callable(...)``).
    - *LLM tools* additionally expose ``llm_schema`` for
      ``model.bind_tools()`` so the LLM can request them.

    Permissions are symmetric — both the tool and the agent declare
    which agents are allowed to use which tools so misconfigured wiring
    is caught early.
    """

    name: str
    description: str
    callable: Callable[..., Any]
    llm_schema: Optional[Dict[str, Any]] = None
    permissions: Set[str] = field(default_factory=lambda: {"*"})


class ToolRegistry:
    """Singleton registry for all pipeline tools.

    Tools are registered once at startup and remain available for the
    lifetime of the process.  The registry provides both name-based
    lookup and permission-filtered queries so agents only see tools
    they are allowed to call.
    """

    _instance: Optional["ToolRegistry"] = None

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, Tool] = {}
        return cls._instance

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def get_permitted_tools(self, agent_class_name: str) -> List[Tool]:
        """Return tools the given agent class is allowed to use."""
        result: List[Tool] = []
        for tool in self._tools.values():
            if "*" in tool.permissions or agent_class_name in tool.permissions:
                result.append(tool)
        return result

    def get_llm_tools(self, agent_class_name: str) -> List[Tool]:
        """Return permitted tools that also expose an LLM-compatible schema."""
        return [
            t
            for t in self.get_permitted_tools(agent_class_name)
            if t.llm_schema is not None
        ]

    def clear(self) -> None:
        self._tools.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def register_standard_tools() -> None:
    registry = ToolRegistry()
    if "generate_image" not in registry:
        from app.services.image_gen import make_generate_image_tool

        registry.register(make_generate_image_tool())
    if "upload_image" not in registry:
        from app.storage.adapter import make_upload_image_tool

        registry.register(make_upload_image_tool())
    if "semantic_search" not in registry:
        # registered externally when vector_store is available
        pass
