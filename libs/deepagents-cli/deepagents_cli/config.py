"""Configuration, constants, and model creation for the CLI."""

import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import dotenv
from langchain_core.language_models import BaseChatModel
from rich.console import Console

dotenv.load_dotenv()

# Color scheme
COLORS = {
    "primary": "#10b981",
    "dim": "#6b7280",
    "user": "#ffffff",
    "agent": "#10b981",
    "thinking": "#34d399",
    "tool": "#fbbf24",
}

# ASCII art banner
DEEP_AGENTS_ASCII = """
 ██████╗  ███████╗ ███████╗ ██████╗
 ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
 ██║  ██║ █████╗   █████╗   ██████╔╝
 ██║  ██║ ██╔══╝   ██╔══╝   ██╔═══╝
 ██████╔╝ ███████╗ ███████╗ ██║
 ╚═════╝  ╚══════╝ ╚══════╝ ╚═╝

  █████╗   ██████╗  ███████╗ ███╗   ██╗ ████████╗ ███████╗
 ██╔══██╗ ██╔════╝  ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██╔════╝
 ███████║ ██║  ███╗ █████╗   ██╔██╗ ██║    ██║    ███████╗
 ██╔══██║ ██║   ██║ ██╔══╝   ██║╚██╗██║    ██║    ╚════██║
 ██║  ██║ ╚██████╔╝ ███████╗ ██║ ╚████║    ██║    ███████║
 ╚═╝  ╚═╝  ╚═════╝  ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚══════╝
"""

# Interactive commands
COMMANDS = {
    "clear": "Clear screen and reset conversation",
    "help": "Show help information",
    "tokens": "Show token usage for current session",
    "quit": "Exit the CLI",
    "exit": "Exit the CLI",
}

# Maximum argument length for display
MAX_ARG_LENGTH = 150

# Agent configuration
config = {"recursion_limit": 1000}

# Rich console instance
console = Console(highlight=False)


def _find_project_root(start_path: Path | None = None) -> Path | None:
    """Find the project root by looking for .git directory.

    Walks up the directory tree from start_path (or cwd) looking for a .git
    directory, which indicates the project root.

    Args:
        start_path: Directory to start searching from. Defaults to current working directory.

    Returns:
        Path to the project root if found, None otherwise.
    """
    current = Path(start_path or Path.cwd()).resolve()

    # Walk up the directory tree
    for parent in [current, *list(current.parents)]:
        git_dir = parent / ".git"
        if git_dir.exists():
            return parent

    return None


def _find_project_agent_md(project_root: Path) -> list[Path]:
    """Find project-specific agent.md file(s).

    Checks two locations and returns ALL that exist:
    1. project_root/.deepagents/agent.md
    2. project_root/agent.md

    Both files will be loaded and combined if both exist.

    Args:
        project_root: Path to the project root directory.

    Returns:
        List of paths to project agent.md files (may contain 0, 1, or 2 paths).
    """
    paths = []

    # Check .deepagents/agent.md (preferred)
    deepagents_md = project_root / ".deepagents" / "agent.md"
    if deepagents_md.exists():
        paths.append(deepagents_md)

    # Check root agent.md (fallback, but also include if both exist)
    root_md = project_root / "agent.md"
    if root_md.exists():
        paths.append(root_md)

    return paths


@dataclass
class Settings:
    """Global settings and environment detection for deepagents-cli.

    This class is initialized once at startup and provides access to:
    - Available models and API keys
    - Current project information
    - Tool availability (e.g., Tavily)
    - File system paths

    Attributes:
        project_root: Current project root directory (if in a git project)

        openai_api_key: OpenAI API key if available
        anthropic_api_key: Anthropic API key if available
        tavily_api_key: Tavily API key if available
    """

    # API keys
    openai_api_key: str | None
    anthropic_api_key: str | None
    google_api_key: str | None
    zhipu_api_key: str | None
    tavily_api_key: str | None

    # Project information
    project_root: Path | None

    @classmethod
    def from_environment(cls, *, start_path: Path | None = None) -> "Settings":
        """Create settings by detecting the current environment.

        Args:
            start_path: Directory to start project detection from (defaults to cwd)

        Returns:
            Settings instance with detected configuration
        """
        # Detect API keys
        openai_key = os.environ.get("OPENAI_API_KEY")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        google_key = os.environ.get("GOOGLE_API_KEY")
        zhipu_key = os.environ.get("ZHIPU_API_KEY")
        tavily_key = os.environ.get("TAVILY_API_KEY")

        # Detect project
        project_root = _find_project_root(start_path)

        return cls(
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            google_api_key=google_key,
            zhipu_api_key=zhipu_key,
            tavily_api_key=tavily_key,
            project_root=project_root,
        )

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.openai_api_key is not None

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.anthropic_api_key is not None

    @property
    def has_google(self) -> bool:
        """Check if Google API key is configured."""
        return self.google_api_key is not None

    @property
    def has_zhipu(self) -> bool:
        """Check if Zhipu API key is configured."""
        return self.zhipu_api_key is not None

    @property
    def has_tavily(self) -> bool:
        """Check if Tavily API key is configured."""
        return self.tavily_api_key is not None

    @property
    def has_project(self) -> bool:
        """Check if currently in a git project."""
        return self.project_root is not None

    @property
    def user_deepagents_dir(self) -> Path:
        """Get the base user-level .deepagents directory.

        Returns:
            Path to ~/.deepagents
        """
        return Path.home() / ".deepagents"

    def get_user_agent_md_path(self, agent_name: str) -> Path:
        """Get user-level agent.md path for a specific agent.

        Returns path regardless of whether the file exists.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}/agent.md
        """
        return Path.home() / ".deepagents" / agent_name / "agent.md"

    def get_project_agent_md_path(self) -> Path | None:
        """Get project-level agent.md path.

        Returns path regardless of whether the file exists.

        Returns:
            Path to {project_root}/.deepagents/agent.md, or None if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "agent.md"

    @staticmethod
    def _is_valid_agent_name(agent_name: str) -> bool:
        """Validate prevent invalid filesystem paths and security issues."""
        if not agent_name or not agent_name.strip():
            return False
        # Allow only alphanumeric, hyphens, underscores, and whitespace
        return bool(re.match(r"^[a-zA-Z0-9_\-\s]+$", agent_name))

    def get_agent_dir(self, agent_name: str) -> Path:
        """Get the global agent directory path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}
        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. "
                "Agent names can only contain letters, numbers, hyphens, underscores, and spaces."
            )
            raise ValueError(msg)
        return Path.home() / ".deepagents" / agent_name

    def ensure_agent_dir(self, agent_name: str) -> Path:
        """Ensure the global agent directory exists and return its path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}
        """
        if not self._is_valid_agent_name(agent_name):
            msg = (
                f"Invalid agent name: {agent_name!r}. "
                "Agent names can only contain letters, numbers, hyphens, underscores, and spaces."
            )
            raise ValueError(msg)
        agent_dir = self.get_agent_dir(agent_name)
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir

    def ensure_project_deepagents_dir(self) -> Path | None:
        """Ensure the project .deepagents directory exists and return its path.

        Returns:
            Path to project .deepagents directory, or None if not in a project
        """
        if not self.project_root:
            return None

        project_deepagents_dir = self.project_root / ".deepagents"
        project_deepagents_dir.mkdir(parents=True, exist_ok=True)
        return project_deepagents_dir

    def get_user_skills_dir(self, agent_name: str) -> Path:
        """Get user-level skills directory path for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}/skills/
        """
        return self.get_agent_dir(agent_name) / "skills"

    def ensure_user_skills_dir(self, agent_name: str) -> Path:
        """Ensure user-level skills directory exists and return its path.

        Args:
            agent_name: Name of the agent

        Returns:
            Path to ~/.deepagents/{agent_name}/skills/
        """
        skills_dir = self.get_user_skills_dir(agent_name)
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir

    def get_project_skills_dir(self) -> Path | None:
        """Get project-level skills directory path.

        Returns:
            Path to {project_root}/.deepagents/skills/, or None if not in a project
        """
        if not self.project_root:
            return None
        return self.project_root / ".deepagents" / "skills"

    def ensure_project_skills_dir(self) -> Path | None:
        """Ensure project-level skills directory exists and return its path.

        Returns:
            Path to {project_root}/.deepagents/skills/, or None if not in a project
        """
        if not self.project_root:
            return None
        skills_dir = self.get_project_skills_dir()
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir


# Global settings instance (initialized once)
settings = Settings.from_environment()


class SessionState:
    """Holds mutable session state (auto-approve mode, etc)."""

    def __init__(self, auto_approve: bool = False, no_splash: bool = False) -> None:
        self.auto_approve = auto_approve
        self.no_splash = no_splash
        self.exit_hint_until: float | None = None
        self.exit_hint_handle = None
        self.thread_id = str(uuid.uuid4())

    def toggle_auto_approve(self) -> bool:
        """Toggle auto-approve and return new state."""
        self.auto_approve = not self.auto_approve
        return self.auto_approve


def get_default_coding_instructions() -> str:
    """Get the default coding agent instructions.

    These are the immutable base instructions that cannot be modified by the agent.
    Long-term memory (agent.md) is handled separately by the middleware.
    """
    default_prompt_path = Path(__file__).parent / "default_agent_prompt.md"
    return default_prompt_path.read_text()


def create_model() -> BaseChatModel:
    """Create the appropriate model based on available API keys.

    Uses the global settings instance to determine which model to create.

    Returns:
        ChatModel instance (OpenAI or Anthropic)

    Raises:
        SystemExit if no API key is configured
    """
    if settings.has_openai:
        from langchain_openai import ChatOpenAI

        model_name = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        base_url = os.environ.get("OPENAI_BASE_URL")
        console.print(f"[dim]Using OpenAI model: {model_name}[/dim]")
        if base_url:
            console.print(f"[dim]Using OpenAI base URL: {base_url}[/dim]")
        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
        )
    if settings.has_anthropic:
        from langchain_anthropic import ChatAnthropic

        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        console.print(f"[dim]Using Anthropic model: {model_name}[/dim]")
        if base_url:
            console.print(f"[dim]Using Anthropic base URL: {base_url}[/dim]")
        return ChatAnthropic(
            model_name=model_name,
            base_url=base_url,
            # The attribute exists, but it has a Pydantic alias which
            # causes issues in IDEs/type checkers.
            max_tokens=20_000,  # type: ignore[arg-type]
        )
    if settings.has_google:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = os.environ.get("GOOGLE_MODEL", "gemini-3-pro-preview")
        console.print(f"[dim]Using Google Gemini model: {model_name}[/dim]")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_tokens=None,
        )
    if settings.has_zhipu:
        # Use zhipuai SDK directly with a LangChain-compatible wrapper
        try:
            import zhipuai
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
            from langchain_core.outputs import ChatGeneration, ChatResult
            from typing import Any, List, Optional, AsyncIterator, Iterator

            class ZhipuChatModel(BaseChatModel):
                """Wrapper for Zhipu AI using zhipuai SDK."""

                model: str = "glm-4"
                api_key: str
                temperature: float = 0

                def __init__(self, model: str = "glm-4", api_key: str | None = None, temperature: float = 0,
                             **kwargs: Any):
                    # Get API key from parameter or environment
                    final_api_key = api_key or os.environ.get("ZHIPU_API_KEY", "")
                    if not final_api_key:
                        raise ValueError("ZHIPU_API_KEY is required")

                    # Initialize with proper field values for Pydantic
                    super().__init__(
                        model=model,
                        api_key=final_api_key,
                        temperature=temperature,
                        **kwargs
                    )
                    # Initialize client after Pydantic validation (not a Pydantic field)
                    object.__setattr__(self, "client", zhipuai.ZhipuAI(api_key=self.api_key))

                def _generate(
                        self,
                        messages: List[BaseMessage],
                        stop: Optional[List[str]] = None,
                        run_manager: Any = None,
                        **kwargs: Any,
                ) -> ChatResult:
                    # Convert LangChain messages to Zhipu format
                    zhipu_messages = []
                    for msg in messages:
                        if isinstance(msg, HumanMessage):
                            zhipu_messages.append({"role": "user", "content": str(msg.content)})
                        elif isinstance(msg, AIMessage):
                            zhipu_messages.append({"role": "assistant", "content": str(msg.content)})
                        elif isinstance(msg, SystemMessage):
                            zhipu_messages.append({"role": "system", "content": str(msg.content)})

                    # Call Zhipu API
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=zhipu_messages,
                            temperature=self.temperature,
                        )

                        # Extract content from response
                        if hasattr(response, 'choices') and len(response.choices) > 0:
                            message_content = response.choices[0].message.content
                        elif hasattr(response, 'data') and len(response.data) > 0:
                            message_content = response.data[0].message.content
                        else:
                            # Try to get content from response directly
                            message_content = str(response) if response else ""

                        # Convert response to LangChain format
                        ai_message = AIMessage(content=message_content)
                        generation = ChatGeneration(message=ai_message)
                        return ChatResult(generations=[generation])
                    except Exception as e:
                        # Return error in ChatResult format
                        error_msg = f"Zhipu API error: {str(e)}"
                        ai_message = AIMessage(content=error_msg)
                        generation = ChatGeneration(message=ai_message)
                        return ChatResult(generations=[generation])

                async def _astream(
                        self,
                        messages: List[BaseMessage],
                        stop: Optional[List[str]] = None,
                        run_manager: Any = None,
                        **kwargs: Any,
                ) -> AsyncIterator[ChatResult]:
                    """Async stream implementation."""
                    # For now, use non-streaming but wrap in async
                    result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                    yield result

                def _stream(
                        self,
                        messages: List[BaseMessage],
                        stop: Optional[List[str]] = None,
                        run_manager: Any = None,
                        **kwargs: Any,
                ) -> Iterator[ChatResult]:
                    """Stream implementation."""
                    # For now, use non-streaming
                    result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                    yield result

                def _should_stream(self, **kwargs: Any) -> bool:
                    """Determine if streaming should be used."""
                    # Check if streaming is requested
                    return kwargs.get("stream", False) or kwargs.get("streaming", False)

                def bind_tools(self, tools, **kwargs):
                    """Bind tools to the model (required for tool calling)."""
                    # For now, return self as tools are handled elsewhere
                    return self

                @property
                def _llm_type(self) -> str:
                    return "zhipu"

            model_name = os.environ.get("ZHIPU_MODEL", "glm-4")
            console.print(f"[dim]Using Zhipu model: {model_name}[/dim]")
            return ZhipuChatModel(model=model_name, api_key=settings.zhipu_api_key, temperature=0)
        except ImportError:
            console.print("[bold red]Error:[/bold red] zhipuai package not found.")
            console.print("Please install it with: uv pip install zhipuai")
            sys.exit(1)
    console.print("[bold red]Error:[/bold red] No API key configured.")
    console.print("\nPlease set one of the following environment variables:")
    console.print("  - OPENAI_API_KEY     (for OpenAI models like gpt-5-mini)")
    console.print("  - ANTHROPIC_API_KEY  (for Claude models)")
    console.print("  - GOOGLE_API_KEY     (for Google Gemini models)")
    console.print("  - ZHIPU_API_KEY      (for Zhipu GLM models)")
    console.print("\nExample:")
    console.print("  export ZHIPU_API_KEY=your_api_key_here")
    console.print("\nOr add it to your .env file.")
    sys.exit(1)
