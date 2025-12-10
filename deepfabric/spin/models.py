"""Data models for Spin integration."""

from pydantic import BaseModel, Field


class SpinExecutionResult(BaseModel):
    """Result from Spin tool execution."""

    success: bool = Field(description="Whether execution succeeded")
    result: str = Field(description="Tool output or error message")
    error_type: str | None = Field(
        default=None, description="Error type if failed (e.g., 'FileNotFound', 'Timeout')"
    )


class SpinHealthResponse(BaseModel):
    """Response from Spin health check."""

    status: str = Field(description="Service status")
    components: list[str] = Field(default_factory=list, description="Available components")


class SpinComponentsResponse(BaseModel):
    """Response from Spin components listing."""

    components: list[str] = Field(default_factory=list, description="Available component names")
