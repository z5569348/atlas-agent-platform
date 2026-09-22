from pydantic import BaseModel, ConfigDict, Field


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    supports_temperature: bool = False
    supports_tool_calling: bool = False
    supports_structured_output: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_reasoning: bool = False
    max_output_tokens: int = Field(gt=0)

class ModelProfile(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    capabilities: ModelCapabilities

MODEL_PROFILES: dict[tuple[str, str], ModelProfile] = {
    ("openai", "gpt-5.6-luna"): ModelProfile(
        provider="openai",
        model_name="gpt-5.6-luna",
        capabilities=ModelCapabilities(
            supports_temperature=False,
            supports_tool_calling=True,
            supports_structured_output=True,
            supports_vision=True,
            supports_streaming=True,
            supports_reasoning=True,
            max_output_tokens=128_000,
        ),
    ),
}

def get_model_profile(
    provider: str,
    model_name: str,
) -> ModelProfile:
    key = (provider, model_name)

    try:
        return MODEL_PROFILES[key]
    except KeyError as error:
        raise ValueError(
            f"Model profile is not registered: {provider}/{model_name}"
        ) from error