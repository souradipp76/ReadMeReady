"""
Select Model
"""

from typing import Dict, List, Optional

import tiktoken

from readme_ready.types import LLMModelDetails, LLMModels, Priority


def get_max_prompt_length(prompts: List[str], model: LLMModels) -> int:
    """Get Max Prompt Length"""
    encoding = tiktoken.encoding_for_model(model)
    return max(len(encoding.encode(prompt)) for prompt in prompts)


def select_model(
    prompts: List[str],
    llms: List[LLMModels],
    models: Dict[LLMModels, LLMModelDetails],
    priority: Priority,
) -> Optional[LLMModelDetails]:
    """Select Model"""
    if priority == Priority.COST:
        for model_enum in [LLMModels.GPT3, LLMModels.GPT4, LLMModels.GPT432k]:
            if model_enum in llms and models[
                model_enum
            ].max_length > get_max_prompt_length(prompts, model_enum):
                return models[model_enum]
        return None
    elif priority == Priority.PERFORMANCE:
        for model_enum in [LLMModels.GPT4, LLMModels.GPT432k, LLMModels.GPT3]:
            if model_enum in llms:
                if models[model_enum].max_length > get_max_prompt_length(
                    prompts, model_enum
                ):
                    return models[model_enum]
        return None
    else:
        # return models[llms[0]]
        return None
