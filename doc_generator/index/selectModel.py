from typing import List, Dict, Optional
from doc_generator.types import LLMModels, LLMModelDetails, Priority

import tiktoken

def getMaxPromptLength(prompts: List[str], model: LLMModels) -> int:
    encoding = tiktoken.get_encoding(model)
    return max(encoding['encode'](prompt) for prompt in prompts)

def select_model(prompts: List[str], llms: List[LLMModels], models: Dict[LLMModels, LLMModelDetails], priority: Priority) -> Optional[LLMModelDetails]:
    if priority == Priority.COST:
        for model_enum in [LLMModels.GPT3, LLMModels.GPT4, LLMModels.GPT432k]:
            if model_enum in llms and models[model_enum].max_length > getMaxPromptLength(prompts, model_enum):
                return models[model_enum]
        return None
    else:
        for model_enum in [LLMModels.GPT4, LLMModels.GPT432k, LLMModels.GPT3]:
            if model_enum in llms:
                if models[model_enum].max_length > getMaxPromptLength(prompts, model_enum):
                    return models[model_enum]
        return None
