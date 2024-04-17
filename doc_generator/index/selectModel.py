from enum import Enum
from typing import List, Dict, Optional

# Assuming encoding_for_model is some function you have that encodes strings
def encoding_for_model(model):
    # This function should encode the prompt based on the model type
    # Placeholder function; this needs to be implemented
    if model == LLMModels.GPT3:
        return {"encode": lambda p: len(p)}  # Just a placeholder
    elif model == LLMModels.GPT4:
        return {"encode": lambda p: len(p) + 1}  # Just a placeholder
    elif model == LLMModels.GPT432k:
        return {"encode": lambda p: len(p) + 2}  # Just a placeholder

class LLMModels(Enum):
    GPT3 = 1
    GPT4 = 2
    GPT432k = 3

class Priority(Enum):
    COST = 1
    PERFORMANCE = 2

class LLMModelDetails:
    def __init__(self, max_length):
        self.max_length = max_length

def getMaxPromptLength(prompts: List[str], model: LLMModels) -> int:
    encoding = encoding_for_model(model)
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

# Example usage
models = {
    LLMModels.GPT3: LLMModelDetails(max_length=1024),
    LLMModels.GPT4: LLMModelDetails(max_length=2048),
    LLMModels.GPT432k: LLMModelDetails(max_length=4096)
}

selected_model = select_model(["hello", "world"], [LLMModels.GPT3, LLMModels.GPT4], models, Priority.COST)
if selected_model:
    print(f"Selected model max length: {selected_model.max_length}")
else:
    print("No suitable model found")
