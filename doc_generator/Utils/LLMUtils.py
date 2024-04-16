import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

class LLMModels:
    GPT3 = 'GPT3'
    GPT4 = 'GPT4'
    GPT432k = 'GPT432k'

@dataclass
class LLMModelDetails:
    name: str
    inputCostPer1KTokens: float
    outputCostPer1KTokens: float
    maxLength: int
    llm: ChatOpenAI
    inputTokens: int = 0
    outputTokens: int = 0
    succeeded: int = 0
    failed: int = 0
    total: int = 0

models = {
    LLMModels.GPT3: LLMModelDetails(
        name=LLMModels.GPT3,
        inputCostPer1KTokens=0.0015,
        outputCostPer1KTokens=0.002,
        maxLength=3050,
        llm=ChatOpenAI(temperature=0.1, openai_api_key=os.getenv('OPENAI_API_KEY'), model_name=LLMModels.GPT3)
    ),
    LLMModels.GPT4: LLMModelDetails(
        name=LLMModels.GPT4,
        inputCostPer1KTokens=0.03,
        outputCostPer1KTokens=0.06,
        maxLength=8192,
        llm=ChatOpenAI(temperature=0.1, openai_api_key=os.getenv('OPENAI_API_KEY'), model_name=LLMModels.GPT4)
    ),
    LLMModels.GPT432k: LLMModelDetails(
        name=LLMModels.GPT432k,
        inputCostPer1KTokens=0.06,
        outputCostPer1KTokens=0.12,
        maxLength=32768,
        llm=ChatOpenAI(temperature=0.1, openai_api_key=os.getenv('OPENAI_API_KEY'), model_name=LLMModels.GPT4)
    )
}

def print_model_details(models):
    output = []
    for model_details in models.values():
        result = {
            'Model': model_details.name,
            'File Count': model_details.total,
            'Succeeded': model_details.succeeded,
            'Failed': model_details.failed,
            'Tokens': model_details.inputTokens + model_details.outputTokens,
            'Cost': ((model_details.inputTokens / 1000) * model_details.inputCostPer1KTokens +
                     (model_details.outputTokens / 1000) * model_details.outputCostPer1KTokens)
        }
        output.append(result)

    totals = {
        'Model': 'Total',
        'File Count': sum(item['File Count'] for item in output),
        'Succeeded': sum(item['Succeeded'] for item in output),
        'Failed': sum(item['Failed'] for item in output),
        'Tokens': sum(item['Tokens'] for item in output),
        'Cost': sum(item['Cost'] for item in output)
    }

    all_results = output + [totals]
    for item in all_results:
        print(item)

def total_index_cost_estimate(models):
    total_cost = sum(
        (model.inputTokens / 1000) * model.inputCostPer1KTokens +
        (model.outputTokens / 1000) * model.outputCostPer1KTokens
        for model in models.values()
    )
    return total_cost
