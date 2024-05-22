"""
LLM Utils
"""
import os

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          pipeline)

from doc_generator.types import LLMModelDetails, LLMModels


def get_gemma_chat_model(model_name: str, model_kwargs):
    """Get GEMMA Chat Model"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = get_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config
    )
    return HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=512
        ),
        model_kwargs=model_kwargs,
    )


def get_llama_chat_model(model_name: str, model_kwargs):
    """Get LLAMA2 Chat Model"""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        # use_exllama=False,
        # exllama_config={"version": 2}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config
    )
    return HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=512
        ),
        model_kwargs=model_kwargs,
    )


def get_openai_chat_model(
    model: str, temperature=None, streaming=None, model_kwargs=None
):
    """Get OpenAI Chat Model"""
    return ChatOpenAI(
        temperature=temperature,
        streaming=streaming,
        model=model,
        model_kwargs=model_kwargs,
    )


def get_openai_api_key():
    """Get OpenAI API Key"""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is not None:
        return api_key
    return ""


def get_tokenizer(model_name: str):
    """Get Tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


models = {
    LLMModels.GPT3: LLMModelDetails(
        name=LLMModels.GPT3,
        input_cost_per_1k_tokens=0.0015,
        output_cost_per_1k_tokens=0.002,
        max_length=3050,
        llm=ChatOpenAI(
            temperature=0.1,
            api_key=get_openai_api_key(),
            model=LLMModels.GPT3,
        ),
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.GPT4: LLMModelDetails(
        name=LLMModels.GPT4,
        input_cost_per_1k_tokens=0.03,
        output_cost_per_1k_tokens=0.06,
        max_length=8192,
        llm=ChatOpenAI(
            temperature=0.1,
            api_key=get_openai_api_key(),
            model=LLMModels.GPT4,
        ),
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.GPT432k: LLMModelDetails(
        name=LLMModels.GPT432k,
        input_cost_per_1k_tokens=0.06,
        output_cost_per_1k_tokens=0.12,
        max_length=32768,
        llm=ChatOpenAI(
            temperature=0.1,
            api_key=get_openai_api_key(),
            model=LLMModels.GPT4,
        ),
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    # LLMModels.LLAMA2_7B_CHAT_GPTQ: LLMModelDetails(
    #     name=LLMModels.LLAMA2_7B_CHAT_GPTQ,
    #     input_cost_per_1k_tokens=0,
    #     output_cost_per_1k_tokens=0,
    #     max_length=4096,
    #     llm=get_llama_chat_model(
    #         LLMModels.LLAMA2_7B_CHAT_GPTQ.value,
    #         model_kwargs={"temperature": 0}
    #     ),
    #     input_tokens=0,
    #     output_tokens=0,
    #     succeeded=0,
    #     failed=0,
    #     total=0,
    # ),
    # LLMModels.LLAMA2_13B_CHAT_GPTQ: LLMModelDetails(
    #     name=LLMModels.LLAMA2_13B_CHAT_GPTQ,
    #     input_cost_per_1k_tokens=0,
    #     output_cost_per_1k_tokens=0,
    #     max_length=4096,
    #     llm=get_llama_chat_model(
    #         LLMModels.LLAMA2_13B_CHAT_GPTQ.value,
    #         model_kwargs={"temperature": 0}
    #     ),
    #     input_tokens=0,
    #     output_tokens=0,
    #     succeeded=0,
    #     failed=0,
    #     total=0,
    # ),
    # LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ: LLMModelDetails(
    #     name=LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ,
    #     input_cost_per_1k_tokens=0,
    #     output_cost_per_1k_tokens=0,
    #     max_length=8192,
    #     llm=get_llama_chat_model(
    #         LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ.value,
    #         model_kwargs={"temperature": 0}
    #     ),
    #     input_tokens=0,
    #     output_tokens=0,
    #     succeeded=0,
    #     failed=0,
    #     total=0,
    # ),
    # LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ: LLMModelDetails(
    #     name=LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ,
    #     input_cost_per_1k_tokens=0,
    #     output_cost_per_1k_tokens=0,
    #     max_length=8192,
    #     llm=get_llama_chat_model(
    #         LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ.value,
    #         model_kwargs={"temperature": 0}
    #     ),
    #     input_tokens=0,
    #     output_tokens=0,
    #     succeeded=0,
    #     failed=0,
    #     total=0,
    # ),
    LLMModels.LLAMA2_7B_CHAT_HF: LLMModelDetails(
        name=LLMModels.LLAMA2_7B_CHAT_HF,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=4096,
        llm=get_llama_chat_model(
            LLMModels.LLAMA2_7B_CHAT_HF.value,
            model_kwargs={"temperature": 0}
        ),
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    # LLMModels.LLAMA2_13B_CHAT_GPTQ: LLMModelDetails(
    #     name=LLMModels.LLAMA2_13B_CHAT_GPTQ,
    #     input_cost_per_1k_tokens=0,
    #     output_cost_per_1k_tokens=0,
    #     max_length=4096,
    #     llm=get_llama_chat_model(
    #         LLMModels.LLAMA2_13B_CHAT_GPTQ.value,
    #         model_kwargs={"temperature": 0}
    #     ),
    #     input_tokens=0,
    #     output_tokens=0,
    #     succeeded=0,
    #     failed=0,
    #     total=0,
    # ),
    # LLMModels.CODELLAMA_7B_INSTRUCT_HF: LLMModelDetails(
    #     name=LLMModels.CODELLAMA_7B_INSTRUCT_HF,
    #     input_cost_per_1k_tokens=0,
    #     output_cost_per_1k_tokens=0,
    #     max_length=8192,
    #     llm=get_llama_chat_model(
    #         LLMModels.CODELLAMA_7B_INSTRUCT_HF.value,
    #         model_kwargs={"temperature": 0}
    #     ),
    #     input_tokens=0,
    #     output_tokens=0,
    #     succeeded=0,
    #     failed=0,
    #     total=0,
    # ),
    # LLMModels.CODELLAMA_13B_INSTRUCT_HF: LLMModelDetails(
    #     name=LLMModels.CODELLAMA_13B_INSTRUCT_HF,
    #     input_cost_per_1k_tokens=0,
    #     output_cost_per_1k_tokens=0,
    #     max_length=8192,
    #     llm=get_llama_chat_model(
    #         LLMModels.CODELLAMA_13B_INSTRUCT_HF.value,
    #         model_kwargs={"temperature": 0}
    #     ),
    #     input_tokens=0,
    #     output_tokens=0,
    #     succeeded=0,
    #     failed=0,
    #     total=0,
    # ),
    # LLMModels.GOOGLE_GEMMA_2B_INSTRUCT: LLMModelDetails(
    #     name=LLMModels.GOOGLE_GEMMA_2B_INSTRUCT,
    #     input_cost_per_1k_tokens=0,
    #     output_cost_per_1k_tokens=0,
    #     max_length=8192,
    #     llm=get_gemma_chat_model(
    #         LLMModels.GOOGLE_GEMMA_2B_INSTRUCT.value,
    #         model_kwargs={"temperature": 0}
    #     ),
    #     input_tokens=0,
    #     output_tokens=0,
    #     succeeded=0,
    #     failed=0,
    #     total=0,
    # ),
    # LLMModels.GOOGLE_GEMMA_7B_INSTRUCT: LLMModelDetails(
    #     name=LLMModels.GOOGLE_GEMMA_7B_INSTRUCT,
    #     input_cost_per_1k_tokens=0,
    #     output_cost_per_1k_tokens=0,
    #     max_length=8192,
    #     llm=get_gemma_chat_model(
    #         LLMModels.GOOGLE_GEMMA_7B_INSTRUCT.value,
    #         model_kwargs={"temperature": 0}
    #     ),
    #     input_tokens=0,
    #     output_tokens=0,
    #     succeeded=0,
    #     failed=0,
    #     total=0,
    # ),
}


def print_model_details(models):
    """Print Model Details"""
    output = []
    for model_details in models.values():
        result = {
            "Model": model_details.name,
            "File Count": model_details.total,
            "Succeeded": model_details.succeeded,
            "Failed": model_details.failed,
            "Tokens": model_details.inputTokens + model_details.output_tokens,
            "Cost": (
                (model_details.inputTokens / 1000)
                * model_details.input_cost_per_1k_tokens
                + (model_details.output_tokens / 1000)
                * model_details.output_cost_per_1k_tokens
            ),
        }
        output.append(result)

    totals = {
        "Model": "Total",
        "File Count": sum(item["File Count"] for item in output),
        "Succeeded": sum(item["Succeeded"] for item in output),
        "Failed": sum(item["Failed"] for item in output),
        "Tokens": sum(item["Tokens"] for item in output),
        "Cost": sum(item["Cost"] for item in output),
    }

    all_results = output + [totals]
    for item in all_results:
        print(item)


def total_index_cost_estimate(model):
    """Total Index Cost Estimate"""
    total_cost = sum(
        (model.input_tokens / 1000) * model.input_cost_per_1k_tokens
        + (model.output_tokens / 1000) * model.output_cost_per_1k_tokens
        for model in models.values()
    )
    return total_cost


def get_embeddings(model: str):
    """Get Embeddings"""
    if "llama" in model.lower() or "gemma" in model.lower():
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        return OpenAIEmbeddings()
