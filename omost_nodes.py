from __future__ import annotations
from typing import Literal, Tuple, TypedDict

import torch
from transformers import AutoModelForCausalLM

import comfy.model_management
from .lib_omost.canvas import Canvas as OmostCanvas


# Type definitions.
class OmostConversationItem(TypedDict):
    role: Literal["user", "assistant"]
    content: str


OmostConversation = list[OmostConversationItem]

# End of type definitions.


class OmostLLMLoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm_name": (
                    [
                        "lllyasviel/omost-phi-3-mini-128k-8bits",
                        "lllyasviel/omost-llama-3-8b-4bits",
                        "lllyasviel/omost-dolphin-2.9-llama3-8b-4bits",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("OMOST_LLM",)
    FUNCTION = "load_llm"

    def load_llm(self, llm_name: str) -> Tuple[AutoModelForCausalLM]:
        """Load LLM model"""
        HF_TOKEN = None
        dtype = (
            torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        )

        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=dtype,  # This is computation type, not load/memory type. The loading quant type is baked in config.
            token=HF_TOKEN,
            device_map="auto",  # This will load model to gpu with an offload system
        )

        return (llm_model,)


class OmostLLMChatNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm": ("OMOST_LLM",),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "conversation": ("OMOST_CONVERSATION",),
            },
        }

    RETURN_TYPES = (
        "OMOST_CONVERSATION",
        "OMOST_CANVAS",
    )
    FUNCTION = "run_llm"

    def run_llm(
        self,
        llm: AutoModelForCausalLM,
        text: str,
        conversation: OmostConversation,
    ) -> Tuple[OmostConversation, OmostCanvas]:
        """Run LLM on text"""

        new_conversation = conversation + [{"role": "user", "content": text}]


class OmostLayoutCondNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "canvas": ("OMOST_CANVAS",),
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "CONDITIONING",
    )
    FUNCTION = "layout_cond"

    def layout_cond(self, positive, negative, canvas: OmostCanvas):
        """Layout conditioning"""


NODE_CLASS_MAPPINGS = {
    "OmostLLMLoaderNode": OmostLLMLoaderNode,
    "OmostLLMChatNode": OmostLLMChatNode,
    "OmostLayoutCondNode": OmostLayoutCondNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmostLLMLoaderNode": "Omost LLM Loader",
    "OmostLLMChatNode": "Omost LLM Chat",
    "OmostLayoutCondNode": "Omost Layout Cond",
}
