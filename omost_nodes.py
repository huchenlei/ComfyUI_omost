from __future__ import annotations
from typing import Literal, Tuple, TypedDict, NamedTuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import comfy.model_management
from .lib_omost.canvas import (
    Canvas as OmostCanvas,
    OmostCanvasOutput,
    OmostCanvasCondition,
)
from .lib_omost.utils import numpy2pytorch


# Type definitions.
class OmostConversationItem(TypedDict):
    role: Literal["user", "assistant"]
    content: str


OmostConversation = list[OmostConversationItem]


class OmostLLM(NamedTuple):
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


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
                    {
                        "default": "lllyasviel/omost-llama-3-8b-4bits",
                    },
                ),
            }
        }

    RETURN_TYPES = ("OMOST_LLM",)
    FUNCTION = "load_llm"

    def load_llm(self, llm_name: str) -> Tuple[OmostLLM]:
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
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, token=HF_TOKEN)

        return (OmostLLM(llm_model, llm_tokenizer),)


class OmostLLMChatNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm": ("OMOST_LLM",),
                "text": ("STRING", {"multiline": True}),
                "max_new_tokens": (
                    "INT",
                    {"min": 128, "max": 4096, "step": 1, "default": 4096},
                ),
                "top_p": (
                    "FLOAT",
                    {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.9},
                ),
                "temperature": (
                    "FLOAT",
                    {"min": 0.0, "max": 2.0, "step": 0.01, "default": 0.6},
                ),
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
        llm: OmostLLM,
        text: str,
        max_new_tokens: int,
        top_p: float,
        temperature: float,
        conversation: OmostConversation | None = None,
    ) -> Tuple[OmostConversation, OmostCanvas]:
        """Run LLM on text"""
        llm_tokenizer: AutoTokenizer = llm.tokenizer
        llm_model: AutoModelForCausalLM = llm.model

        conversation = conversation or []  # Default to empty list
        new_conversation = conversation + [{"role": "user", "content": text}]

        input_ids: torch.Tensor = llm_tokenizer.apply_chat_template(
            new_conversation, return_tensors="pt", add_generation_prompt=True
        ).to(llm_model.device)

        output_ids = llm_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature != 0,
        )

        generated_text: str = llm_tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            skip_prompt=True,
            timeout=10,
        )

        final_conversation = new_conversation + [
            {"role": "assistant", "content": generated_text}
        ]

        return (final_conversation, OmostCanvas.from_bot_response(generated_text))


class OmostCanvasRenderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "canvas": ("OMOST_CANVAS",),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "OMOST_CANVAS_CONDITIONING",
    )
    FUNCTION = "render_canvas"

    def render_canvas(
        self, canvas: OmostCanvas
    ) -> Tuple[torch.Tensor, list[OmostCanvasCondition]]:
        """Render canvas"""
        canvas_output: OmostCanvasOutput = canvas.process()
        return (
            numpy2pytorch(canvas_output["initial_latent"]),
            canvas_output["bag_of_conditions"],
        )


class OmostLayoutCondNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "canvas_conds": ("OMOST_CANVAS_CONDITIONING",),
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "CONDITIONING",
    )
    FUNCTION = "layout_cond"

    def layout_cond(self, positive, negative, canvas_conds: list[OmostCanvasCondition]):
        """Layout conditioning"""


NODE_CLASS_MAPPINGS = {
    "OmostLLMLoaderNode": OmostLLMLoaderNode,
    "OmostLLMChatNode": OmostLLMChatNode,
    "OmostCanvasRenderNode": OmostCanvasRenderNode,
    "OmostLayoutCondNode": OmostLayoutCondNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmostLLMLoaderNode": "Omost LLM Loader",
    "OmostLLMChatNode": "Omost LLM Chat",
    "OmostCanvasRenderNode": "Omost Canvas Render",
    "OmostLayoutCondNode": "Omost Layout Cond",
}
