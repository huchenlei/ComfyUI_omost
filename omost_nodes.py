from __future__ import annotations
from enum import Enum
import json
from typing import Literal, Tuple, TypedDict, NamedTuple
import sys
import os
import logging
from typing_extensions import NotRequired

import requests
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import comfy.model_management
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from nodes import CLIPTextEncode, ConditioningSetMask
from .lib_omost.canvas import (
    Canvas as OmostCanvas,
    OmostCanvasCondition,
    system_prompt,
)
from .lib_omost.utils import numpy2pytorch, scoped_numpy_random, scoped_torch_random
from .lib_omost.greedy_encode import (
    encode_bag_of_subprompts_greedy,
    CLIPTokens,
    EncoderOutput,
    SPECIAL_TOKENS,
)


def create_logger(level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger


logger = create_logger(level=logging.INFO)

# Canvas size used in original Omost repo.
CANVAS_SIZE = 90


# Type definitions.
class OmostConversationItem(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


OmostConversation = list[OmostConversationItem]


class OmostLLM(NamedTuple):
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


class OmostLLMServer(NamedTuple):
    client: OpenAI
    model_id: str


ComfyUIConditioning = list  # Dummy type definitions for ComfyUI
ComfyCLIPTokensWithWeight = list[Tuple[int, float]]


class ComfyCLIPTokens(TypedDict):
    l: list[ComfyCLIPTokensWithWeight]
    g: NotRequired[list[ComfyCLIPTokensWithWeight]]


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
    CATEGORY = "omost"

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
            trust_remote_code=True,
        )
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, token=HF_TOKEN)

        return (OmostLLM(llm_model, llm_tokenizer),)


class OmostLLMHTTPServerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "address": ("STRING", {"multiline": True}),
                "api_type":(
                    [
                        "OpenAI",
                        "TGI"
                    ],
                    {
                        "default": "OpenAI"
                    }
                )
            }
        }

    RETURN_TYPES = ("OMOST_LLM",)
    FUNCTION = "init_client"
    CATEGORY = "omost"

    def init_client(self, address: str, api_type: str) -> Tuple[OmostLLMServer]:
        """Initialize LLM client with HTTP server address."""

        if api_type == "OpenAI":
            if address.endswith("v1"):
                server_address = address
            else:
                server_address = os.path.join(address, "v1")
        
            model_id = ""

        elif api_type == "TGI":
            if address.endswith("v1"):
                server_address = address
                server_info_url = address.replace("v1", "info")
            else:
                server_address = os.path.join(address, "v1")
                server_info_url = os.path.join(address, "info")
            #Get model_id from server info
            server_info = requests.get(server_info_url, timeout=5).json()
            model_id = server_info["model_id"]

        client = OpenAI(base_url=server_address, api_key="_")

        return (OmostLLMServer(client, model_id),)


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
                # Note: ComfyUI's front-end code randomizes the seed to 64-bit int.
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "conversation": ("OMOST_CONVERSATION",),
            },
        }

    RETURN_TYPES = (
        "OMOST_CONVERSATION",
        "OMOST_CANVAS_CONDITIONING",
    )
    FUNCTION = "run_llm"
    CATEGORY = "omost"

    def prepare_conversation(
        self, text: str, conversation: OmostConversation | None = None
    ) -> Tuple[OmostConversation, OmostConversation, OmostConversationItem]:
        conversation = conversation or []  # Default to empty list
        system_conversation_item: OmostConversationItem = {
            "role": "system",
            "content": system_prompt,
        }
        user_conversation_item: OmostConversationItem = {
            "role": "user",
            "content": text,
        }
        input_conversation: list[OmostConversationItem] = [
            system_conversation_item,
            *conversation,
            user_conversation_item,
        ]
        return conversation, input_conversation, user_conversation_item

    def run_local_llm(
        self,
        llm: OmostLLM,
        input_conversation: list[OmostConversationItem],
        max_new_tokens: int,
        top_p: float,
        temperature: float,
        seed: int,
    ) -> str:
        with scoped_torch_random(seed), scoped_numpy_random(seed):
            llm_tokenizer: AutoTokenizer = llm.tokenizer
            llm_model: AutoModelForCausalLM = llm.model

            input_ids: torch.Tensor = llm_tokenizer.apply_chat_template(
                input_conversation, return_tensors="pt", add_generation_prompt=True
            ).to(llm_model.device)
            input_length = input_ids.shape[1]

            output_ids: torch.Tensor = llm_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature != 0,
            )
            generated_ids = output_ids[:, input_length:]
            generated_text: str = llm_tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                skip_prompt=True,
                timeout=10,
            )
            return generated_text

    def run_llm(
        self,
        llm: OmostLLM | OmostLLMServer,
        text: str,
        max_new_tokens: int,
        top_p: float,
        temperature: float,
        seed: int,
        conversation: OmostConversation | None = None,
    ) -> Tuple[OmostConversation, OmostCanvas]:
        """Run LLM on text"""
        if seed > 0xFFFFFFFF:
            seed = seed & 0xFFFFFFFF
            logger.warning("Seed is too large. Truncating to 32-bit: %d", seed)

        conversation, input_conversation, user_conversation_item = (
            self.prepare_conversation(text, conversation)
        )

        if isinstance(llm, OmostLLM):
            generated_text = self.run_local_llm(
                llm, input_conversation, max_new_tokens, top_p, temperature, seed
            )
        else:
            generated_text = (
                llm.client.chat.completions.create(
                    model=llm.model_id,
                    messages=input_conversation,
                    top_p=top_p,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    seed=seed,
                )
                .choices[0]
                .message.content
            )

        output_conversation = [
            *conversation,
            user_conversation_item,
            {"role": "assistant", "content": generated_text},
        ]
        return (
            output_conversation,
            OmostCanvas.from_bot_response(generated_text).process(),
        )


class OmostRenderCanvasConditioningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "canvas_conds": ("OMOST_CANVAS_CONDITIONING",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "render_canvas"
    CATEGORY = "omost"

    def render_canvas(
        self, canvas_conds: list[OmostCanvasCondition]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render canvas conditioning to image"""
        return (
            numpy2pytorch(imgs=[OmostCanvas.render_initial_latent(canvas_conds)]),
            torch.cat(
                [OmostCanvas.render_mask(cond).unsqueeze(0) for cond in canvas_conds],
                dim=0,
            ),
        )


class PromptEncoding:
    """Namespace for different prompt encoding methods"""

    ENCODE_NODE = CLIPTextEncode()

    @staticmethod
    def encode_bag_of_subprompts(
        clip: CLIP, prefixes: list[str], suffixes: list[str]
    ) -> ComfyUIConditioning:
        """@Deprecated
        Simplified way to encode bag of subprompts without omost's greedy approach.
        """
        conds: ComfyUIConditioning = []

        logger.debug("Start encoding bag of subprompts")
        for target in suffixes:
            complete_prompt = "".join(prefixes + [target])
            logger.debug(f"Encoding prompt: {complete_prompt}")
            cond: ComfyUIConditioning = PromptEncoding.ENCODE_NODE.encode(
                clip, complete_prompt
            )[0]
            assert len(cond) == 1
            conds.extend(cond)

        logger.debug("End encoding bag of subprompts. Total conditions: %d", len(conds))

        # Concat all conditions
        return [
            [
                # cond
                torch.cat([cond for cond, _ in conds], dim=1),
                # extra_dict
                {"pooled_output": conds[0][1]["pooled_output"]},
            ]
        ]

    @staticmethod
    def encode_subprompts(
        clip: CLIP, prefixes: list[str], suffixes: list[str]
    ) -> ComfyUIConditioning:
        """@Deprecated
        Simplified way to encode subprompts by joining them together. This is
        more direct without re-organizing the prompts into optimal batches like
        with the greedy approach.
        Note: This function has the issue of semantic truncation.
        """
        complete_prompt = ",".join(
            ["".join(prefixes + [target]) for target in suffixes]
        )
        logger.debug("Encoding prompt: %s", complete_prompt)
        return PromptEncoding.ENCODE_NODE.encode(clip, complete_prompt)[0]

    @staticmethod
    def encode_bag_of_subprompts_greedy(
        clip: CLIP, prefixes: list[str], suffixes: list[str]
    ) -> ComfyUIConditioning:
        """Encode bag of subprompts with greedy approach. This approach is used
        by the original Omost repo."""

        def convert_comfy_tokens(
            comfy_tokens: list[ComfyCLIPTokensWithWeight],
        ) -> list[int]:
            assert len(comfy_tokens) >= 1
            tokens: list[int] = [token for token, _ in comfy_tokens[0]]
            # Strip the first token which is the CLIP prefix.
            # Strip padding tokens.
            return tokens[1 : tokens.index(SPECIAL_TOKENS["end"])]

        def convert_to_comfy_tokens(tokens: CLIPTokens) -> ComfyCLIPTokens:
            return {
                "l": [[(token, 1.0) for token in tokens.clip_l_tokens]],
                "g": (
                    [[(token, 1.0) for token in tokens.clip_g_tokens]]
                    if tokens.clip_g_tokens is not None
                    else None
                ),
            }

        def tokenize(text: str) -> CLIPTokens:
            tokens: ComfyCLIPTokens = clip.tokenize(text)
            return CLIPTokens(
                clip_l_tokens=convert_comfy_tokens(tokens["l"]),
                clip_g_tokens=(
                    convert_comfy_tokens(tokens.get("g")) if "g" in tokens else None
                ),
            )

        def encode(tokens: CLIPTokens) -> EncoderOutput:
            cond, pooled = clip.encode_from_tokens(
                convert_to_comfy_tokens(tokens), return_pooled=True
            )
            return EncoderOutput(cond=cond, pooler=pooled)

        encoder_output = encode_bag_of_subprompts_greedy(
            prefixes,
            suffixes,
            tokenize_func=tokenize,
            encode_func=encode,
            logger=logger,
        )

        return [
            [
                encoder_output.cond,
                {"pooled_output": encoder_output.pooler},
            ]
        ]


class OmostDenseDiffusionLayoutNode:
    """Apply Omost layout with Omost's area condition system. This is the regional
    prompt system implemented in the original Omost repo.

    You need to install https://github.com/huchenlei/ComfyUI_densediffusion to use this node.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "canvas_conds": ("OMOST_CANVAS_CONDITIONING",),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "layout_cond"
    CATEGORY = "omost"

    def __init__(self):
        try:
            from custom_nodes.ComfyUI_densediffusion.densediffusion_node import (
                DenseDiffusionApplyNode,
                DenseDiffusionAddCondNode,
            )
        except Exception as e:
            logger.error(
                "Failed to import ComfyUI_densediffusion. Make sure it's installed."
                "https://github.com/huchenlei/ComfyUI_densediffusion"
            )
            raise e

        self.dense_diffusion_apply_node = DenseDiffusionApplyNode()
        self.dense_diffusion_add_cond_node = DenseDiffusionAddCondNode()

    def layout_cond(
        self,
        model: ModelPatcher,
        canvas_conds: list[OmostCanvasCondition],
        clip: CLIP,
    ) -> tuple[ModelPatcher, ComfyUIConditioning]:
        """Layout conditioning"""
        work_model: ModelPatcher = model.clone()

        for canvas_cond in canvas_conds:
            cond: ComfyUIConditioning = PromptEncoding.encode_bag_of_subprompts_greedy(
                clip, canvas_cond["prefixes"], canvas_cond["suffixes"]
            )
            # Set area cond
            work_model = self.dense_diffusion_add_cond_node.append(
                work_model,
                conditioning=cond,
                mask=OmostCanvas.render_mask(canvas_cond),
                strength=1.0,
            )[0]

        return self.dense_diffusion_apply_node.apply(work_model)


class OmostGreedyBagsTextEmbeddingNode:
    """Just encode the omost canvas conditions with greedy bags approach.
    Ignoring region conditions."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "canvas_conds": ("OMOST_CANVAS_CONDITIONING",),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "layout_cond"
    CATEGORY = "omost"

    def layout_cond(
        self,
        canvas_conds: list[OmostCanvasCondition],
        clip: CLIP,
    ) -> tuple[ComfyUIConditioning]:
        conds: ComfyUIConditioning = [
            PromptEncoding.encode_bag_of_subprompts_greedy(
                clip, canvas_cond["prefixes"], canvas_cond["suffixes"]
            )[0]
            for canvas_cond in canvas_conds
        ]
        assert len(conds) > 0

        return ([
            [
                # cond
                torch.cat([cond[0] for cond in conds], dim=1),
                # pooled_output
                {"pooled_output": conds[0][1]["pooled_output"]},
            ]
        ],)


class OmostComfyLayoutNode:
    """Apply Omost layout with ComfyUI's area condition system."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "canvas_conds": ("OMOST_CANVAS_CONDITIONING",),
                "clip": ("CLIP",),
                "global_strength": (
                    "FLOAT",
                    {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.2},
                ),
                "region_strength": (
                    "FLOAT",
                    {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.8},
                ),
                "overlap_method": (
                    [e.value for e in OmostComfyLayoutNode.AreaOverlapMethod],
                    {"default": OmostComfyLayoutNode.AreaOverlapMethod.AVERAGE.value},
                ),
            },
            "optional": {
                "positive": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "MASK")
    FUNCTION = "layout_cond"
    CATEGORY = "omost"

    class AreaOverlapMethod(Enum):
        """Methods to handle overlapping areas."""

        # The top layer overwrites the bottom layer.
        OVERLAY = "overlay"
        # Take the average of the two layers.
        AVERAGE = "average"

    def __init__(self):
        self.cond_set_mask_node = ConditioningSetMask()

    @staticmethod
    def calc_cond_mask(
        canvas_conds: list[OmostCanvasCondition],
        method: AreaOverlapMethod = AreaOverlapMethod.OVERLAY,
    ) -> list[OmostCanvasCondition]:
        """Calculate canvas cond mask."""
        assert len(canvas_conds) > 0
        canvas_conds = canvas_conds.copy()

        global_cond = canvas_conds[0]
        global_cond["mask"] = torch.ones(
            [CANVAS_SIZE, CANVAS_SIZE], dtype=torch.float32
        )
        region_conds = canvas_conds[1:]

        canvas_state = torch.zeros([CANVAS_SIZE, CANVAS_SIZE], dtype=torch.float32)
        if method == OmostComfyLayoutNode.AreaOverlapMethod.OVERLAY:
            for canvas_cond in region_conds[::-1]:
                a, b, c, d = canvas_cond["rect"]
                mask = torch.zeros([CANVAS_SIZE, CANVAS_SIZE], dtype=torch.float32)
                mask[a:b, c:d] = 1.0
                mask = mask * (1 - canvas_state)
                canvas_state += mask
                canvas_cond["mask"] = mask
        elif method == OmostComfyLayoutNode.AreaOverlapMethod.AVERAGE:
            canvas_state += 1e-6  # Avoid division by zero
            for canvas_cond in region_conds:
                a, b, c, d = canvas_cond["rect"]
                canvas_state[a:b, c:d] += 1.0

            for canvas_cond in region_conds:
                a, b, c, d = canvas_cond["rect"]
                mask = torch.zeros([CANVAS_SIZE, CANVAS_SIZE], dtype=torch.float32)
                mask[a:b, c:d] = 1.0
                mask = mask / canvas_state
                canvas_cond["mask"] = mask

        return canvas_conds

    def layout_cond(
        self,
        canvas_conds: list[OmostCanvasCondition],
        clip: CLIP,
        global_strength: float,
        region_strength: float,
        overlap_method: str,
        positive: ComfyUIConditioning | None = None,
    ):
        """Layout conditioning"""
        overlap_method = OmostComfyLayoutNode.AreaOverlapMethod(overlap_method)
        positive: ComfyUIConditioning = positive or []
        positive = positive.copy()
        masks: list[torch.Tensor] = []
        canvas_conds = OmostComfyLayoutNode.calc_cond_mask(
            canvas_conds, method=overlap_method
        )

        for i, canvas_cond in enumerate(canvas_conds):
            is_global = i == 0

            prefixes = canvas_cond["prefixes"]
            # Skip the global prefix for region prompts.
            if not is_global:
                prefixes = prefixes[1:]

            cond: ComfyUIConditioning = PromptEncoding.encode_bag_of_subprompts_greedy(
                clip, prefixes, canvas_cond["suffixes"]
            )
            # Set area cond
            cond: ComfyUIConditioning = self.cond_set_mask_node.append(
                cond,
                mask=canvas_cond["mask"],
                set_cond_area="default",
                strength=global_strength if is_global else region_strength,
            )[0]
            assert len(cond) == 1
            positive.extend(cond)
            masks.append(canvas_cond["mask"].unsqueeze(0))

        return (
            positive,
            # Output masks in case it's needed for debugging or the user might
            # want to apply extra condition such as ControlNet/IPAdapter to
            # specified region.
            torch.cat(masks, dim=0),
        )


class OmostLoadCanvasConditioningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_str": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("OMOST_CANVAS_CONDITIONING",)
    FUNCTION = "load_canvas"
    CATEGORY = "omost"

    def load_canvas(self, json_str: str) -> Tuple[list[OmostCanvasCondition]]:
        """Load canvas from file"""
        return (json.loads(json_str),)


class OmostLoadCanvasPythonCodeNode:
    """Load python code generated by Omost demo app."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "python_str": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("OMOST_CANVAS_CONDITIONING",)
    FUNCTION = "load_canvas"
    CATEGORY = "omost"

    def load_canvas(self, python_str: str) -> Tuple[list[OmostCanvasCondition]]:
        """Load canvas from file"""
        canvas = OmostCanvas.from_python_code(python_str)
        return (canvas.process(),)


NODE_CLASS_MAPPINGS = {
    "OmostLLMLoaderNode": OmostLLMLoaderNode,
    "OmostLLMHTTPServerNode": OmostLLMHTTPServerNode,
    "OmostLLMChatNode": OmostLLMChatNode,
    "OmostGreedyBagsTextEmbeddingNode": OmostGreedyBagsTextEmbeddingNode,
    "OmostLayoutCondNode": OmostComfyLayoutNode,
    "OmostDenseDiffusionLayoutNode": OmostDenseDiffusionLayoutNode,
    "OmostLoadCanvasConditioningNode": OmostLoadCanvasConditioningNode,
    "OmostLoadCanvasPythonCodeNode": OmostLoadCanvasPythonCodeNode,
    "OmostRenderCanvasConditioningNode": OmostRenderCanvasConditioningNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmostLLMLoaderNode": "Omost LLM Loader",
    "OmostLLMHTTPServerNode": "Omost LLM HTTP Server",
    "OmostLLMChatNode": "Omost LLM Chat",
    "OmostGreedyBagsTextEmbeddingNode": "Omost Greedy Bags Text Embedding",
    "OmostLayoutCondNode": "Omost Layout Cond (ComfyUI-Area)",
    "OmostDenseDiffusionLayoutNode": "Omost Layout Cond (OmostDenseDiffusion)",
    "OmostLoadCanvasConditioningNode": "Omost Load Canvas Conditioning",
    "OmostLoadCanvasPythonCodeNode": "Omost Load Canvas Python Code",
    "OmostRenderCanvasConditioningNode": "Omost Render Canvas Conditioning",
}
