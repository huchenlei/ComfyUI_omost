from .lib_omost.canvas import Canvas as OmostCanvas


class OmostLLMNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("OMOST_CANVAS",)
    FUNCTION = "run_llm"

    def run_llm(self, text: str) -> OmostCanvas:
        """Run LLM on text"""


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
    "OmostLLMNode": OmostLLMNode,
    "OmostLayoutCondNode": OmostLayoutCondNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmostLLMNode": "Omost LLM",
    "OmostLayoutCondNode": "Omost Layout Cond",
}
