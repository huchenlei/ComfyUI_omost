# ComfyUI_omost

ComfyUI implementation of [Omost](https://github.com/lllyasviel/Omost).

## WIP
The node structure in this repo are subject to change in recent development.

## How to use

As you can see from the screenshot, there are 2 parts of omost:
- LLM Chat
- Region Condition

### LLM Chat
LLM Chat allows user interact with LLM to obtain a JSON-like structure. There are 3 nodes in this pack to interact with the Omost LLM:
- `Omost LLM Loader`: Load a LLM
- `Omost LLM Chat`: Chat with LLM to obtain JSON layout prompt
- `Omost Load Canvas Conditioning`: Load the JSON layout prompt previously saved

Optionally you can use the show-anything node to display the json text and save it for later.
The official LLM's method runs slow. Each chat takes about 3~5min on 4090. (But now we can use TGI to deploy accelerated inference. For details, refer [**Accelerating LLM**](#accelerating-llm).)

Examples:
- Simple LLM Chat: ![image](https://github.com/huchenlei/ComfyUI_omost/assets/20929282/896eb810-6137-4682-8236-67cfefdbae99)
- Multi-round LLM Chat: ![image](https://github.com/huchenlei/ComfyUI_omost/assets/20929282/fada801a-0116-4b39-8334-b62664dbf153)


Here is the a sample JSON output used in the examples:
<details>
  <summary>Click to expand JSON</summary>

```json
[
    {
        "rect": [
            0,
            90,
            0,
            90
        ],
        "prefixes": [
            "An Asian girl sitting on a chair."
        ],
        "suffixes": [
            "The image depicts an Asian girl sitting gracefully on a chair.",
            "She has long, flowing black hair and is wearing a traditional Korean dress, known as a hanbok, which is adorned with intricate floral patterns.",
            "Her posture is relaxed yet elegant, with one hand gently resting on her knee and the other hand holding a delicate fan.",
            "The background is a simple, neutral-colored room with soft, natural light filtering in from a window.",
            "The overall atmosphere is serene and contemplative, capturing a moment of quiet reflection.",
            "Asian girl, sitting, chair, traditional dress, hanbok, floral patterns, long black hair, elegant posture, delicate fan, neutral background, natural light, serene atmosphere, contemplative, quiet reflection, simple room, graceful, intricate patterns, flowing hair, cultural attire, traditional Korean dress, relaxed posture."
        ],
        "color": [
            211,
            211,
            211
        ]
    },
    {
        "color": [
            173,
            216,
            230
        ],
        "rect": [
            5,
            45,
            0,
            55
        ],
        "prefixes": [
            "An Asian girl sitting on a chair.",
            "Window."
        ],
        "suffixes": [
            "The window is a simple, rectangular frame with clear glass panes.",
            "It allows natural light to filter into the room, casting soft, diffused light over the scene.",
            "The window is partially open, with a gentle breeze creating a soft, flowing motion in the curtains.",
            "The view outside is blurred, suggesting a peaceful outdoor setting.",
            "The window adds a sense of openness and connection to the outside world, enhancing the serene and contemplative atmosphere of the image.",
            "window, rectangular frame, clear glass panes, natural light, soft light, diffused light, partially open window, gentle breeze, flowing curtains, blurred view, peaceful outdoor setting, sense of openness, connection to outside, serene atmosphere, contemplative.",
            "The window adds a sense of openness and connection to the outside world.",
            "The style is simple and natural, with a focus on soft light and gentle breeze.",
            "High-quality image with detailed textures and natural lighting."
        ]
    },
    {
        "color": [
            139,
            69,
            19
        ],
        "rect": [
            25,
            85,
            5,
            45
        ],
        "prefixes": [
            "An Asian girl sitting on a chair.",
            "Chair."
        ],
        "suffixes": [
            "The chair on which the girl is sitting is a simple, elegant wooden chair.",
            "It has a smooth, polished finish and a classic design with curved legs and a high backrest.",
            "The chair's wood is a rich, dark brown, adding a touch of warmth to the overall scene.",
            "The girl sits gracefully on the chair, her posture relaxed yet elegant.",
            "The chair complements her traditional Korean dress, enhancing the cultural and elegant atmosphere of the image.",
            "chair, wooden chair, elegant design, curved legs, high backrest, polished finish, dark brown wood, warm touch, traditional Korean dress, cultural attire, elegant posture, graceful sitting, classic design, simple chair, rich wood, polished finish.",
            "The chair adds a touch of warmth and elegance to the overall scene.",
            "The style is classic and simple, with a focus on elegant design and polished finish.",
            "High-quality image with detailed textures and natural lighting."
        ]
    },
    {
        "color": [
            245,
            245,
            220
        ],
        "rect": [
            40,
            90,
            40,
            90
        ],
        "prefixes": [
            "An Asian girl sitting on a chair.",
            "Delicate fan."
        ],
        "suffixes": [
            "The delicate fan held by the girl is a traditional accessory, crafted from fine bamboo with intricate carvings.",
            "The fan is adorned with delicate floral designs, adding to its beauty and cultural significance.",
            "The girl holds the fan gently, its soft movements enhancing the graceful and elegant atmosphere of the image.",
            "The fan is a symbol of refinement and tradition, adding a touch of cultural elegance to the overall scene.",
            "delicate fan, traditional accessory, fine bamboo, intricate carvings, floral designs, cultural significance, graceful holding, soft movements, elegant atmosphere, symbol of refinement, cultural elegance, intricate carvings, delicate floral designs, traditional accessory, fine craftsmanship.",
            "The delicate fan adds a touch of cultural elegance and refinement to the scene.",
            "The style is traditional and refined, with a focus on intricate carvings and delicate designs.",
            "High-quality image with detailed textures and natural lighting."
        ]
    },
    {
        "color": [
            255,
            255,
            240
        ],
        "rect": [
            15,
            75,
            15,
            75
        ],
        "prefixes": [
            "An Asian girl sitting on a chair.",
            "Asian girl."
        ],
        "suffixes": [
            "The Asian girl is the focal point of the image.",
            "She is dressed in a traditional Korean hanbok, which is a beautiful garment made from silk and adorned with intricate floral patterns.",
            "Her black hair is long and flowing, cascading down her back in soft waves.",
            "Her expression is calm and thoughtful, with a slight smile playing on her lips.",
            "She sits gracefully on the chair, her posture relaxed yet elegant.",
            "One hand rests gently on her knee, while the other hand holds a delicate fan, adding a touch of grace to her appearance.",
            "Asian girl, focal point, traditional Korean dress, hanbok, intricate floral patterns, long black hair, flowing hair, calm expression, thoughtful, slight smile, graceful posture, relaxed, elegant, delicate fan, cultural attire.",
            "The atmosphere is serene and contemplative, capturing a moment of quiet reflection.",
            "The style is elegant and traditional, with a focus on cultural attire and graceful posture.",
            "High-quality image with detailed textures and natural lighting."
        ]
    },
    {
        "color": [
            218,
            165,
            32
        ],
        "rect": [
            5,
            65,
            45,
            85
        ],
        "prefixes": [
            "An Asian girl sitting on a chair.",
            "Traditional Korean dress."
        ],
        "suffixes": [
            "The traditional Korean dress, known as a hanbok, is a beautiful garment made from silk.",
            "It is adorned with intricate floral patterns in vibrant colors, including reds, blues, and yellows.",
            "The dress is designed to flow gracefully, with delicate folds and soft movements.",
            "The girl wears the dress with pride, its cultural significance evident in its elegant design and intricate details.",
            "The hanbok complements her graceful posture and adds a touch of cultural elegance to the overall scene.",
            "traditional Korean dress, hanbok, beautiful garment, silk fabric, intricate floral patterns, vibrant colors, reds, blues, yellows, graceful flow, delicate folds, soft movements, cultural significance, elegant design, intricate details, graceful posture, cultural elegance.",
            "The hanbok adds a touch of cultural elegance and intricate beauty to the scene.",
            "The style is traditional and elegant, with a focus on intricate floral patterns and vibrant colors.",
            "High-quality image with detailed textures and natural lighting."
        ]
    }
]
```
</details>

### Region condition
Region condition part converts the JSON condition to ComfyUI's area format. Under the hood, it is calling `ConditioningSetMask` node to set non-overlap area for each cond.
According to https://github.com/lllyasviel/Omost#regional-prompter, original Omost repo is using method 3, while ComfyUI's built-in method is method 2. So expect there to be some
difference on results. I will implement ComfyUI version of [densediffusion](https://github.com/naver-ai/DenseDiffusion) soon.

There are 2 overlap methods:
- Overlay: The layer on top completely overwrites layer below
- Average: The overlapped area is the average of all conditions
![image](https://github.com/huchenlei/ComfyUI_omost/assets/20929282/e7d007e4-1175-4435-adf4-a9211937d8c1)

Optionally you can also pass the image generated from Omost canvas as initial latent as described in the original Omost repo:
![image](https://github.com/huchenlei/ComfyUI_omost/assets/20929282/f913d141-9045-41fa-998f-770a840adc69)

### Compose with other control methods
You can freely compose the region condition with other control methods like ControlNet/IPAdapter. Following workflow applies an ipadapter model to the character region by selecting the corresponding mask.
![image](https://github.com/huchenlei/ComfyUI_omost/assets/20929282/191a5ea1-776a-42da-89ee-fd17a3a08eae)

### Accelerating LLM

Now you can leverage [TGI](https://huggingface.co/docs/text-generation-inference) to deploy LLM services and achieve up to 6x faster inference speeds. If you need long-term support for your work, this method is highly recommended to save you a lot of time.

**Preparation**: You will need an additional 20GB of VRAM to deploy an 8B LLM (trading space for time).

**First**, you can easily start the service using Docker with the following steps:
```
port=8080
modelID=lllyasviel/omost-llama-3-8b
memoryRate=0.9 # Normal operation requires 20GB of VRAM, adjust the ratio according to the VRAM of the deployment machine
volume=$HOME/.cache/huggingface/hub # Model cache files

docker run --gpus all -p $port:80 \
    -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:2.0.4 \
    --model-id $modelID --max-total-tokens 9216 --cuda-memory-fraction $memoryRate
```
Once the service is successfully started, you will see a Connected log message. 

(Note: If you get stuck while downloading the model, try using a network proxy.)

**Then**, test if the LLM service has successfully started.
```
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Omost?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

**Next**, add an `Omost LLM HTTP Server` node and enter the service address of the LLM.
![image](https://github.com/huchenlei/ComfyUI_omost/assets/6883957/8cf1f3a8-f4d7-416c-a1d0-be27bc300c96)


For more information about TGI, refer to the official documentation: https://huggingface.co/docs/text-generation-inference/quicktour


## TODOs
- Implement Omost's region area cond ([DenseDiffusion](https://github.com/naver-ai/DenseDiffusion))
- Add a progress bar to the Chat node
- A region condition editor for easier manipulation post generation
