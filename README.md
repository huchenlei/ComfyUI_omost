# ComfyUI_omost

ComfyUI implementation of [Omost](https://github.com/lllyasviel/Omost), and everything about regional prompt.

## News
- [2024-06-09] Canvas editor added https://github.com/huchenlei/ComfyUI_omost/pull/28
- [2024-06-09] Add option to connect to external LLM service https://github.com/huchenlei/ComfyUI_omost/pull/25
- [2024-06-10] Add OmostDenseDiffusion regional prompt backend support (The same as original Omost repo) https://github.com/huchenlei/ComfyUI_omost/pull/27

## TODOs
- Add a progress bar to the Chat node
- Implement gradient optimization regional prompt
- Implement multi-diffusion regional prompt

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
According to https://github.com/lllyasviel/Omost#regional-prompter, there are 6 ways to perform region guided diffusion.

#### Method 1: Multi-diffusion / mixture-of-diffusers
> These method run UNet on different locations, and then merge the estimated epsilon or x0 using weights or masks for different regions.

TO be implemented

#### Method 2: Attention decomposition
> lets say attention is like y=softmax(q@k)@v, then one can achieve attention decomposition like y=mask_A * softmax(q@k_A)@v_A + mask_B * softmax(q@k_B)@v_B where mask_A, k_A, v_A are masks, k, v for region A; mask_B, k_B, v_B are masks, k, v for region B. This method usually yields image quality a bit better than (1) and some people call it Attention Couple or Region Prompter Attention Mode. But this method has a consideration: the mask only makes regional attention numerically possible, but it does not force the UNet to really attend its activations in those regions. That is to say, the attention is indeed masked, but there is no promise that the attention softmax will really be activated in the masked area, and there is also no promise that the attention softmax will never be activated outside the masked area.

This is the built-in regional prompt method in ComfyUI. Use `Omost Layout Cond (ComfyUI-Area)` node for this method.

There are 2 overlap methods:
- Overlay: The layer on top completely overwrites layer below
- Average: The overlapped area is the average of all conditions
![image](https://github.com/huchenlei/ComfyUI_omost/assets/20929282/e7d007e4-1175-4435-adf4-a9211937d8c1)

#### Method 3: Attention score manipulation
> this is a more advanced method compared to (2). It directly manipulates the attention scores to make sure that the activations in mask each area are encouraged and those outside the masks are discouraged. The formulation is like y=softmax(modify(q@k))@v where modify() is a complicated non-linear function with many normalizations and tricks to change the score's distributions. This method goes beyond a simple masked attention to really make sure that those layers get wanted activations. A typical example is Dense Diffusion.

This is the method used by original Omost repo. To use this method:
- Install https://github.com/huchenlei/ComfyUI_densediffusion
- Include `Omost Layout Cond (OmostDenseDiffusion)` node to your workflow

Note: ComfyUI_densediffusion does not compose with IPAdapter.

![10 06 2024_16 37 22_REC](https://github.com/huchenlei/ComfyUI_omost/assets/20929282/30cf059d-929a-4f11-8f5d-0160d9c5cd22)

#### Method 4: Gradient optimization
> since the attention can tell us where each part is corresponding to what prompts, we can split prompts into segments and then get attention activations to each prompt segment. Then we compare those activations with external masks to compute a loss function, and back propagate the gradients. Those methods are usually very high quality but VRAM hungry and very slow. Typical methods are BoxDiff and Attend-and-Excite.

To be implemented

#### Method 5: Use external control models like gligen and InstanceDiffusion
> Those methods give the highest benchmark performance on region following but will also introduce some style offset to the base model since they are trained parameters. Also, those methods need to convert prompts to vectors and usually do not support prompts of arbitary length (but one can use them together with other attention methods to achieve arbitrary length).

To be implemented

#### Method 6: Some more possible layer options like layerdiffuse and mulan
To be implemented

Optionally you can also pass the image generated from Omost canvas as initial latent as described in the original Omost repo:
![image](https://github.com/huchenlei/ComfyUI_omost/assets/20929282/f913d141-9045-41fa-998f-770a840adc69)

### Edit Region condition
You can use the built-in region editor on `Omost Load Canvas Conditioning` node to freely manipulate the LLM output.
![image](https://github.com/huchenlei/ComfyUI_omost/assets/20929282/bff0f6d5-ea28-41b2-ae7c-fec29691584f)
![image](https://github.com/huchenlei/ComfyUI_omost/assets/20929282/eb2a692f-3643-434a-a1d9-4443c82629b8)

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
