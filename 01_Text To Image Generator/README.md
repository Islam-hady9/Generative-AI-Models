# Text To Image Generator

This script demonstrates how to generate images from textual descriptions using the Stable Diffusion model. The implementation leverages the `diffusers` and `transformers` libraries provided by Hugging Face, as well as other Python packages such as `torch` and `cv2`.

## Dependencies

The following Python libraries are required for this script:

- `diffusers`: A library for running diffusion models.
- `transformers`: A library for NLP models, used here to generate prompts.
- `torch`: PyTorch library for tensor computations and model inference.
- `cv2`: OpenCV library for image processing.
- `matplotlib`: Library for plotting images.
- `tqdm`: Library for progress bars.
- `numpy`: Numerical computations.

You can install these dependencies using pip:

```bash
!pip install --upgrade diffusers transformers -q
```

## Configuration

The `CFG` class defines various configuration settings used throughout the script:

- **device:** Specifies the device for computation (e.g., "cuda" for GPU).
- **seed:** Seed for random number generation to ensure reproducibility.
- **generator:** PyTorch random number generator initialized with the specified seed.
- **image_gen_steps:** Number of inference steps for image generation.
- **image_gen_model_id:** Identifier for the Stable Diffusion model.
- **image_gen_size:** Desired size of the generated images.
- **image_gen_guidance_scale:** Guidance scale for image generation, controlling adherence to the text prompt.
- **prompt_gen_model_id:** Identifier for the GPT-2 model, which can be used for generating text prompts.
- **prompt_dataset_size:** Number of prompts to generate.
- **prompt_max_length:** Maximum length of the generated prompts.

## Model Initialization

The Stable Diffusion model is loaded using the `StableDiffusionPipeline` class. The model is then transferred to the specified device (e.g., GPU) for efficient computation.

```python
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='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', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)
```

## Image Generation Function

The `generate_image` function takes a text prompt and generates an image based on it. The function performs the following steps:

1. **Image Generation:** The model generates an image based on the provided prompt.
2. **Image Resizing:** The generated image is resized to the specified dimensions.

```python
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image
```

## User Interaction

The script prompts the user to input a description of the desired image. It then generates the image based on the input and displays or saves it.

```python
image_description = input("Enter a description of the image you want to generate: ")
generate_image(image_description, image_gen_model)
```

## Usage Example

1. **Run the script:** Execute the script in a Python environment with access to the required libraries.
2. **Enter a prompt:** When prompted, enter a text description of the image you want to generate (e.g., "A sunset over a mountain range").
3. **View the result:** The generated image will be displayed or saved based on your implementation. 
