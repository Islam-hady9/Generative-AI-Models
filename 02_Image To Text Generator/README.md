# Image To Text Generator

This script demonstrates how to generate descriptive captions for images using a Vision Transformer (ViT) for image processing and GPT-2 for text generation. The model used in this script is the `nlpconnect/vit-gpt2-image-captioning`, a pre-trained model available in the Hugging Face Model Hub.

## Dependencies

The following Python libraries are required for this script:

- `transformers`: A library for NLP and vision models, used here for both image processing and text generation.
- `torch`: PyTorch library for tensor computations and model inference.
- `PIL`: Python Imaging Library (Pillow) for image handling.

You can install these dependencies using pip:

```bash
!pip install transformers torch pillow
```

## Model Initialization

The script initializes the model and the necessary components to generate captions:

1. **VisionEncoderDecoderModel:** Combines a vision encoder (ViT) and a text decoder (GPT-2).
2. **ViTImageProcessor:** Processes images into a format suitable for the model's vision encoder.
3. **AutoTokenizer:** Tokenizes and decodes the text for the model's text decoder.

```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
```

## Configuration

The script sets up the device for computation (CPU or GPU) and configures parameters for caption generation:

- **device:** Determines if the script runs on a GPU (if available) or CPU.
- **max_length:** Maximum length of the generated caption.
- **num_beams:** Number of beams for beam search, controlling the quality of the generated caption.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
```

## Caption Generation Function

The `predict_step` function generates captions for a list of image paths. The function performs the following steps:

1. **Image Loading and Conversion:** Each image is loaded using PIL and converted to RGB if necessary.
2. **Feature Extraction:** The images are processed into pixel values suitable for the vision encoder.
3. **Caption Generation:** The processed images are passed through the model to generate captions.
4. **Text Decoding:** The generated text tokens are decoded into human-readable captions.

```python
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds
```

## User Interaction

The script prompts the user to input the path to an image file. It then generates and prints a caption for the image.

```python
image_path = input("Enter the image path: ")
print("\n")
print(f"Predicted Caption: {predict_step([image_path])}")
```

## Usage Example

1. **Run the script:** Execute the script in a Python environment with access to the required libraries.
2. **Enter the image path:** When prompted, enter the path to the image file you want to caption (e.g., "path/to/your/image.jpg").
3. **View the caption:** The script will generate and display a descriptive caption for the image.
