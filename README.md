# Generative AI Models

<div align="center">
  <p align="center">
    <img src="https://digitalpress.fra1.cdn.digitaloceanspaces.com/75xl145/2023/12/Foundation-Models---The-Engines-of--Generative-AI-s-Progress.jpg" alt="Generative AI" />
  </p>
<p align="center">
<strong>Generative AI Models</strong></p>
</div>

---

Welcome to the **Generative-AI-Models** repository! This repository contains a collection of generative AI models implemented using Python and popular libraries like `transformers`, `torch`, `diffusers`, and more. These models can be used for various generative tasks such as image captioning, text-to-image generation, and more.

## Getting Started

To get started with the models in this repository, you can use Google Colab, which provides a free and powerful environment for running your code with GPU acceleration.

### Prerequisites

Before using the models, make sure you have the following:

- A Google account to access Google Colab.
- Basic knowledge of Python and deep learning concepts.

### Running Models on Google Colab

We have provided Google Colab templates for each model to ensure that you can run them quickly and efficiently. Follow the steps below to get started:

1. **Open Google Colab:** Click the links provided below for each model to open the respective Colab notebook.

2. **Connect to a GPU:** In Colab, go to `Runtime` > `Change runtime type`, and select `GPU` as the hardware accelerator.

3. **Run the Notebook:** Follow the instructions within the notebook to run the cells step by step. The models are pre-configured to run efficiently on Colab's environment.

### Available Models

Below is a list of available models in this repository along with their corresponding Google Colab templates:

#### 1. Text To Image Generator

Generate images from textual descriptions using the Stable Diffusion model.

- **Model Overview:** Uses a diffusion model to create high-quality images from text prompts.
- **Colab Template:** [text_to_image_generator](https://drive.google.com/file/d/1sL1GqOeDoOYhP15I-NQnVr-Wv4POgLxM/view?usp=drive_link)
- **Image Generation Output:**
![Image_generation_output](https://github.com/Islam-hady9/Generative-AI-Models/blob/main/01_Text%20To%20Image%20Generator/Image_generation_output.png)

#### 2. Image To Text Generator

Generate descriptive captions for images using the Vision Transformer (ViT) and GPT-2 models.

- **Model Overview:** Combines ViT for image processing and GPT-2 for text generation.
- **Colab Template:** [image_to_text_generator](https://drive.google.com/file/d/1WtfTozk-zMV2763B3ZMg3Pw4VZdiNiBa/view?usp=drive_link)
- **Text Generation Output:**
![Text_generation_output](https://github.com/Islam-hady9/Generative-AI-Models/blob/main/02_Image%20To%20Text%20Generator/Text_generation_output.png)

### How to Contribute

We welcome contributions from the community! If you have a model implementation you'd like to add or improvements to suggest, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeatureName`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/YourFeatureName`.
5. Submit a pull request.

### License

This project is licensed under the MIT License.

### Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the `transformers` and `diffusers` libraries.
- [Google Colab](https://colab.research.google.com/) for offering a free and powerful platform for running deep learning models.
