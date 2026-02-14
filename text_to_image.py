# Simple text-to-image generation using Hugging Face Diffusers
# Run with: python text_to_image.py

from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Generate image from prompt
prompt = "A futuristic 3D scene of a cyberpunk city with flying cars, highly detailed, cinematic lighting"
image = pipe(prompt).images[0]

# Save the result
image.save("generated_cyberpunk_city.png")
print("Image generated and saved as generated_cyberpunk_city.png")