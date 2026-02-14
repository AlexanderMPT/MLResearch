# Experiment: comparing different inference steps in Stable Diffusion
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "A detailed 3D Gaussian splatting scene of a forest with glowing mushrooms"

# Experiment with different number of inference steps
for steps in [20, 50, 100]:
    image = pipe(prompt, num_inference_steps=steps).images[0]
    image.save(f"forest_mushrooms_{steps}_steps.png")
    print(f"Generated with {steps} steps")