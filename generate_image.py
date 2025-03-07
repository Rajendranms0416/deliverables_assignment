from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to("cpu") 

# Define prompt
prompt = "a serene sunset over a futuristic city"

# Generate 3 images
for i in range(3):
    image = pipe(prompt).images[0]
    image.save(f"generated_image_{i+1}.png")
    print(f"Saved: generated_image_{i+1}.png")

print("Image generation completed.")