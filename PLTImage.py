from diffusers import AutoPipelineForImage2Image
from PIL import Image
import torch
import random

model_id = "runwayml/stable-diffusion-v1-5" 
prompt = "anime style cat, graceful, adorable, well-defined face and limbs, some mechanical parts, cables, wires sparking"

reference_img_folder_address = "/Users/andrewfong/Downloads/SDReferenceAndGenerated/PicturesCat/"
batch_size = 2
run_times = 5
range_seed = 10000
range_images = 4

pipeline = AutoPipelineForImage2Image.from_pretrained(
    model_id,
    use_safetensors=True
).to("mps")
pipeline.enable_attention_slicing()
prompt = batch_size * [prompt]

for _ in range(run_times):
    random_seed_num = random.randrange(range_seed)
    random_guidance_scale = random.randrange(700, 1350, 50)/100.0
    random_image_num = random.randrange(range_images)
    random_image_num = 5

    image = Image.open(reference_img_folder_address + "{}.png".format(random_image_num)).convert("RGB")
    image.thumbnail((768, 768))
    image = batch_size * [image]
    generator = [torch.manual_seed(random_image_num+i) for i in range(batch_size)]

    images = pipeline(prompt, image, generator = generator, num_inference_steps=25, 
                      strength=0.6, guidance_scale=random_guidance_scale).images
    
    for image in images:
        image.save("/Users/andrewfong/Downloads/SDReferenceAndGenerated/SDPicturesImg2Img/{}.png".format(
        "S#: " + str(random_seed_num) + "; " +  "GS: " + str(random_guidance_scale) + "; " + "I#: " + str(random_image_num)))
        random_seed_num+=1