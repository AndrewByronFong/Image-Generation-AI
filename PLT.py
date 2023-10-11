from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import random

model_id = "runwayml/stable-diffusion-v1-5" 
pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    use_safetensors=True
).to("mps")
prompt = "a cyperpunk cat, hd, highly-detailed"
batch_size = 8
run_times = 10
range_seed = 10000
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

pipeline.enable_attention_slicing()
prompt = batch_size * [prompt]

for _ in range(run_times):
    random_seed_num = random.randrange(range_seed)
    random_guidance_scale = random.randrange(700, 1350, 50)/100.0
    generator = [torch.manual_seed(random_seed_num+i) for i in range(batch_size)]
    gs = "GS: " + str(random_guidance_scale)

    images = pipeline(prompt = prompt, generator = generator, num_inference_steps=20, guidance_scale=random_guidance_scale).images
    for image in images:
        image.save("/Users/andrewfong/Downloads/SDReferenceAndGenerated/SDPicturesTxt2Img/{}.png".format(
        "S#: " + str(random_seed_num) + "; " + gs))
        random_seed_num+=1