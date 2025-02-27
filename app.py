import torch
import gradio as gr
from PIL import Image
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from accelerate.utils import set_seed
from torchvision.transforms.functional import to_pil_image, to_tensor

from pipeline_sd import ADPipeline
from pipeline_sdxl import ADPipeline as ADXLPipeline
from webui import (
    create_interface_texture_synthesis,
    create_interface_style_t2i,
    create_interface_style_transfer
)
from utils import Controller

import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"


class Runner:
    def __init__(self):
        self.pipeline = None
        self.model_name = None
        self.loss_fn = torch.nn.L1Loss(reduction="mean")
    
    def load_pipeline(self, model_path_or_name):

        self.model_name = model_path_or_name 
        
        scheduler = DDIMScheduler.from_pretrained(model_path_or_name, subfolder="scheduler")
        pipe_class = ADXLPipeline if 'xl' in model_path_or_name else ADPipeline
        self.pipeline = pipe_class.from_pretrained(model_path_or_name, scheduler=scheduler, safety_checker=None)
        self.pipeline.classifier = self.pipeline.unet

    def preprocecss(self, image: Image.Image, height=None, width=None):
        if width is None or height is None: 
            width, height = image.size
        new_width = (width // 64) * 64
        new_height = (height // 64) * 64
        size = (new_width, new_height)
        image = image.resize(size, Image.BICUBIC)
        return to_tensor(image).unsqueeze(0)

    def run_style_transfer(self, content_image, style_image, seed, num_steps, lr, content_weight, mixed_precision, model, **kwargs):
        if self.pipeline is None or self.model_name != model:
            self.load_pipeline(model) 

        content_image = self.preprocecss(content_image)
        style_image = self.preprocecss(style_image, height=512, width=512)

        height, width = content_image.shape[-2:]
        set_seed(seed)
        controller = Controller(self_layers=(10, 16))
        result = self.pipeline.optimize(
            lr=lr,
            batch_size=1,
            iters=1,
            width=width,
            height=height,
            weight=content_weight,
            controller=controller,
            style_image=style_image,
            content_image=content_image,
            mixed_precision=mixed_precision,
            num_inference_steps=num_steps,
            enable_gradient_checkpoint=False,
        )
        output_image = to_pil_image(result[0])
        del result
        torch.cuda.empty_cache()
        return [output_image]

    def run_style_t2i_generation(self, style_image, prompt, negative_prompt, guidance_scale, height, width, seed, num_steps, iterations, lr, num_images_per_prompt, mixed_precision, is_adain, model):
        if self.pipeline is None or self.model_name != model:
            self.load_pipeline(model)

        height, width = (1024, 1024) if 'xl' in self.model_name else (512, 512)
        style_image = self.preprocecss(style_image, height=height, width=width)

        set_seed(seed)
        if isinstance(self.pipeline, StableDiffusionPipeline):
            self_layers = (10, 16)
        elif isinstance(self.pipeline, StableDiffusionXLPipeline):
            self_layers = (64, 70)
        else:
            raise ValueError
        controller = Controller(self_layers=self_layers)

        images = self.pipeline.sample(
            controller=controller,
            iters=iterations,
            lr=lr,
            adain=is_adain,
            height=height,
            width=width,
            mixed_precision=mixed_precision,
            style_image=style_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            num_images_per_prompt=num_images_per_prompt,
            enable_gradient_checkpoint=False
        )
        output_images = [to_pil_image(image) for image in images]

        del images
        torch.cuda.empty_cache()
        return output_images

    def run_texture_synthesis(self, texture_image, height, width, seed, num_steps, iterations, lr, mixed_precision, num_images_per_prompt, synthesis_way,model):
        if self.pipeline is None or self.model_name != model:
            self.load_pipeline(model) 

        texture_image = self.preprocecss(texture_image, height=512, width=512)

        set_seed(seed)
        controller = Controller(self_layers=(10, 16))

        if synthesis_way == 'Sampling':
            results = self.pipeline.sample(
                lr=lr,
                adain=False,
                iters=iterations,
                width=width,
                height=height,
                weight=0.,
                controller=controller,
                style_image=texture_image,
                content_image=None,
                prompt="",
                negative_prompt="",
                mixed_precision=mixed_precision,
                num_inference_steps=num_steps,
                guidance_scale=1.,
                num_images_per_prompt=num_images_per_prompt,
                enable_gradient_checkpoint=False,
            )
        elif synthesis_way == 'MultiDiffusion':   
            results = self.pipeline.panorama(
                lr=lr,
                iters=iterations,
                width=width,
                height=height,
                weight=0.,
                controller=controller,
                style_image=texture_image,
                content_image=None,
                prompt="",
                negative_prompt="",
                stride=8,
                view_batch_size=8,
                mixed_precision=mixed_precision,
                num_inference_steps=num_steps,
                guidance_scale=1.,
                num_images_per_prompt=num_images_per_prompt,
                enable_gradient_checkpoint=False,
            )
        else:
            raise ValueError
        
        output_images = [to_pil_image(image) for image in results]
        del results
        torch.cuda.empty_cache()
        return output_images


def main():
    runner = Runner()

    with gr.Blocks(analytics_enabled=False,
                   title='Attention Distillation',
                   ) as demo:

        md_txt = "# Attention Distillation" \
                 "\nOfficial demo of the paper [Attention Distillation: A Unified Approach to Visual Characteristics Transfer]()"
        gr.Markdown(md_txt)

        with gr.Tabs(selected='tab_style_transfer'):
            with gr.TabItem("Style Transfer", id='tab_style_transfer'):
                create_interface_style_transfer(runner=runner)

            with gr.TabItem("Style-Specific Text-to-Image Generation", id='tab_style_t2i'):
                create_interface_style_t2i(runner=runner)

            with gr.TabItem("Texture Synthesis", id='tab_texture_syn'):
                create_interface_texture_synthesis(runner=runner)
            
        demo.launch(share=False, debug=False)


if __name__ == '__main__':
    main()
