import torch
from PIL import Image
from diffusers import DDIMScheduler
from accelerate.utils import set_seed
from torchvision.transforms.functional import to_pil_image, to_tensor, resize

from pipeline_sd import ADPipeline
from pipeline_sdxl import ADPipeline as ADXLPipeline
from utils import Controller

import os
# import spaces


class Runner:
    def __init__(self):
        self.sd15 = None
        self.sdxl = None
        self.loss_fn = torch.nn.L1Loss(reduction="mean")
    
    def load_pipeline(self, model_path_or_name):
        if 'xl' in model_path_or_name and self.sdxl is None:
            scheduler = DDIMScheduler.from_pretrained(model_path_or_name, subfolder="scheduler")
            self.sdxl = ADXLPipeline.from_pretrained(model_path_or_name, scheduler=scheduler, safety_checker=None)
            self.sdxl.classifier = self.sdxl.unet
        elif self.sd15 is None:
            scheduler = DDIMScheduler.from_pretrained(model_path_or_name, subfolder="scheduler")
            self.sd15 = ADPipeline.from_pretrained(model_path_or_name, scheduler=scheduler, safety_checker=None)
            self.sd15.classifier = self.sd15.unet

    def preprocecss(self, image: Image.Image, height=None, width=None):
        # TODO: resize the input image
        image = resize(image, size=512)

        if width is None or height is None: 
            width, height = image.size
        new_width = (width // 64) * 64
        new_height = (height // 64) * 64
        size = (new_width, new_height)
        image = image.resize(size, Image.BICUBIC)
        return to_tensor(image).unsqueeze(0)

    # @spaces.GPU
    def run_style_transfer(self, content_image, style_image, seed, num_steps, lr, content_weight, mixed_precision, model, **kwargs):
        self.load_pipeline(model)

        content_image = self.preprocecss(content_image)
        style_image = self.preprocecss(style_image, height=512, width=512)

        print(content_image.shape, style_image.shape)

        height, width = content_image.shape[-2:]
        set_seed(seed)
        controller = Controller(self_layers=(10, 16))
        result = self.sd15.optimize(
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
        output_image = to_pil_image(result[0].float())
        del result
        torch.cuda.empty_cache()
        return [output_image]

    # @spaces.GPU
    def run_style_t2i_generation(self, style_image, prompt, negative_prompt, guidance_scale, height, width, seed, num_steps, iterations, lr, num_images_per_prompt, mixed_precision, is_adain, model):
        self.load_pipeline(model)

        use_xl = 'xl' in model
        height, width = (1024, 1024) if 'xl' in model else (512, 512)
        style_image = self.preprocecss(style_image, height=height, width=width)

        set_seed(seed)
        self_layers = (64, 70) if use_xl else (10, 16)
        
        controller = Controller(self_layers=self_layers)

        pipeline = self.sdxl if use_xl else self.sd15
        images = pipeline.sample(
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
        output_images = [to_pil_image(image.float()) for image in images]

        del images
        torch.cuda.empty_cache()
        return output_images

    # @spaces.GPU
    def run_texture_synthesis(self, texture_image, height, width, seed, num_steps, iterations, lr, mixed_precision, num_images_per_prompt, synthesis_way,model):
        self.load_pipeline(model) 

        texture_image = self.preprocecss(texture_image, height=512, width=512)

        set_seed(seed)
        controller = Controller(self_layers=(10, 16))

        if synthesis_way == 'Sampling':
            results = self.sd15.sample(
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
            results = self.sd15.panorama(
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
        
        output_images = [to_pil_image(image.float()) for image in results]
        del results
        torch.cuda.empty_cache()
        return output_images
    
