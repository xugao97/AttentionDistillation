import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import utils
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import is_compiled_module
from losses import *

# from peft import LoraConfig, set_peft_model_state_dict
from tqdm import tqdm


class ADPipeline(StableDiffusionXLPipeline):
    def freeze(self):
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.classifier.requires_grad_(False)

    @torch.no_grad()
    def image2latent(self, image):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        image = image.to(device=device, dtype=dtype) * 2.0 - 1.0
        latent = self.vae.encode(image)["latent_dist"].mean
        latent = latent * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def latent2image(self, latent):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        latent = latent.to(device=device, dtype=dtype)
        latent = latent / self.vae.config.scaling_factor
        image = self.vae.decode(latent)[0]
        return (image * 0.5 + 0.5).clamp(0, 1)

    def init(self, enable_gradient_checkpoint):
        self.freeze()
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        self.unet.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        self.text_encoder_2.to(self.accelerator.device, dtype=weight_dtype)
        self.classifier.to(self.accelerator.device, dtype=weight_dtype)
        self.classifier = self.accelerator.prepare(self.classifier)
        if enable_gradient_checkpoint:
            self.classifier.enable_gradient_checkpointing()
            # self.classifier.train()
     

    def sample(
        self,
        lr=0.05,
        iters=1,
        adain=True,
        controller=None,
        style_image=None,
        mixed_precision="no",
        init_from_style=False,
        start_time=999,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        enable_gradient_checkpoint=False,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, gradient_accumulation_steps=1
        )
        self.init(enable_gradient_checkpoint)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        null_add_time_ids = add_time_ids.to(device)
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        self.timestep_cond = timestep_cond
        (null_embeds, _, null_pooled_embeds, _) = self.encode_prompt("", device=device)

        added_cond_kwargs = {
            "text_embeds": add_text_embeds, 
            "time_ids": add_time_ids
            }
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            added_cond_kwargs["image_embeds"] = image_embeds

        self.scheduler.set_timesteps(num_inference_steps)

        timesteps = self.scheduler.timesteps
        style_latent = self.image2latent(style_image)
        if init_from_style:
            latents = torch.cat([style_latent] * latents.shape[0])
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(
                latents,
                noise,
                torch.tensor([999]),
            )

        self.style_latent = style_latent
        self.null_embeds_for_latents = torch.cat([null_embeds] * (latents.shape[0]))
        self.null_embeds_for_style = torch.cat([null_embeds] * style_latent.shape[0])
        self.null_added_cond_kwargs_for_latents = {
            "text_embeds": torch.cat([null_pooled_embeds] * (latents.shape[0])),
            "time_ids": torch.cat([null_add_time_ids] * (latents.shape[0])),
        }
        self.null_added_cond_kwargs_for_style = {
            "text_embeds": torch.cat([null_pooled_embeds] * style_latent.shape[0]),
            "time_ids": torch.cat([null_add_time_ids] * style_latent.shape[0]),
        }
        self.adain = adain
        self.cache = utils.DataCache()
        self.controller = controller
        utils.register_attn_control(
            self.classifier, controller=controller, cache=self.cache
        )
        print("Total self attention layers of Unet: ", controller.num_self_layers)
        print("Self attention layers for AD: ", controller.self_layers)

        pbar = tqdm(timesteps, desc="Sample")
        for i, t in enumerate(pbar):
            with torch.no_grad():
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
            if iters > 0 and t < start_time:
                latents = self.AD(latents, t, lr, iters, pbar)    
                
                
        # Offload all models
        # self.enable_model_cpu_offload()
        images = self.latent2image(latents)
        self.maybe_free_model_hooks()
        return images

    def AD(self, latents, t, lr, iters, pbar):
        t = max(
            t
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps,
            torch.tensor([0], device=self.device),
        )

        if self.adain:
            noise = torch.randn_like(self.style_latent)
            style_latent = self.scheduler.add_noise(self.style_latent, noise, t)
            latents = utils.adain(latents, style_latent)

        with torch.no_grad():
            qs_list, ks_list, vs_list, s_out_list = self.extract_feature(
                self.style_latent,
                t,
                self.null_embeds_for_style,
                self.timestep_cond,
                self.null_added_cond_kwargs_for_style,
                add_noise=True,
            )
        # latents = latents.to(dtype=torch.float32)
        latents = latents.detach()
        optimizer = torch.optim.Adam([latents.requires_grad_()], lr=lr)
        optimizer, latents = self.accelerator.prepare(optimizer, latents)

        for j in range(iters):
            optimizer.zero_grad()
            q_list, k_list, v_list, self_out_list = self.extract_feature(
                latents,
                t,
                self.null_embeds_for_latents,
                self.timestep_cond,
                self.null_added_cond_kwargs_for_latents,
                add_noise=False,
            )

            loss = ad_loss(q_list, ks_list, vs_list, self_out_list)
            self.accelerator.backward(loss)
            optimizer.step()

            pbar.set_postfix(loss=loss.item(), time=t.item(), iter=j)
        latents = latents.detach()
        return latents

    def extract_feature(
        self,
        latent,
        t,
        encoder_hidden_states,
        timestep_cond,
        added_cond_kwargs,
        add_noise=False,
    ):
        self.cache.clear()
        self.controller.step()
        if add_noise:
            noise = torch.randn_like(latent)
            latent_ = self.scheduler.add_noise(latent, noise, t)
        else:
            latent_ = latent
        self.classifier(
            latent_,
            t,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        return self.cache.get()
