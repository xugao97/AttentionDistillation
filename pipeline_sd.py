import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import utils
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import PipelineImageInput
from losses import *
from tqdm import tqdm

from torch import Tensor
from torch.nn import Conv2d
from torch.nn.modules.utils import _pair


# asymmetric tiling from https://github.com/tjm35/asymmetric-tiling-sd-webui/blob/main/scripts/asymmetric_tiling.py
def make_circular_asymm(model, tileX: bool, tileY: bool):
    for layer in [
        layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)
    ]:
        layer.padding_modeX = 'circular' if tileX else 'constant'
        layer.padding_modeY = 'circular' if tileY else 'constant'
        layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
        layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
        layer._conv_forward = __replacementConv2DConvForward.__get__(layer, Conv2d)
    return model


def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    working = F.pad(input, self.paddingX, mode=self.padding_modeX)
    working = F.pad(working, self.paddingY, mode=self.padding_modeY)
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


class ADPipeline(StableDiffusionPipeline):
    def freeze(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
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
        self.enable_vae_slicing()
        # self.enable_model_cpu_offload()
        # self.enable_vae_tiling()
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        self.unet.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        self.classifier.to(self.accelerator.device, dtype=weight_dtype)
        self.classifier = self.accelerator.prepare(self.classifier)
        if enable_gradient_checkpoint:
            self.classifier.enable_gradient_checkpointing()

    def enable_tiling(self, tiling):
        if tiling == "enable":
            make_circular_asymm(self.unet, True, True)
            make_circular_asymm(self.vae, True, True)
        elif tiling == "x_only":
            make_circular_asymm(self.unet, True, False)
            make_circular_asymm(self.vae, True, False)
        elif tiling == "y_only":
            make_circular_asymm(self.unet, False, True)
            make_circular_asymm(self.vae, False, True)
        else:
            make_circular_asymm(self.unet, False, False)
            make_circular_asymm(self.vae, False, False)



    def sample(
        self,
        lr=0.05,
        iters=1,
        attn_scale=1,
        adain=False,
        weight=0.25,
        tiling=False,
        controller=None,
        style_image=None,
        content_image=None,
        mixed_precision="no",
        start_time=999,
        enable_gradient_checkpoint=False,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, gradient_accumulation_steps=1
        )
        self.init(enable_gradient_checkpoint)
        self.enable_tiling(tiling)

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
        do_cfg = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_cfg,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                do_cfg,
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

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        self.style_latent = self.image2latent(style_image)
        if content_image is not None:
            self.content_latent = self.image2latent(content_image)
        else:
            self.content_latent = None
        null_embeds = self.encode_prompt("", device, 1, False)[0]
        self.null_embeds = null_embeds
        self.null_embeds_for_latents = torch.cat([null_embeds] * latents.shape[0])
        self.null_embeds_for_style = torch.cat(
            [null_embeds] * self.style_latent.shape[0]
        )
        
        self.adain = adain
        self.attn_scale = attn_scale
        self.cache = utils.DataCache()
        self.controller = controller
        utils.register_attn_control(
            self.classifier, controller=self.controller, cache=self.cache
        )
        print("Total self attention layers of Unet: ", controller.num_self_layers)
        print("Self attention layers for AD: ", controller.self_layers)

        pbar = tqdm(timesteps, desc="Sample")
        for i, t in enumerate(pbar):
            with torch.no_grad():
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
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
                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
            if iters > 0 and t < start_time:
                latents = self.AD(latents, t, lr, iters, pbar, weight)
                
        images = self.latent2image(latents)
        # Offload all models
        self.maybe_free_model_hooks()
        return images

    def optimize(
        self,
        latents=None,
        attn_scale=1.0,
        lr=0.05,
        iters=1,
        weight=0,
        width=512,
        height=512,
        batch_size=1,
        controller=None,
        style_image=None,
        content_image=None,
        mixed_precision="no",
        num_inference_steps=50,
        enable_gradient_checkpoint=False,
        source_mask=None,
        target_mask=None,
    ):
        height = height // self.vae_scale_factor
        width = width // self.vae_scale_factor

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, gradient_accumulation_steps=1
        )
        self.init(enable_gradient_checkpoint)

        style_latent = self.image2latent(style_image)
        latents = torch.randn((batch_size, 4, height, width), device=self.device)
        null_embeds = self.encode_prompt("", self.device, 1, False)[0]
        null_embeds_for_latents = null_embeds.repeat(latents.shape[0], 1, 1)
        null_embeds_for_style = null_embeds.repeat(style_latent.shape[0], 1, 1)

        if content_image is not None:
            content_latent = self.image2latent(content_image)
            latents = torch.cat([content_latent.clone()] * batch_size)
            null_embeds_for_content = null_embeds.repeat(content_latent.shape[0], 1, 1)

        self.cache = utils.DataCache()
        self.controller = controller
        utils.register_attn_control(
            self.classifier, controller=self.controller, cache=self.cache
        )
        print("Total self attention layers of Unet: ", controller.num_self_layers)
        print("Self attention layers for AD: ", controller.self_layers)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        latents = latents.detach().float()
        optimizer = torch.optim.Adam([latents.requires_grad_()], lr=lr)
        optimizer = self.accelerator.prepare(optimizer)
        pbar = tqdm(timesteps, desc="Optimize")
        for i, t in enumerate(pbar):
            # t = torch.tensor([1], device=self.device)
            with torch.no_grad():
                qs_list, ks_list, vs_list, s_out_list = self.extract_feature(
                    style_latent,
                    t,
                    null_embeds_for_style,
                )
                if content_image is not None:
                    qc_list, kc_list, vc_list, c_out_list = self.extract_feature(
                        content_latent,
                        t,
                        null_embeds_for_content,
                    )
            for j in range(iters):
                style_loss = 0
                content_loss = 0
                optimizer.zero_grad()
                q_list, k_list, v_list, self_out_list = self.extract_feature(
                    latents,
                    t,
                    null_embeds_for_latents,
                )
                style_loss = ad_loss(q_list, ks_list, vs_list, self_out_list, scale=attn_scale, source_mask=source_mask, target_mask=target_mask)
                if content_image is not None:
                    content_loss = q_loss(q_list, qc_list)
                    # content_loss = qk_loss(q_list, k_list, qc_list, kc_list)
                    # content_loss = qkv_loss(q_list, k_list, vc_list, c_out_list)
                loss = style_loss + content_loss * weight
                self.accelerator.backward(loss)
                optimizer.step()
                pbar.set_postfix(loss=loss.item(), time=t.item(), iter=j)
        images = self.latent2image(latents)
        # Offload all models
        self.maybe_free_model_hooks()
        return images

    def panorama(
        self,
        lr=0.05,
        iters=1,
        attn_scale=1,
        adain=False,
        controller=None,
        style_image=None,
        mixed_precision="no",
        enable_gradient_checkpoint=False,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1,
        stride=8,
        view_batch_size: int = 16,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        **kwargs,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
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
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_cfg = guidance_scale > 1.0

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_cfg,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

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

        # 6. Define panorama grid and initialize views for synthesis.
        # prepare batch grid
        views = self.get_views_(height, width, window_size=64, stride=stride)
        views_batch = [
            views[i : i + view_batch_size]
            for i in range(0, len(views), view_batch_size)
        ]
        print(len(views), len(views_batch), views_batch)
        self.scheduler.set_timesteps(num_inference_steps)
        views_scheduler_status = [copy.deepcopy(self.scheduler.__dict__)] * len(
            views_batch
        )
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 8. Denoising loop
        # Each denoising step also includes refinement of the latents with respect to the
        # views.

        timesteps = self.scheduler.timesteps
        self.style_latent = self.image2latent(style_image)
        self.content_latent = None
        null_embeds = self.encode_prompt("", device, 1, False)[0]
        self.null_embeds = null_embeds
        self.null_embeds_for_latents = torch.cat([null_embeds] * latents.shape[0])
        self.null_embeds_for_style = torch.cat(
            [null_embeds] * self.style_latent.shape[0]
        )
        self.adain = adain
        self.attn_scale = attn_scale
        self.cache = utils.DataCache()
        self.controller = controller
        utils.register_attn_control(
            self.classifier, controller=self.controller, cache=self.cache
        )
        print("Total self attention layers of Unet: ", controller.num_self_layers)
        print("Self attention layers for AD: ", controller.self_layers)

        pbar = tqdm(timesteps, desc="Sample")
        for i, t in enumerate(pbar):
            count.zero_()
            value.zero_()
            # generate views
            # Here, we iterate through different spatial crops of the latents and denoise them. These
            # denoised (latent) crops are then averaged to produce the final latent
            # for the current timestep via MultiDiffusion. Please see Sec. 4.1 in the
            # MultiDiffusion paper for more details: https://arxiv.org/abs/2302.08113
            # Batch views denoise
            for j, batch_view in enumerate(views_batch):
                vb_size = len(batch_view)
                # get the latents corresponding to the current view coordinates
                latents_for_view = torch.cat(
                    [
                        latents[:, :, h_start:h_end, w_start:w_end]
                        for h_start, h_end, w_start, w_end in batch_view
                    ]
                )
                # rematch block's scheduler status
                self.scheduler.__dict__.update(views_scheduler_status[j])

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    latents_for_view.repeat_interleave(2, dim=0)
                    if do_cfg
                    else latents_for_view
                )

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # repeat prompt_embeds for batch
                prompt_embeds_input = torch.cat([prompt_embeds] * vb_size)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds_input,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                    # perform guidance
                    if do_cfg:
                        noise_pred_uncond, noise_pred_text = (
                            noise_pred[::2],
                            noise_pred[1::2],
                        )
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_denoised_batch = self.scheduler.step(
                        noise_pred, t, latents_for_view, **extra_step_kwargs
                    ).prev_sample
                if iters > 0:
                    self.null_embeds_for_latents = torch.cat(
                        [self.null_embeds] * noise_pred.shape[0]
                    )
                    latents_denoised_batch = self.AD(
                        latents_denoised_batch, t, lr, iters, pbar
                    )
                # save views scheduler status after sample
                views_scheduler_status[j] = copy.deepcopy(self.scheduler.__dict__)

                # extract value from batch
                for latents_view_denoised, (h_start, h_end, w_start, w_end) in zip(
                    latents_denoised_batch.chunk(vb_size), batch_view
                ):

                    value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                    count[:, :, h_start:h_end, w_start:w_end] += 1

            # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
            latents = torch.where(count > 0, value / count, value)

        images = self.latent2image(latents)
        # Offload all models
        self.maybe_free_model_hooks()
        return images

    def AD(self, latents, t, lr, iters, pbar, weight=0):
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
                add_noise=True,
            )
            if self.content_latent is not None:
                qc_list, kc_list, vc_list, c_out_list = self.extract_feature(
                    self.content_latent,
                    t,
                    self.null_embeds,
                    add_noise=True,
                )

        latents = latents.detach()
        optimizer = torch.optim.Adam([latents.requires_grad_()], lr=lr)
        optimizer = self.accelerator.prepare(optimizer)

        for j in range(iters):
            style_loss = 0
            content_loss = 0
            optimizer.zero_grad()
            q_list, k_list, v_list, self_out_list = self.extract_feature(
                latents,
                t,
                self.null_embeds_for_latents,
                add_noise=False,
            )
            style_loss = ad_loss(q_list, ks_list, vs_list, self_out_list, scale=self.attn_scale)
            if self.content_latent is not None:
                content_loss = q_loss(q_list, qc_list)
                # content_loss = qk_loss(q_list, k_list, qc_list, kc_list)
                # content_loss = qkv_loss(q_list, k_list, vc_list, c_out_list)
            loss = style_loss + content_loss * weight
            self.accelerator.backward(loss)
            optimizer.step()

            pbar.set_postfix(loss=loss.item(), time=t.item(), iter=j)
        latents = latents.detach()
        return latents

    def extract_feature(
        self,
        latent,
        t,
        embeds,
        add_noise=False,
    ):
        self.cache.clear()
        self.controller.step()
        if add_noise:
            noise = torch.randn_like(latent)
            latent_ = self.scheduler.add_noise(latent, noise, t)
        else:
            latent_ = latent
        _ = self.classifier(latent_, t, embeds)[0]
        return self.cache.get()

    def get_views_(
        self,
        panorama_height: int,
        panorama_width: int,
        window_size: int = 64,
        stride: int = 8,
    ) -> List[Tuple[int, int, int, int]]:
        panorama_height //= 8
        panorama_width //= 8

        num_blocks_height = (
            math.ceil((panorama_height - window_size) / stride) + 1
            if panorama_height > window_size
            else 1
        )
        num_blocks_width = (
            math.ceil((panorama_width - window_size) / stride) + 1
            if panorama_width > window_size
            else 1
        )

        views = []
        for i in range(int(num_blocks_height)):
            for j in range(int(num_blocks_width)):
                h_start = int(min(i * stride, panorama_height - window_size))
                w_start = int(min(j * stride, panorama_width - window_size))

                h_end = h_start + window_size
                w_end = w_start + window_size

                views.append((h_start, h_end, w_start, w_end))

        return views
