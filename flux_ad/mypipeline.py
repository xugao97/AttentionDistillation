import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import utils
from accelerate import Accelerator
from diffusers import FluxPipeline


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class AttDistPipeline(FluxPipeline):

    def freeze(self):
        self.transformer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)

    @torch.no_grad()
    def image2latent(self, image):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        image = image.to(device=device, dtype=dtype) * 2.0 - 1.0
        latent = retrieve_latents(self.vae.encode(image))
        latent = (
            latent - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def latent2image(self, latent, height, width):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        latent = latent.to(device=device, dtype=dtype)
        latents = self._unpack_latents(latent, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return (image * 0.5 + 0.5).clamp(0, 1)

    def sample(
        self,
        style_image=None,
        controller=None,
        loss_fn=None,
        start_time=9999,
        lr=0.05,
        iters=2,
        mixed_precision="no",
        enable_gradient_checkpoint=False,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        # timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        device = self._execution_device
        accelerator = Accelerator(
            mixed_precision=mixed_precision, gradient_accumulation_steps=1
        )
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        self.transformer.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder_2.to(accelerator.device, dtype=weight_dtype)
        self.transformer = accelerator.prepare(self.transformer)
        if enable_gradient_checkpoint:
            self.transformer.enable_gradient_checkpointing()
            # self.transformer.train()

        (null_embeds, null_pooled_embeds, null_text_ids) = self.encode_prompt(
            prompt="",
            prompt_2=prompt_2,
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            null_embeds.dtype,
            device,
            generator,
            latents,
        )

        print(style_image.shape)
        height_, width_ = style_image.shape[2], style_image.shape[3]
        style_latent = self.image2latent(style_image)
        print(style_latent.shape)
        print(latents.shape)
        style_latent = self._pack_latents(style_latent, 1, num_channels_latents, style_latent.shape[2], style_latent.shape[3])

        _, null_image_id = self.prepare_latents(
            num_images_per_prompt,
            num_channels_latents,
            height_,
            width_,
            null_embeds.dtype,
            device,
            generator,
            style_latent,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            None,
            sigmas,
            mu=mu,
        )

        timesteps = self.scheduler.timesteps
        print(f"timesteps: {timesteps}")
        self._num_timesteps = len(timesteps)

        cache = utils.DataCache()
        
        utils.register_attn_control(
            self.transformer.transformer_blocks,
            controller=controller,
            cache=cache,
        )
        utils.register_attn_control(
            self.transformer.single_transformer_blocks,
            controller=controller,
            cache=cache,
        )
        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        null_guidance = torch.full(
                [1], 1, device=device, dtype=torch.float32
            )
        
        print(controller.num_self_layers)


        pbar = tqdm(timesteps, desc="Sample")
        for i, t in enumerate(pbar):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            with torch.no_grad():
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]
            if t < start_time:
                if i < num_inference_steps - 1:
                    timestep = timesteps[i+1:i+2]
                    # print(timestep)
                    noise = torch.randn_like(style_latent)
                    # print(style_latent.shape)
                    style_latent_ = self.scheduler.scale_noise(style_latent, timestep, noise)
                else:
                    timestep = torch.tensor([0], device=style_latent.device)
                    style_latent_ = style_latent

                cache.clear()
                controller.step()
                
                _ = self.transformer(
                    hidden_states=style_latent_,
                    timestep=timestep / 1000,
                    guidance=null_guidance,
                    pooled_projections=null_pooled_embeds,
                    encoder_hidden_states=null_embeds,
                    txt_ids=null_text_ids,
                    img_ids=null_image_id,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                _, ref_k_list, ref_v_list, _ = cache.get()


                latents = utils.adain(latents, style_latent_)
                latents = latents.detach()
                optimizer = torch.optim.Adam([latents.requires_grad_()], lr=lr)
                optimizer = accelerator.prepare(optimizer)

                for _ in range(iters):
                    cache.clear()
                    controller.step()
                    optimizer.zero_grad()
                    _ = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=null_guidance,
                        pooled_projections=null_pooled_embeds,
                        encoder_hidden_states=null_embeds,
                        txt_ids=null_text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    q_list, _, _, self_out_list = cache.get()
                    ref_self_out_list = [
                        F.scaled_dot_product_attention(
                            q,
                            ref_k,
                            ref_v,
                        )
                        for q, ref_k, ref_v in zip(q_list, ref_k_list, ref_v_list)
                    ]
                    style_loss = sum(
                        [
                            loss_fn(self_out, ref_self_out.detach())
                            for self_out, ref_self_out in zip(
                                self_out_list, ref_self_out_list
                            )
                        ]
                    )
                    loss = style_loss
                    accelerator.backward(loss)
                    # loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss=loss.item(), time=t.item())
        torch.cuda.empty_cache()
        latents = latents.detach()
        return self.latent2image(latents, height, width)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
