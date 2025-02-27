# Attention Distillation: A Unified Approach to Visual Characteristics Transfer


### [Project Page](https://xxx.github.io) &ensp; [Paper](https://xxx.pdf)
![alt text](assets/1.jpg)

## Setup

This code was tested with Python 3.10, Pytorch 2.5 and Diffusers 0.32.

## Examples
### Texture Synthesis
- See [**Texture Synthesis**] part of [ad] notebook for generating texture images using SD1.5.

![alt text](assets/2.jpg)

### Style/Appearance Transfer
- See [**Style/Appearance Transfer**] part of [ad] notebook for style/appearance transfer using SD1.5.

![alt text](assets/3.jpg)

### Style-specific T2I Generation
- See [**Style-specific T2I Generation**] part of [ad] notebook for style-specific T2I generation using SD1.5 or SDXL.

![alt text](assets/4.jpg)

[ad]: ad.ipynb


### VAE Finetuning

```bash
python train_vae.py \
    --image_path=/path/to/image \
    --vae_model_path=/path/to/vae
```
