import argparse
import os

import torch
from diffusers import AutoencoderKL
from torch import nn
from torch.optim import Adam
from utils import load_image, save_image


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(
        device, dtype=torch.float32
    )
    vae.requires_grad_(False)

    image = load_image(args.image_path, size=(512, 512)).to(device, dtype=torch.float32)
    image = image * 2 - 1
    save_image(image / 2 + 0.5, f"{args.out_dir}/ori_image.png")

    latents = vae.encode(image)["latent_dist"].mean
    save_image(latents, f"{args.out_dir}/latents.png")

    rec_image = vae.decode(latents, return_dict=False)[0]
    save_image(rec_image / 2 + 0.5, f"{args.out_dir}/rec_image.png")

    for param in vae.decoder.parameters():
        param.requires_grad = True

    loss_fn = nn.L1Loss()
    optimizer = Adam(vae.decoder.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        reconstructed = vae.decode(latents, return_dict=False)[0]
        loss = loss_fn(reconstructed, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss.item()}")

    rec_image = vae.decode(latents, return_dict=False)[0]
    save_image(rec_image / 2 + 0.5, f"{args.out_dir}/trained_rec_image.png")
    vae.save_pretrained(
        f"{args.out_dir}/trained_vae_{os.path.basename(args.image_path)}"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a VAE with given image and settings."
    )

    # Add arguments
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./trained_vae/",
        help="Output directory to save results",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        required=True,
        help="Path to the pretrained VAE model",
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=75, help="Number of training epochs"
    )

    args = parser.parse_args()
    main(args)
