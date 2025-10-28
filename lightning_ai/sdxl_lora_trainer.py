"""Lightning AI oriented SDXL LoRA trainer."""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor2_0,
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnAddedKVProcessor2_0,
    LoRAAttnProcessor2_0,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.peft_utils import convert_peft_state_dict_to_diffusers, get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from peft import LoraConfig, get_peft_model
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "El paquete `peft` es necesario para entrenar LoRAs en los text encoders. "
        "Instálalo con `uv pip install peft`."
    ) from exc


LIGHTNING_ROOT = Path("/teamspace/studios/this_studio").resolve()
LOGGER = get_logger(__name__)


@dataclass
class TrainingConfig:
    dataset_metadata: Path
    images_root: Path
    output_dir: Path
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation: int = 1
    num_repeats: int = 1
    resolution: int = 1024
    network_rank: int = 64
    network_alpha: int = 128
    unet_lr: float = 1e-4
    text_encoder_lr: float = 1e-5
    train_text_encoders: bool = True
    mixed_precision: str = "fp16"
    seed: Optional[int] = None
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_model_name_or_path: Optional[str] = None

    def normalised_paths(self) -> "TrainingConfig":
        def _resolve(path: Path) -> Path:
            return path if path.is_absolute() else (LIGHTNING_ROOT / path).resolve()

        return TrainingConfig(
            dataset_metadata=_resolve(self.dataset_metadata),
            images_root=_resolve(self.images_root),
            output_dir=_resolve(self.output_dir),
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            gradient_accumulation=self.gradient_accumulation,
            num_repeats=self.num_repeats,
            resolution=self.resolution,
            network_rank=self.network_rank,
            network_alpha=self.network_alpha,
            unet_lr=self.unet_lr,
            text_encoder_lr=self.text_encoder_lr,
            train_text_encoders=self.train_text_encoders,
            mixed_precision=self.mixed_precision,
            seed=self.seed,
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            vae_model_name_or_path=self.vae_model_name_or_path,
        )


class JSONCaptionDataset(Dataset):
    def __init__(self, metadata_path: Path, images_root: Path, resolution: int) -> None:
        self.images_root = images_root
        self.resolution = resolution
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        with metadata_path.open("r", encoding="utf-8") as handle:
            self.entries = [json.loads(line) for line in handle if line.strip()]

        if not self.entries:
            raise ValueError(f"No se encontraron datos en {metadata_path}.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.entries[idx]
        image_path = self.images_root / record["file"]
        caption = record["prompt"]

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        return {"pixel_values": pixel_values, "prompt": caption}


def collate_examples(examples: Iterable[Dict[str, object]]) -> Dict[str, object]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    return {"pixel_values": pixel_values, "prompts": prompts}


def calculate_total_steps(num_images: int, num_repeats: int, num_epochs: int, batch_size: int) -> int:
    return math.ceil((num_images * num_repeats * num_epochs) / batch_size)


def setup_unet_lora_layers(pipe: StableDiffusionXLPipeline, rank: int, alpha: int) -> List[torch.nn.Parameter]:
    lora_attn_procs = {}
    for name, attn_processor in pipe.unet.attn_processors.items():
        if not isinstance(attn_processor, (AttnProcessor, AttnProcessor2_0, AttnAddedKVProcessor2_0)):
            continue

        if "mid_block" in name:
            hidden_size = pipe.unet.config.block_out_channels[-1]
        elif "up_blocks" in name:
            block_id = int(name.split(".")[1])
            hidden_size = pipe.unet.config.block_out_channels[::-1][block_id]
        else:
            block_id = int(name.split(".")[1])
            hidden_size = pipe.unet.config.block_out_channels[block_id]

        cross_attention_dim = None if name.endswith("attn1.processor") else pipe.unet.config.cross_attention_dim

        if name.endswith("attn2.processor"):
            lora_attn_procs[name] = LoRAAttnAddedKVProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
                network_alpha=alpha,
            )
        else:
            lora_attn_procs[name] = LoRAAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
                network_alpha=alpha,
            )

    pipe.unet.set_attn_processor(lora_attn_procs)

    params: List[torch.nn.Parameter] = []
    for module in lora_attn_procs.values():
        params.extend(list(module.parameters()))
    for param in params:
        param.requires_grad_(True)
    return params


def setup_text_encoder_lora(model: torch.nn.Module, rank: int, alpha: int) -> torch.nn.Module:
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        bias="none",
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.enable_input_require_grads()
    return peft_model


def train(config: TrainingConfig) -> None:
    accelerator = Accelerator(
        mixed_precision=None if config.mixed_precision == "no" else config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation,
    )
    LOGGER.info("Configuración final: %s", json.dumps(asdict(config), indent=2, default=str))

    dataset = JSONCaptionDataset(config.dataset_metadata, config.images_root, config.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_examples,
        num_workers=min(8, os.cpu_count() or 1),
    )

    num_images = len(dataset)
    total_steps = calculate_total_steps(num_images, config.num_repeats, config.num_epochs, config.batch_size)
    LOGGER.info(
        "Dataset: %d imágenes | repeats: %d | epochs: %d => %d steps",
        num_images,
        config.num_repeats,
        config.num_epochs,
        total_steps,
    )

    if config.mixed_precision == "fp16":
        model_dtype = torch.float16
        variant = "fp16"
    elif config.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
        variant = None
    else:
        model_dtype = torch.float32
        variant = None

    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        torch_dtype=model_dtype,
        variant=variant,
    )

    if config.vae_model_name_or_path:
        pipe.vae = pipe.vae.from_pretrained(config.vae_model_name_or_path, torch_dtype=model_dtype)

    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    if is_xformers_available():  # pragma: no cover
        pipe.enable_xformers_memory_efficient_attention()

    unet_params = setup_unet_lora_layers(pipe, config.network_rank, config.network_alpha)

    text_encoder_params: List[torch.nn.Parameter] = []
    if config.train_text_encoders:
        pipe.text_encoder = setup_text_encoder_lora(pipe.text_encoder, config.network_rank, config.network_alpha)
        pipe.text_encoder_2 = setup_text_encoder_lora(pipe.text_encoder_2, config.network_rank, config.network_alpha)
        text_encoder_params.extend(list(pipe.text_encoder.parameters()))
        text_encoder_params.extend(list(pipe.text_encoder_2.parameters()))

    pipe.unet.to(accelerator.device)
    pipe.text_encoder.to(accelerator.device)
    pipe.text_encoder_2.to(accelerator.device)
    pipe.vae.to(accelerator.device)

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    if config.seed is not None:
        torch.manual_seed(config.seed)

    optimizer_groups = [{"params": [p for p in unet_params if p.requires_grad], "lr": config.unet_lr}]
    if text_encoder_params:
        optimizer_groups.append({"params": [p for p in text_encoder_params if p.requires_grad], "lr": config.text_encoder_lr})

    optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 10, 1),
        num_training_steps=total_steps,
    )

    unet, text_encoder_one, text_encoder_two, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        pipe.unet, pipe.text_encoder, pipe.text_encoder_2, optimizer, lr_scheduler, dataloader
    )

    pipe.unet = unet
    pipe.text_encoder = text_encoder_one
    pipe.text_encoder_2 = text_encoder_two

    pipe.unet.train()
    if config.train_text_encoders:
        pipe.text_encoder.train()
        pipe.text_encoder_2.train()

    for epoch in range(config.num_epochs):
        for repeat in range(config.num_repeats):
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(unet):
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    noise = torch.randn_like(pixel_values)
                    timesteps = torch.randint(
                        0,
                        pipe.scheduler.config.num_train_timesteps,
                        (pixel_values.shape[0],),
                        device=accelerator.device,
                        dtype=torch.long,
                    )

                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * 0.18215
                    noise = noise.to(latents.dtype)
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                    prompt_embeds, pooled_prompt_embeds = pipe.encode_prompt(
                        batch["prompts"],
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )

                    add_time_ids = pipe._get_add_time_ids(
                        original_size=(config.resolution, config.resolution),
                        crops_coords_top_left=(0, 0),
                        target_size=(config.resolution, config.resolution),
                        dtype=prompt_embeds.dtype,
                    )
                    add_time_ids = add_time_ids.to(accelerator.device)
                    add_time_ids = add_time_ids.repeat(pixel_values.shape[0], 1)

                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                    ).sample

                    if pipe.scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif pipe.scheduler.config.prediction_type == "v_prediction":
                        target = pipe.scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unsupported prediction type: {pipe.scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.is_main_process and step % 10 == 0:
                    log_loss = accelerator.gather(loss.detach()).mean().item()
                    LOGGER.info(
                        "Epoch %d/%d | Repeat %d/%d | Step %d | Loss %.4f",
                        epoch + 1,
                        config.num_epochs,
                        repeat + 1,
                        config.num_repeats,
                        step,
                        log_loss,
                    )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(config.output_dir / "accelerate_state")

        unet_state_dict = accelerator.unwrap_model(unet).get_attn_procs_state_dict()
        text_encoder_lora_state = {}
        text_encoder2_lora_state = {}

        if config.train_text_encoders:
            te1 = accelerator.unwrap_model(text_encoder_one)
            te2 = accelerator.unwrap_model(text_encoder_two)
            text_encoder_lora_state = convert_peft_state_dict_to_diffusers(
                get_peft_model_state_dict(te1),
                pipeline=pipe,
                adapter_name="default",
            )
            text_encoder2_lora_state = convert_peft_state_dict_to_diffusers(
                get_peft_model_state_dict(te2),
                pipeline=pipe,
                adapter_name="default",
            )

        pipe.save_lora_weights(
            config.output_dir,
            unet_lora_layers=unet_state_dict,
            text_encoder_lora_layers=text_encoder_lora_state or None,
            text_encoder_2_lora_layers=text_encoder2_lora_state or None,
            safe_serialization=True,
        )
        LOGGER.info("Pesos LoRA guardados en %s", config.output_dir)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Entrena adaptadores LoRA para SDXL en Lightning AI.")
    parser.add_argument("--dataset-metadata", type=Path, required=True, help="Archivo JSONL con campos `file` y `prompt`.")
    parser.add_argument("--images-root", type=Path, required=True, help="Directorio que contiene las imágenes del dataset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directorio donde se guardarán los pesos LoRA.")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--network-rank", type=int, default=64)
    parser.add_argument("--network-alpha", type=int, default=128)
    parser.add_argument("--unet-lr", type=float, default=1e-4)
    parser.add_argument("--text-encoder-lr", type=float, default=1e-5)
    parser.add_argument("--no-text-encoders", action="store_true", help="Desactiva el entrenamiento de los text encoders.")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrained-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae", type=str, default=None)

    args = parser.parse_args()
    config = TrainingConfig(
        dataset_metadata=args.dataset_metadata,
        images_root=args.images_root,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        num_repeats=args.num_repeats,
        resolution=args.resolution,
        network_rank=args.network_rank,
        network_alpha=args.network_alpha,
        unet_lr=args.unet_lr,
        text_encoder_lr=args.text_encoder_lr,
        train_text_encoders=not args.no_text_encoders,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        pretrained_model_name_or_path=args.pretrained_model,
        vae_model_name_or_path=args.vae,
    )
    return config.normalised_paths()


if __name__ == "__main__":  # pragma: no cover
    train(parse_args())
