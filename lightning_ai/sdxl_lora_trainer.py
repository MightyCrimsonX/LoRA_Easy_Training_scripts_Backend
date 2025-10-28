"""Lightning AI oriented SDXL LoRA trainer."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.models import attention_processor as attention_processors
from diffusers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from diffusers.utils.import_utils import is_xformers_available
try:
    from diffusers.utils.peft_utils import convert_peft_state_dict_to_diffusers, get_peft_model_state_dict
except ImportError:  # pragma: no cover - older diffusers versions
    from diffusers.utils.peft_utils import get_peft_model_state_dict

    try:
        from diffusers.loaders.peft import (
            convert_peft_state_dict_to_diffusers as _convert_peft_state_dict_to_diffusers,
        )
    except Exception:  # pragma: no cover - very old diffusers versions

        def convert_peft_state_dict_to_diffusers(state_dict: Dict[str, torch.Tensor], *args, **kwargs):
            warnings.warn(
                "diffusers no dispone de `convert_peft_state_dict_to_diffusers`; se devolverá el estado sin convertir. "
                "Actualiza diffusers para obtener compatibilidad total.",
                ImportWarning,
            )
            return state_dict

    else:

        def convert_peft_state_dict_to_diffusers(state_dict: Dict[str, torch.Tensor], *args, **kwargs):
            return _convert_peft_state_dict_to_diffusers(state_dict, *args, **kwargs)
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


AttnProcessor = attention_processors.AttnProcessor
AttnProcessor2_0 = getattr(attention_processors, "AttnProcessor2_0", AttnProcessor)
AttnAddedKVProcessor2_0 = getattr(
    attention_processors,
    "AttnAddedKVProcessor2_0",
    attention_processors.AttnAddedKVProcessor,
)
LoRAAttnProcessor2_0 = getattr(
    attention_processors,
    "LoRAAttnProcessor2_0",
    attention_processors.LoRAAttnProcessor,
)
LoRAAttnAddedKVProcessor2_0 = getattr(
    attention_processors,
    "LoRAAttnAddedKVProcessor2_0",
    attention_processors.LoRAAttnAddedKVProcessor,
)

LIGHTNING_ROOT = Path("/teamspace/studios/this_studio").resolve()
LOGGER = get_logger(__name__)

OPTIMIZER_CHOICES = {
    "adamw",
    "adamw8bit",
    "prodigy",
    "dadaptation",
    "dadaptadam",
    "dadaptlion",
    "lion",
    "sgdnesterov",
    "sgdnesterov8bit",
    "adafactor",
    "came",
}

SCHEDULER_CHOICES = {
    "constant",
    "cosine",
    "cosine_with_restarts",
    "constant_with_warmup",
    "linear",
    "polynomial",
    "rex",
}

BASE_MODEL_CHOICES: Dict[str, str] = {
    "Pony Diffusion V6 XL": "PonyDiffusion/Pony-Diffusion-V6-XL",
    "Animagine XL V3": "cagliostrolab/animagine-xl-3.0",
    "animagine_4.0_zero": "cagliostrolab/animagine-xl-4.0-zero",
    "Illustrious_0.1": "stabilityai/illustrious-xl-0.1",
    "Illustrious_2.0": "stabilityai/illustrious-xl-2.0",
    "NoobAI-XL0.75": "NoobAI/NoobAI-XL0.75",
    "Stable Diffusion XL 1.0 base": "stabilityai/stable-diffusion-xl-base-1.0",
    "NoobAIXL0_75vpred": "NoobAI/NoobAIXL0.75-vPred",
    "RouWei_v080vpred": "RouWei/RouWei-v0.80-vPred",
}

RECOMMENDED_OPTIMIZER_SETTINGS: Dict[str, Dict[str, object]] = {
    "adafactor": {
        "optimizer_kwargs": {
            "scale_parameter": False,
            "relative_step": False,
            "warmup_init": False,
        }
    },
    "adamw8bit": {
        "weight_decay": 0.1,
        "betas": (0.9, 0.99),
    },
    "prodigy": {
        "weight_decay": 0.01,
        "betas": (0.9, 0.99),
        "optimizer_kwargs": {
            "decouple": True,
            "use_bias_correction": False,
            "safeguard_warmup": True,
            "full_precision": True,
        },
    },
    "came": {
        "weight_decay": 0.04,
    },
}


@dataclass
class TrainingConfig:
    dataset_dir: Path
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
    base_model: Optional[str] = None
    vae_model_name_or_path: Optional[str] = None
    optimizer_type: str = "adamw"
    weight_decay: float = 1e-2
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    optimizer_momentum: float = 0.9
    scheduler_type: str = "cosine"
    lr_warmup_steps: Optional[int] = None
    scheduler_first_cycle_steps: Optional[int] = None
    scheduler_cycle_multiplier: float = 1.0
    scheduler_gamma: float = 1.0
    scheduler_min_lr: float = 1e-6
    scheduler_d: float = 0.9
    scheduler_power: float = 1.0
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    shuffle_tags: bool = False
    activation_tags: Sequence[str] = field(default_factory=tuple)
    use_optimizer_recommended_args: bool = False
    lora_name: str = "sdxl_lora"

    def normalised_paths(self) -> "TrainingConfig":
        def _resolve(path: Path) -> Path:
            return path if path.is_absolute() else (LIGHTNING_ROOT / path).resolve()

        return TrainingConfig(
            dataset_dir=_resolve(self.dataset_dir),
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
            optimizer_type=self.optimizer_type,
            weight_decay=self.weight_decay,
            optimizer_beta1=self.optimizer_beta1,
            optimizer_beta2=self.optimizer_beta2,
            optimizer_eps=self.optimizer_eps,
            optimizer_momentum=self.optimizer_momentum,
            scheduler_type=self.scheduler_type,
            lr_warmup_steps=self.lr_warmup_steps,
            scheduler_first_cycle_steps=self.scheduler_first_cycle_steps,
            scheduler_cycle_multiplier=self.scheduler_cycle_multiplier,
            scheduler_gamma=self.scheduler_gamma,
            scheduler_min_lr=self.scheduler_min_lr,
            scheduler_d=self.scheduler_d,
            scheduler_power=self.scheduler_power,
            optimizer_kwargs=dict(self.optimizer_kwargs or {}),
            scheduler_kwargs=dict(self.scheduler_kwargs or {}),
            shuffle_tags=self.shuffle_tags,
            activation_tags=tuple(self.activation_tags),
            use_optimizer_recommended_args=self.use_optimizer_recommended_args,
            base_model=self.base_model,
            lora_name=self.lora_name,
        )

    def resolved_pretrained_model(self) -> str:
        base_model_key = (self.base_model or "").strip()
        if base_model_key:
            try:
                return BASE_MODEL_CHOICES[base_model_key]
            except KeyError as exc:  # pragma: no cover - guard against inconsistent config
                raise ValueError(f"Modelo base desconocido: {base_model_key}") from exc

        candidate = (self.pretrained_model_name_or_path or "").strip()
        if candidate in BASE_MODEL_CHOICES:
            return BASE_MODEL_CHOICES[candidate]
        return candidate

    def resolved_lora_weight_name(self) -> str:
        raw_name = (self.lora_name or "").strip()
        if not raw_name:
            raw_name = "sdxl_lora"
        if not raw_name.endswith(".safetensors"):
            raw_name = f"{raw_name}.safetensors"
        return raw_name


class FolderCaptionDataset(Dataset):
    _TAG_SPLITTER = re.compile(r"[,\n]+")
    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

    def __init__(
        self,
        dataset_dir: Path,
        resolution: int,
        activation_tags: Sequence[str] | None = None,
        shuffle_tags: bool = False,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.resolution = resolution
        self.shuffle_tags = shuffle_tags
        self.activation_tags = tuple(tag.strip() for tag in (activation_tags or ()) if tag.strip())
        self._activation_lookup = {tag for tag in self.activation_tags}

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"El directorio del dataset no existe: {dataset_dir}")
        if not self.dataset_dir.is_dir():
            raise NotADirectoryError(f"Se esperaba un directorio para el dataset y se recibió: {dataset_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.entries: List[Dict[str, object]] = []
        for image_path in sorted(self.dataset_dir.rglob("*")):
            if image_path.suffix.lower() not in self._IMAGE_EXTENSIONS:
                continue

            caption_path = image_path.with_suffix(".txt")
            if not caption_path.exists():
                raise FileNotFoundError(
                    f"No se encontró el archivo de texto para la imagen {image_path.name} en {caption_path}."
                )

            caption_raw = caption_path.read_text(encoding="utf-8").strip()
            if not caption_raw:
                raise ValueError(f"El archivo de texto {caption_path} está vacío.")

            tags = [tag.strip() for tag in self._TAG_SPLITTER.split(caption_raw) if tag.strip()]
            if not tags:
                raise ValueError(
                    f"El archivo de texto {caption_path} no contiene tags separados por comas o saltos de línea."
                )

            self.entries.append({"image_path": image_path, "tags": tags})

        if not self.entries:
            raise ValueError(f"No se encontraron imágenes válidas en {dataset_dir}.")

    def __len__(self) -> int:
        return len(self.entries)

    def _build_prompt(self, tags: Sequence[str]) -> str:
        ordered_tags = list(tags)
        if self.shuffle_tags and len(ordered_tags) > 1:
            ordered_tags = random.sample(ordered_tags, len(ordered_tags))

        if self.activation_tags:
            remaining = [tag for tag in ordered_tags if tag not in self._activation_lookup]
            ordered_tags = list(self.activation_tags) + remaining

        return ", ".join(ordered_tags)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        record = self.entries[idx]
        image_path: Path = record["image_path"]  # type: ignore[assignment]
        tags: Sequence[str] = record["tags"]  # type: ignore[assignment]

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        prompt = self._build_prompt(tags)
        return {"pixel_values": pixel_values, "prompt": prompt}


def collate_examples(examples: Iterable[Dict[str, object]]) -> Dict[str, object]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    return {"pixel_values": pixel_values, "prompts": prompts}


def calculate_total_steps(num_images: int, num_repeats: int, num_epochs: int, batch_size: int) -> int:
    return math.ceil((num_images * num_repeats * num_epochs) / batch_size)


def create_optimizer(config: TrainingConfig, optimizer_groups: List[Dict[str, object]]) -> torch.optim.Optimizer:
    name = config.optimizer_type.lower()
    if name not in OPTIMIZER_CHOICES:
        raise ValueError(f"Optimizer '{config.optimizer_type}' no es compatible.")

    betas = (config.optimizer_beta1, config.optimizer_beta2)
    weight_decay = config.weight_decay
    eps = config.optimizer_eps
    momentum = config.optimizer_momentum
    optimizer_kwargs = dict(config.optimizer_kwargs or {})

    if config.use_optimizer_recommended_args:
        recommended = RECOMMENDED_OPTIMIZER_SETTINGS.get(name)
        if recommended:
            weight_decay = recommended.get("weight_decay", weight_decay)  # type: ignore[assignment]
            betas = recommended.get("betas", betas)  # type: ignore[assignment]
            eps = recommended.get("eps", eps)  # type: ignore[assignment]
            momentum = recommended.get("momentum", momentum)  # type: ignore[assignment]
            rec_kwargs = dict(recommended.get("optimizer_kwargs", {}))
            rec_kwargs.update(optimizer_kwargs)
            optimizer_kwargs = rec_kwargs
            LOGGER.info(
                "Usando argumentos recomendados para %s: %s",
                name,
                json.dumps(
                    {
                        k: v
                        for k, v in {
                            "weight_decay": weight_decay,
                            "betas": list(betas) if isinstance(betas, tuple) else betas,
                            "eps": eps,
                            "momentum": momentum,
                            "extras": optimizer_kwargs,
                        }.items()
                        if v not in ({}, None)
                    },
                    default=str,
                ),
            )
        else:
            LOGGER.warning(
                "No hay argumentos recomendados registrados para el optimizador %s. Ignorando la bandera.",
                name,
            )
    if name in {"lion", "dadaptlion"} and config.optimizer_beta2 == 0.999:
        betas = (config.optimizer_beta1, 0.99)

    if name == "adamw":
        return torch.optim.AdamW(
            optimizer_groups,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **optimizer_kwargs,
        )

    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise RuntimeError(
                "El optimizador AdamW8bit requiere el paquete `bitsandbytes`. "
                "Instálalo con `uv pip install bitsandbytes`."
            ) from exc

        return bnb.optim.AdamW8bit(
            optimizer_groups,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **optimizer_kwargs,
        )

    if name == "prodigy":
        try:
            from prodigyopt import Prodigy  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise RuntimeError(
                "El optimizador Prodigy requiere el paquete `prodigyopt`. "
                "Instálalo con `uv pip install prodigyopt`."
            ) from exc

        prodigy_kwargs = {
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        prodigy_kwargs.update(optimizer_kwargs)
        return Prodigy(optimizer_groups, **prodigy_kwargs)

    if name == "dadaptation":
        from dadaptation import DAdaptAdaGrad

        ada_kwargs = {
            "eps": eps,
            "weight_decay": weight_decay,
        }
        ada_kwargs.update(optimizer_kwargs)
        return DAdaptAdaGrad(optimizer_groups, **ada_kwargs)

    if name == "dadaptadam":
        from dadaptation import DAdaptAdam

        dadam_kwargs = {
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "decouple": True,
        }
        dadam_kwargs.update(optimizer_kwargs)
        return DAdaptAdam(optimizer_groups, **dadam_kwargs)

    if name == "dadaptlion":
        from dadaptation import DAdaptLion

        dlion_kwargs = {
            "betas": betas,
            "weight_decay": weight_decay,
        }
        dlion_kwargs.update(optimizer_kwargs)
        return DAdaptLion(optimizer_groups, **dlion_kwargs)

    if name == "lion":
        try:
            from lion_pytorch import Lion  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise RuntimeError(
                "El optimizador Lion requiere el paquete `lion-pytorch`. "
                "Instálalo con `uv pip install lion-pytorch`."
            ) from exc

        lion_kwargs = {
            "betas": betas,
            "weight_decay": weight_decay,
        }
        lion_kwargs.update(optimizer_kwargs)
        return Lion(optimizer_groups, **lion_kwargs)

    if name == "sgdnesterov":
        sgd_kwargs = dict(optimizer_kwargs)
        momentum = sgd_kwargs.pop("momentum", momentum)
        return torch.optim.SGD(
            optimizer_groups,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
            **sgd_kwargs,
        )

    if name == "sgdnesterov8bit":
        try:
            import bitsandbytes as bnb  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise RuntimeError(
                "El optimizador SGDNesterov8bit requiere `bitsandbytes`. "
                "Instálalo con `uv pip install bitsandbytes`."
            ) from exc

        sgd8_kwargs = dict(optimizer_kwargs)
        momentum = sgd8_kwargs.pop("momentum", momentum)
        return bnb.optim.SGD8bit(
            optimizer_groups,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
            **sgd8_kwargs,
        )

    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise RuntimeError(
                "El optimizador Adafactor requiere el paquete `transformers`. "
                "Instálalo con `uv pip install transformers`."
            ) from exc

        adafactor_kwargs: Dict[str, Any] = {
            "lr": config.unet_lr,
            "weight_decay": weight_decay,
        }
        adafactor_kwargs.update(optimizer_kwargs)
        return Adafactor(optimizer_groups, **adafactor_kwargs)

    if name == "came":
        try:
            from custom_scheduler.LoraEasyCustomOptimizer.came import CAME
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise RuntimeError(
                "El optimizador CAME requiere las utilidades personalizadas incluidas en este repositorio."
            ) from exc

        came_kwargs = {"weight_decay": weight_decay, "weight_decouple": True}
        came_kwargs.update(optimizer_kwargs)
        return CAME(optimizer_groups, **came_kwargs)

    raise AssertionError("Ruta de optimizador no cubierta")


def create_lr_scheduler(
    config: TrainingConfig, optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    name = config.scheduler_type.lower()
    if name not in SCHEDULER_CHOICES:
        raise ValueError(f"Scheduler '{config.scheduler_type}' no es compatible.")

    warmup_steps = config.lr_warmup_steps
    scheduler_kwargs = dict(config.scheduler_kwargs or {})

    if name == "constant":
        return get_constant_schedule(optimizer)

    if name == "constant_with_warmup":
        warmup = warmup_steps or max(total_steps // 10, 1)
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)

    if name == "cosine":
        warmup = warmup_steps or max(total_steps // 10, 1)
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )

    if name == "linear":
        warmup = warmup_steps or max(total_steps // 10, 1)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )

    if name == "polynomial":
        warmup = warmup_steps or max(total_steps // 10, 1)
        power = scheduler_kwargs.pop("power", config.scheduler_power)
        lr_end = scheduler_kwargs.pop("lr_end", config.scheduler_min_lr)
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
            lr_end=lr_end,
            power=power,
        )

    if name == "cosine_with_restarts":
        from custom_scheduler.LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts import (
            CosineAnnealingWarmRestarts,
        )

        first_cycle = scheduler_kwargs.pop(
            "first_cycle_max_steps", config.scheduler_first_cycle_steps or total_steps
        )
        warmup = warmup_steps or max(first_cycle // 10, 1)
        return CosineAnnealingWarmRestarts(
            optimizer,
            gamma=scheduler_kwargs.pop("gamma", config.scheduler_gamma),
            cycle_multiplier=scheduler_kwargs.pop("cycle_multiplier", config.scheduler_cycle_multiplier),
            first_cycle_max_steps=first_cycle,
            min_lr=scheduler_kwargs.pop("min_lr", config.scheduler_min_lr),
            warmup_steps=warmup,
            **scheduler_kwargs,
        )

    if name == "rex":
        from custom_scheduler.LoraEasyCustomOptimizer.RexAnnealingWarmRestarts import (
            RexAnnealingWarmRestarts,
        )

        first_cycle = scheduler_kwargs.pop(
            "first_cycle_max_steps", config.scheduler_first_cycle_steps or total_steps
        )
        warmup = warmup_steps or max(first_cycle // 10, 1)
        return RexAnnealingWarmRestarts(
            optimizer,
            gamma=scheduler_kwargs.pop("gamma", config.scheduler_gamma),
            cycle_multiplier=scheduler_kwargs.pop("cycle_multiplier", config.scheduler_cycle_multiplier),
            first_cycle_max_steps=first_cycle,
            min_lr=scheduler_kwargs.pop("min_lr", config.scheduler_min_lr),
            warmup_steps=warmup,
            d=scheduler_kwargs.pop("d", config.scheduler_d),
            **scheduler_kwargs,
        )

    raise AssertionError("Ruta de scheduler no cubierta")


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

    dataset = FolderCaptionDataset(
        config.dataset_dir,
        config.resolution,
        activation_tags=config.activation_tags,
        shuffle_tags=config.shuffle_tags,
    )
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
        "Dataset (%s): %d imágenes | repeats: %d | epochs: %d => %d steps | shuffle_tags=%s | activation_tags=%s",
        config.dataset_dir,
        num_images,
        config.num_repeats,
        config.num_epochs,
        total_steps,
        config.shuffle_tags,
        ", ".join(config.activation_tags) if config.activation_tags else "(ninguno)",
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

    resolved_model_name = config.resolved_pretrained_model()
    LOGGER.info(
        "Modelo base seleccionado: %s",
        f"{config.base_model} -> {resolved_model_name}" if config.base_model else resolved_model_name,
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        resolved_model_name,
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

    optimizer = create_optimizer(config, optimizer_groups)
    lr_scheduler = create_lr_scheduler(config, optimizer, total_steps)

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

        weight_name = config.resolved_lora_weight_name()
        pipe.save_lora_weights(
            config.output_dir,
            unet_lora_layers=unet_state_dict,
            text_encoder_lora_layers=text_encoder_lora_state or None,
            text_encoder_2_lora_layers=text_encoder2_lora_state or None,
            safe_serialization=True,
            weight_name=weight_name,
        )
        LOGGER.info("Pesos LoRA guardados en %s/%s", config.output_dir, weight_name)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Entrena adaptadores LoRA para SDXL en Lightning AI.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directorio que contiene las imágenes y archivos .txt emparejados.",
    )
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
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        choices=sorted(BASE_MODEL_CHOICES),
        help=(
            "Nombre del modelo base predefinido a usar. Si se especifica, tiene prioridad sobre --pretrained-model."
        ),
    )
    parser.add_argument("--vae", type=str, default=None)
    parser.add_argument("--optimizer", type=str.lower, choices=sorted(OPTIMIZER_CHOICES), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--optimizer-beta1", type=float, default=0.9)
    parser.add_argument("--optimizer-beta2", type=float, default=0.999)
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--optimizer-momentum", type=float, default=0.9)
    parser.add_argument(
        "--use-optimizer-recommended-args",
        action="store_true",
        help="Aplica automáticamente los valores recomendados para el optimizador seleccionado.",
    )
    parser.add_argument("--lr-scheduler", type=str.lower, choices=sorted(SCHEDULER_CHOICES), default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument("--scheduler-first-cycle-steps", type=int, default=None)
    parser.add_argument("--scheduler-cycle-multiplier", type=float, default=1.0)
    parser.add_argument("--scheduler-gamma", type=float, default=1.0)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--scheduler-d", type=float, default=0.9)
    parser.add_argument("--scheduler-power", type=float, default=1.0)
    parser.add_argument(
        "--shuffle-tags",
        action="store_true",
        help="Baraja aleatoriamente los tags de cada prompt manteniendo primero los activadores.",
    )
    parser.add_argument(
        "--activation-tags",
        type=str,
        default="",
        help="Cadena de tags de activación separados por comas que se antepondrán a cada prompt.",
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default="sdxl_lora",
        help="Nombre base del archivo .safetensors resultante. Se añadirá la extensión si falta.",
    )

    args = parser.parse_args()
    activation_tags = [tag.strip() for tag in re.split(r"[,\n]+", args.activation_tags) if tag.strip()]
    config = TrainingConfig(
        dataset_dir=args.dataset_dir,
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
        base_model=args.base_model,
        vae_model_name_or_path=args.vae,
        optimizer_type=args.optimizer,
        weight_decay=args.weight_decay,
        optimizer_beta1=args.optimizer_beta1,
        optimizer_beta2=args.optimizer_beta2,
        optimizer_eps=args.optimizer_eps,
        optimizer_momentum=args.optimizer_momentum,
        use_optimizer_recommended_args=args.use_optimizer_recommended_args,
        scheduler_type=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        scheduler_first_cycle_steps=args.scheduler_first_cycle_steps,
        scheduler_cycle_multiplier=args.scheduler_cycle_multiplier,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_min_lr=args.scheduler_min_lr,
        scheduler_d=args.scheduler_d,
        scheduler_power=args.scheduler_power,
        shuffle_tags=args.shuffle_tags,
        activation_tags=tuple(activation_tags),
        lora_name=args.lora_name,
    )
    return config.normalised_paths()


if __name__ == "__main__":  # pragma: no cover
    train(parse_args())
