"""UI helpers for configuring the Kohya training session from the notebook.

The original notebook embedded all of the ipywidgets setup logic directly in a
cell.  Moving the implementation to this module keeps the notebook tidy while
preserving the same behaviour.  The only requirement from the notebook is to
call :func:`render_quick_training_config` passing ``globals()`` so the widgets
can read and update the shared state exactly like before.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import os
import subprocess
from urllib.parse import urlparse

import ipywidgets as widgets
from IPython.display import Markdown, display
from pathlib import Path


PROJECTS_ROOT = Path("/teamspace/studios/this_studio/lora_projects")
MODELS_ROOT = Path("/teamspace/studios/this_studio/models")


@dataclass(frozen=True)
class _DropdownOption:
    label: str
    value: Any

    def as_tuple(self) -> Tuple[str, Any]:
        return self.label, self.value


def _format_scientific(value: Any) -> str:
    """Return ``value`` formatted in scientific notation when possible."""
    try:
        return format(float(value), ".0e")
    except (TypeError, ValueError):
        return str(value)


def _resolve_precision(default: str, options: Iterable[str]) -> str:
    if default in options:
        return default
    default = str(default)
    if "bf16" in default:
        return "mixed bf16"
    if "fp16" in default:
        return "mixed fp16"
    return next(iter(options))


def render_quick_training_config(namespace: Dict[str, Any]) -> None:
    """Render the interactive configuration UI.

    Parameters
    ----------
    namespace:
        Usually the ``globals()`` dictionary from the notebook.  The widgets
        will read default values from it and write the updated selections back
        into the same mapping so that the following cells behave exactly like
        the original implementation.
    """

    training_model_options = [
        "Pony Diffusion V6 XL",
        "Animagine XL V3",
        "animagine_4.0_zero",
        "Illustrious_0.1",
        "Illustrious_2.0",
        "NoobAI-XL0.75",
        "NoobAI-XL0.5",
        "Stable Diffusion XL 1.0 base",
        "NoobAIXL0_75vpred",
        "RouWei_v080vpred",
    ]

    force_load_diffusers_options = [
        _DropdownOption("❌ No (ckpt)", False),
        _DropdownOption("✅ Sí (diffusers)", True),
    ]

    lr_scheduler_options = [
        "constant",
        "cosine",
        "cosine_with_restarts",
        "constant_with_warmup",
        "rex",
    ]

    precision_options = [
        "full fp16",
        "full bf16",
        "mixed fp16",
        "mixed bf16",
    ]

    optimizer_options = [
        "AdamW8bit",
        "Prodigy",
        "DAdaptation",
        "DadaptAdam",
        "DadaptLion",
        "AdamW",
        "AdaFactor",
        "Came",
    ]

    # Widgets -----------------------------------------------------------------
    base_style = {"description_width": "160px"}
    number_layout = widgets.Layout(width="100%")

    project_name_widget = widgets.Text(
        value=str(namespace.get("project_name", "")),
        description="project_name",
        placeholder="Nombre del proyecto",
        style=base_style,
    )
    training_model_widget = widgets.Dropdown(
        options=training_model_options,
        value=namespace.get("training_model", "Illustrious_2.0"),
        description="training_model",
        style=base_style,
    )
    use_custom_model_widget = widgets.Checkbox(
        value=bool(namespace.get("use_optional_custom_training_model", False)),
        description="usar_modelo_descargado",
        style={"description_width": "160px"},
        indent=False,
    )
    custom_model_widget = widgets.Text(
        value=str(namespace.get("optional_custom_training_model", "")),
        description="modelo_personalizado",
        placeholder="URL o ruta local",
        style=base_style,
        layout=widgets.Layout(width="100%"),
    )
    custom_model_filename_widget = widgets.Text(
        value=str(namespace.get("custom_model_filename", "")),
        description="archivo_descarga",
        placeholder="Opcional (.safetensors)",
        style=base_style,
        layout=widgets.Layout(width="100%"),
    )
    force_load_diffusers_widget = widgets.Dropdown(
        options=[opt.as_tuple() for opt in force_load_diffusers_options],
        value=bool(namespace.get("force_load_diffusers", False)),
        description="force_load_diffusers",
        style=base_style,
    )
    resolution_widget = widgets.IntText(
        value=int(namespace.get("resolution", 1024)),
        description="resolution",
        style=base_style,
        layout=number_layout,
    )
    num_repeats_widget = widgets.IntText(
        value=int(namespace.get("num_repeats", 2)),
        description="num_repeats",
        style=base_style,
        layout=number_layout,
    )
    how_many_widget = widgets.IntText(
        value=int(namespace.get("how_many", 40)),
        description="how_many",
        style=base_style,
        layout=number_layout,
    )
    unet_lr_widget = widgets.Text(
        value=_format_scientific(namespace.get("unet_lr", 1e-4)),
        description="unet_lr",
        style=base_style,
    )
    text_encoder_lr_widget = widgets.Text(
        value=_format_scientific(namespace.get("text_encoder_lr", 5e-5)),
        description="text_encoder_lr",
        style=base_style,
    )
    lr_scheduler_widget = widgets.Dropdown(
        options=lr_scheduler_options,
        value=namespace.get("lr_scheduler", "constant_with_warmup"),
        description="lr_scheduler",
        style=base_style,
    )
    lora_type_widget = widgets.Dropdown(
        options=["LoRA", "LoCon"],
        value=namespace.get("lora_type", "LoRA"),
        description="lora_type",
        style=base_style,
    )
    network_dim_widget = widgets.IntText(
        value=int(namespace.get("network_dim", 16)),
        description="network_dim",
        style=base_style,
        layout=number_layout,
    )
    network_alpha_widget = widgets.IntText(
        value=int(namespace.get("network_alpha", 32)),
        description="network_alpha",
        style=base_style,
        layout=number_layout,
    )
    conv_dim_widget = widgets.IntText(
        value=int(namespace.get("conv_dim", 16)),
        description="conv_dim",
        style=base_style,
        layout=widgets.Layout(width="100%"),
    )
    conv_alpha_widget = widgets.IntText(
        value=int(namespace.get("conv_alpha", 8)),
        description="conv_alpha",
        style=base_style,
        layout=widgets.Layout(width="100%"),
    )
    train_batch_size_widget = widgets.IntText(
        value=int(namespace.get("train_batch_size", 8)),
        description="train_batch_size",
        style=base_style,
        layout=number_layout,
    )
    precision_widget = widgets.Dropdown(
        options=precision_options,
        value=_resolve_precision(namespace.get("precision", "fp16"), precision_options),
        description="precision",
        style=base_style,
    )
    optimizer_widget = widgets.Dropdown(
        options=optimizer_options,
        value=namespace.get("optimizer", "Prodigy"),
        description="optimizer",
        style=base_style,
    )

    status_output = widgets.HTML()
    apply_button = widgets.Button(
        description="Aplicar parámetros",
        button_style="success",
        icon="check",
        layout=widgets.Layout(width="auto", align_self="flex-end"),
    )
    download_button = widgets.Button(
        description="Descargar modelo personalizado",
        button_style="info",
        icon="download",
        layout=widgets.Layout(width="auto", align_self="flex-end"),
    )

    grid_layout = widgets.Layout(
        grid_template_columns="repeat(2, minmax(0, 1fr))",
        grid_gap="12px",
        width="100%",
    )
    basics_grid = widgets.GridBox(
        children=[
            project_name_widget,
            training_model_widget,
            force_load_diffusers_widget,
            resolution_widget,
            num_repeats_widget,
            how_many_widget,
        ],
        layout=grid_layout,
    )
    advanced_grid = widgets.GridBox(
        children=[
            unet_lr_widget,
            text_encoder_lr_widget,
            lr_scheduler_widget,
            lora_type_widget,
            network_dim_widget,
            network_alpha_widget,
            conv_dim_widget,
            conv_alpha_widget,
            train_batch_size_widget,
            precision_widget,
            optimizer_widget,
        ],
        layout=grid_layout,
    )
    custom_model_status = widgets.HTML()

    def apply_params(_=None) -> None:
        try:
            updates = {
                "project_name": project_name_widget.value.strip(),
                "training_model": training_model_widget.value,
                "use_optional_custom_training_model": bool(use_custom_model_widget.value),
                "optional_custom_training_model": custom_model_widget.value.strip(),
                "custom_model_filename": custom_model_filename_widget.value.strip(),
                "force_load_diffusers": bool(force_load_diffusers_widget.value),
                "resolution": int(resolution_widget.value),
                "num_repeats": int(num_repeats_widget.value),
                "how_many": int(how_many_widget.value),
                "unet_lr": float(unet_lr_widget.value),
                "text_encoder_lr": float(text_encoder_lr_widget.value),
                "lr_scheduler": lr_scheduler_widget.value,
                "lora_type": lora_type_widget.value,
                "network_dim": int(network_dim_widget.value),
                "network_alpha": int(network_alpha_widget.value),
                "conv_dim": int(conv_dim_widget.value),
                "conv_alpha": int(conv_alpha_widget.value),
                "train_batch_size": int(train_batch_size_widget.value),
                "precision": precision_widget.value,
                "optimizer": optimizer_widget.value,
            }
        except ValueError as exc:
            status_output.value = f"<b>Error:</b> {exc}"
            return

        namespace.update(updates)
        unet_lr_widget.value = _format_scientific(updates["unet_lr"])
        text_encoder_lr_widget.value = _format_scientific(updates["text_encoder_lr"])
        custom_model_widget.value = updates["optional_custom_training_model"]
        custom_model_filename_widget.value = updates["custom_model_filename"]
        use_custom_model_widget.value = updates["use_optional_custom_training_model"]

        project_name_value = updates["project_name"]
        directory_message = ""
        if project_name_value:
            project_dir = PROJECTS_ROOT / project_name_value
            dataset_dir = project_dir / "dataset"
            try:
                dataset_dir.mkdir(parents=True, exist_ok=True)
                directory_message = f" Directorios preparados en <code>{project_dir}</code>."
            except Exception as exc:  # pragma: no cover - user environment errors
                status_output.value = f"<b>Error al preparar directorios:</b> {exc}"
                return

        status_output.value = f"<b>Parámetros actualizados correctamente.</b>{directory_message}"

    def _update_locon_visibility(_=None) -> None:
        is_locon = lora_type_widget.value.lower() == "locon"
        display_value = None if is_locon else "none"
        conv_dim_widget.layout.display = display_value
        conv_alpha_widget.layout.display = display_value
        conv_dim_widget.disabled = not is_locon
        conv_alpha_widget.disabled = not is_locon

    _update_locon_visibility()

    def _handle_lora_change(change: Dict[str, Any]) -> None:
        if change.get("name") == "value":
            _update_locon_visibility()

    lora_type_widget.observe(_handle_lora_change, names="value")

    apply_button.on_click(apply_params)

    def _infer_filename_from_url(url: str) -> str:
        parsed = urlparse(url)
        candidate = Path(parsed.path).name
        if not candidate:
            candidate = "custom_model.safetensors"
        if not candidate.endswith(".safetensors"):
            candidate = f"{candidate}.safetensors"
        return candidate

    def download_custom_model(_=None) -> None:
        url_or_path = custom_model_widget.value.strip()
        filename_hint = custom_model_filename_widget.value.strip()
        custom_model_status.value = ""

        if not url_or_path:
            custom_model_status.value = "<b>Error:</b> proporciona una URL o ruta válida."
            return

        parsed = urlparse(url_or_path)
        if parsed.scheme in {"http", "https"}:
            target_name = filename_hint or _infer_filename_from_url(url_or_path)
            MODELS_ROOT.mkdir(parents=True, exist_ok=True)
            target_path = MODELS_ROOT / target_name

            command = [
                "aria2c",
                url_or_path,
                "--console-log-level=warn",
                "-c",
                "-s",
                "16",
                "-x",
                "16",
                "-k",
                "10M",
                "-d",
                str(MODELS_ROOT),
                "-o",
                target_name,
            ]

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                custom_model_status.value = f"<b>Error al descargar:</b> {exc}"
                return

            resolved_path = target_path.resolve()
            namespace["optional_custom_training_model"] = str(resolved_path)
            custom_model_widget.value = str(resolved_path)
            namespace["use_optional_custom_training_model"] = True
            use_custom_model_widget.value = True
            apply_params()
            custom_model_status.value = f"<b>Descarga completada:</b> {resolved_path}"
        else:
            expanded_path = Path(os.path.expanduser(url_or_path))
            if not expanded_path.is_absolute():
                expanded_path = Path(namespace.get("root_dir", "/teamspace/studios/this_studio")) / expanded_path

            if not expanded_path.exists():
                custom_model_status.value = f"<b>Error:</b> la ruta {expanded_path} no existe."
                return

            namespace["optional_custom_training_model"] = str(expanded_path)
            custom_model_widget.value = str(expanded_path)
            namespace["use_optional_custom_training_model"] = True
            use_custom_model_widget.value = True
            apply_params()
            custom_model_status.value = f"<b>Ruta lista:</b> {expanded_path}"

    download_button.on_click(download_custom_model)

    display(Markdown("""
### Configuración rápida del entrenamiento
Personaliza tu sesión desde este panel compacto. Haz clic en **Aplicar parámetros** para guardar los cambios.
"""))
    display(
        widgets.VBox(
            [
                widgets.HTML("<h4 style='margin-bottom:4px;'>Datos básicos</h4>"),
                basics_grid,
                widgets.HTML("<h4 style='margin:16px 0 4px;'>Modelo personalizado (opcional)</h4>"),
                widgets.VBox(
                    [
                        use_custom_model_widget,
                        custom_model_widget,
                        custom_model_filename_widget,
                        widgets.HBox([
                            widgets.HBox([], layout=widgets.Layout(flex="1")),
                            download_button,
                        ]),
                        custom_model_status,
                    ]
                ),
                widgets.HTML("<h4 style='margin:16px 0 4px;'>Ajustes avanzados</h4>"),
                advanced_grid,
                widgets.HBox([widgets.HBox([], layout=widgets.Layout(flex="1")), apply_button]),
                status_output,
            ]
        )
    )

    apply_params()


__all__ = ["render_quick_training_config"]
