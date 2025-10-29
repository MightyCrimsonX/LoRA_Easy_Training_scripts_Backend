"""Utilities for interacting with the kohya-ss/sd-scripts repository.

This module centralises the logic required to locate the repository, expose
its modules on ``sys.path`` and build execution commands for the training
scripts.  Keeping this functionality in one place makes it easier to update
the backend whenever the upstream project changes its entry points.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Iterable, Sequence, Union


def get_repo_root() -> Path:
    """Return the absolute path to the bundled ``sd_scripts`` repository.

    Raises:
        FileNotFoundError: If the repository directory is missing.
        NotADirectoryError: If a file with the expected name exists.
    """

    repo_root = Path(__file__).resolve().parents[1] / "sd_scripts"
    if not repo_root.exists():
        raise FileNotFoundError(
            "The kohya-ss/sd-scripts repository is missing. Initialise the "
            "git submodule or place the repository at 'sd_scripts/'."
        )
    if not repo_root.is_dir():
        raise NotADirectoryError(
            "Expected 'sd_scripts/' to be a directory containing the kohya "
            "repository."
        )
    return repo_root


def ensure_on_path(repo_root: Path | None = None) -> None:
    """Expose the repository on ``sys.path`` if it is not already."""

    repo_root = repo_root or get_repo_root()
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _script_mapping() -> dict[tuple[str, bool, bool], str]:
    return {
        ("lora", False, False): "train_network.py",
        ("lora", True, False): "sdxl_train_network.py",
        ("lora", False, True): "flux_train_network.py",
        ("textual_inversion", False, False): "train_textual_inversion.py",
        ("textual_inversion", True, False): "sdxl_train_textual_inversion.py",
    }


def get_training_script(train_mode: str, is_sdxl: bool, is_flux: bool) -> Path:
    """Return the kohya training script for the requested configuration."""

    key = (train_mode, is_sdxl, is_flux)
    script_name = _script_mapping().get(key)
    if script_name is None:
        raise ValueError(
            "Unsupported training configuration: "
            f"mode={train_mode}, sdxl={is_sdxl}, flux={is_flux}"
        )
    repo_root = get_repo_root()
    script_path = repo_root / script_name
    if not script_path.exists():
        raise FileNotFoundError(
            "Could not find the requested training script inside the kohya "
            f"repository: {script_path}"
        )
    return script_path


def _accelerate_module_args() -> list[str]:
    return ["-m", "accelerate.commands.launch"]


def build_training_command(
    python_executable: str,
    script_path: Path,
    config_path: Path,
    dataset_config_path: Path,
    extra_args: Sequence[str] | None = None,
    use_accelerate: bool = True,
) -> list[str]:
    """Build the command used to invoke kohya's training scripts.

    Args:
        python_executable: Python interpreter used to launch the process.
        script_path: Path to the training script within the repository.
        config_path: Path to the generated ``config.toml`` file.
        dataset_config_path: Path to the generated ``dataset.toml`` file.
        extra_args: Additional CLI arguments to append to the command.
        use_accelerate: When ``True`` runs the script via ``accelerate``.
    """

    command: list[str] = [python_executable]
    if use_accelerate:
        command.extend(_accelerate_module_args())
    command.append(str(script_path))
    command.append(f"--config_file={config_path}")
    command.append(f"--dataset_config={dataset_config_path}")
    if extra_args:
        command.extend(extra_args)
    return command


def inherit_environment(repo_root: Path | None = None) -> dict[str, str]:
    """Return environment variables for launching kohya scripts.

    The kohya scripts expect ``PYTHONPATH`` to include the repository root.
    This helper ensures that requirement while keeping the caller's
    environment intact.
    """

    repo_root = repo_root or get_repo_root()
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH", "")
    repo_str = str(repo_root)
    if python_path:
        if repo_str not in python_path.split(os.pathsep):
            python_path = os.pathsep.join([repo_str, python_path])
    else:
        python_path = repo_str
    environment["PYTHONPATH"] = python_path
    return environment


def normalise_extra_args(arguments: Union[Iterable[str], str, None]) -> list[str]:
    """Convert any iterable of arguments into a list, filtering empties."""

    if not arguments:
        return []
    if isinstance(arguments, str):
        parsed = shlex.split(arguments)
    else:
        parsed = list(arguments)
    return [arg for arg in parsed if arg]

