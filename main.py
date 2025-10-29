try:
    import warnings
except ImportError:
    pass
else:
    warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from threading import Thread

import uvicorn
from starlette import status
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import CLIPTokenizer
from utils.kohya import (
    build_training_command,
    ensure_on_path,
    get_repo_root,
    get_training_script,
    inherit_environment,
    normalise_extra_args,
)

try:
    ensure_on_path()
except (FileNotFoundError, NotADirectoryError) as exc:
    print(f"Warning: {exc}")

from utils.process import process_args, process_dataset_args
from utils.tunnel_service import CloudflaredTunnel, create_tunnel
from utils.validation import validate

if len(sys.argv) > 1:
    os.chdir(sys.argv[1])

if not Path("runtime_store").exists():
    Path("runtime_store").mkdir()


async def stop_server(_: Request = None) -> JSONResponse:
    global server
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse({"detail": "training still running"})
    server.should_exit = True
    server.force_exit = True


async def start_tunnel_service(request: Request) -> JSONResponse:
    config_data = json.loads(app.state.CONFIG.read_text())
    if app.state.TUNNEL:
        return JSONResponse({"service_started": False}, status_code=409)
    app.state.TUNNEL = create_tunnel(config_data)
    if isinstance(app.state.TUNNEL, CloudflaredTunnel):
        config_path = request.query_params.get(
            "config_path", config_data.get("cloudflared_config_path", None)
        )
        if config_path:
            config_path = Path(config_path)
        app.state.TUNNEL.run_tunnel(
            port=config_data.get("port", 8000),
            config=Path(config_path) if config_path else None,
        )
    else:
        app.state.TUNNEL.run_tunnel(port=config_data.get("port", 8000))
    return JSONResponse({"service_started": bool(app.state.TUNNEL)})


async def kill_tunnel_service(_: Request = None) -> JSONResponse:
    if not app.state.TUNNEL:
        return JSONResponse(
            {"killed": False, "reason": "No Tunnel Service Running"},
            status_code=400,
        )
    app.state.TUNNEL.kill_service()
    app.state.TUNNEL = None
    return JSONResponse({"killed": True, "reason": "Tunnel Service Successfully Killed"})


async def check_path(request: Request) -> JSONResponse:
    body = await request.body()
    body = json.loads(body)
    file_path = Path(body["path"])
    valid = False
    if body["type"] == "folder" and file_path.is_dir():
        valid = True
    if body["type"] == "file" and file_path.is_file() and file_path.suffix in body["extensions"]:
        valid = True
    return JSONResponse({"valid": valid})


async def validate_inputs(request: Request) -> JSONResponse:
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse(
            {"detail": "Training Already Running"},
            status_code=status.HTTP_409_CONFLICT,
        )
    body = await request.body()
    body = json.loads(body)
    passed_validation, sdxl, errors, args, dataset_args, tags = validate(body)
    if passed_validation:
        output_args, _ = process_args(args)
        output_dataset_args, _ = process_dataset_args(dataset_args)
        final_args = {"args": output_args, "dataset": output_dataset_args, "tags": tags}
        return JSONResponse(final_args)
    return JSONResponse(
        errors,
        status_code=status.HTTP_400_BAD_REQUEST,
    )


async def is_training(_: Request) -> JSONResponse:
    exit_id = app.state.TRAINING_THREAD.poll() if app.state.TRAINING_THREAD else 0
    return JSONResponse(
        {
            "training": exit_id is None,
            "errored": exit_id is not None and exit_id != 0,
        }
    )


async def tokenize_text(request: Request) -> JSONResponse:
    if not app.state.TOKENIZER:
        app.state.TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text = request.query_params.get("text")
    tokens = app.state.TOKENIZER.tokenize(text)
    token_ids = app.state.TOKENIZER.convert_tokens_to_ids(tokens)
    return JSONResponse({"tokens": tokens, "token_ids": token_ids, "length": len(tokens)})


async def start_training(request: Request) -> JSONResponse:
    global server
    config_payload = json.loads(app.state.CONFIG.read_text())
    if config_payload.get("colab"):
        await kill_tunnel_service()
        await stop_server()
        return
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse(
            {"detail": "Training Already Running"},
            status_code=status.HTTP_409_CONFLICT,
        )
    is_sdxl = request.query_params.get("sdxl", "False") == "True"
    train_type = request.query_params.get("train_mode", "lora")
    is_flux = request.query_params.get("flux", "False") == "True"
    server_config_dict = config_payload if app.state.CONFIG else {}
    python = server_config_dict.get("kohya_python", sys.executable)
    config = Path("runtime_store/config.toml")
    dataset = Path("runtime_store/dataset.toml")
    if not config.is_file() or not dataset.is_file():
        return JSONResponse(
            {"detail": "No Previously Validated Args"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    try:
        script_path = get_training_script(train_type, is_sdxl, is_flux)
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse(
            {"detail": str(exc)},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    app.state.TRAIN_SCRIPT = script_path.name

    extra_cli_args = normalise_extra_args(server_config_dict.get("kohya_extra_args"))
    launch_with_accelerate = server_config_dict.get("use_accelerate_launcher", True)
    command = build_training_command(
        python,
        script_path,
        config.resolve(),
        dataset.resolve(),
        extra_cli_args,
        launch_with_accelerate,
    )
    try:
        environment = inherit_environment(get_repo_root())
    except (FileNotFoundError, NotADirectoryError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_400_BAD_REQUEST)
    cwd = script_path.parent

    print("Launching kohya training:", shlex.join(command))
    app.state.TRAINING_THREAD = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=environment,
    )

    if server_config_dict.get("kill_tunnel_on_train_start") and app.state.TUNNEL:
        app.state.TUNNEL.kill_service()
        app.state.TUNNEL = None
    if server_config_dict.get("kill_server_on_train_end"):
        app.state.MONITOR_THREAD = Thread(target=monitor_training_thread, daemon=True)
        app.state.MONITOR_THREAD.start()
    return JSONResponse({"detail": "Training Started", "training": True})


async def stop_training(request: Request) -> JSONResponse:
    force = request.query_params.get("force", "False").lower() == "true"
    if not app.state.TRAINING_THREAD or app.state.TRAINING_THREAD.poll() is not None:
        return JSONResponse(
            {"detail": "Not Currently Training"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if force:
        app.state.TRAINING_THREAD.stderr = None
        app.state.TRAINING_THREAD.kill()
        return JSONResponse({"detail": "Training Thread Killed"})
    else:
        app.state.TRAINING_THREAD.terminate()
    return JSONResponse({"detail": "Training Thread Requested to Die"})


async def start_resize(request: Request) -> JSONResponse:
    if app.state.TRAINING_THREAD and app.state.TRAINING_THREAD.poll() is None:
        return JSONResponse({"detail": "Training Already Running"}, status_code=status.HTTP_409_CONFLICT)
    data = await request.body()
    data: list[str] = json.loads(data)
    python = sys.executable
    app.state.TRAINING_THREAD = subprocess.Popen([python, f"{Path('utils/resize_lora.py').resolve()}"] + data)
    return JSONResponse({"detail": "Resizing Started"})


def monitor_training_thread():
    if not app.state.TRAINING_THREAD:
        return
    global server
    app.state.TRAINING_THREAD.wait()
    server.should_exit = True
    server.force_exit = True


def load_tokenizer(_: Request) -> JSONResponse:
    app.state.TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return JSONResponse({"success": True})


routes = [
    Route("/stop_server", stop_server, methods=["GET"]),
    Route("/start_tunnel_service", start_tunnel_service, methods=["GET"]),
    Route("/kill_tunnel_service", kill_tunnel_service, methods=["GET"]),
    Route("/check_path", check_path, methods=["POST"]),
    Route("/validate", validate_inputs, methods=["POST"]),
    Route("/is_training", is_training, methods=["GET"]),
    Route("/train", start_training, methods=["GET"]),
    Route("/tokenize", tokenize_text, methods=["GET"]),
    Route("/stop_training", stop_training, methods=["GET"]),
    Route("/resize", start_resize, methods=["POST"]),
    Route("/load_tokenizer", load_tokenizer, methods=["GET"]),
]


print("Starting server...")
app = Starlette(debug=True, routes=routes)
app.state.TRAIN_SCRIPT = None
app.state.TRAINING_THREAD = None
app.state.CONFIG = Path("config.json")
app.state.MONITOR_THREAD = None

if not app.state.CONFIG.exists():
    with app.state.CONFIG.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"remote": False, "port": 8000}, indent=2))

config_data = json.loads(app.state.CONFIG.read_text())
if config_data.get("remote", False):
    app.state.TUNNEL = create_tunnel(config_data)
    if isinstance(app.state.TUNNEL, CloudflaredTunnel):
        config_path = config_data.get("cloudflared_config_path", None)
        app.state.TUNNEL.run_tunnel(
            port=config_data.get("port", 8000), config=Path(config_path) if config_path else None
        )
    else:
        app.state.TUNNEL.run_tunnel(port=config_data.get("port", 8000))
uvi_config = uvicorn.Config(
    app,
    host="0.0.0.0",
    loop="asyncio",
    log_level="critical",
    port=config_data.get("port", 8000),
)
server = uvicorn.Server(config=uvi_config)
print("Server started")

if __name__ == "__main__":
    server.run()
