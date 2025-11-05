"""Notebook entry-point for launching the training run.

This script mirrors the original third notebook cell so it can be executed with
``exec(open(...).read(), globals())`` while keeping the notebook clean.  It may
still rely on global variables defined by previous cells.
"""
import os, re, sys, toml
from pathlib import Path
from time import time
import time
from IPython.display import Markdown, display, HTML, clear_output
from huggingface_hub.utils import disable_progress_bars
import logging

from IPython import get_ipython
import subprocess


disable_progress_bars()
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("ACCELERATE_DISABLE_RICH_PROGRESS", "1")
os.environ.setdefault("ACCELERATE_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("TQDM_MININTERVAL", "2")
os.environ.setdefault("TQDM_NOPOS", "1")
logging.getLogger("accelerate").setLevel(logging.WARNING)
logging.getLogger("accelerate.tracking").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.WARNING)

logging.getLogger("tqdm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)

def _run_cmd(command: str) -> None:
    ip = get_ipython()
    if ip is not None:
        ip.system(command)
    else:
        subprocess.run(command, shell=True, check=True)


root_dir = "/teamspace/studios/this_studio"
trainer_dir = os.path.join(root_dir, "LoRA_Easy_Training_scripts_Backend")
kohya_dir = os.path.join(trainer_dir, "sd-scripts")
models_dir = "/teamspace/studios/this_studio/models"
downloads_dir = os.path.join(root_dir, "downloads")
custom_optimizer_path = os.path.join(trainer_dir, "custom_scheduler")
if custom_optimizer_path not in sys.path:
  sys.path.append(custom_optimizer_path)
os.environ["PYTHONPATH"] = custom_optimizer_path + os.pathsep + os.environ.get("PYTHONPATH", "")

# Lightning notebooks run continuously; automatic shutdown is not managed here.
print("🔵 Lightning environment detectado. Detén el cuaderno manualmente cuando termines.")

# These carry information from past executions
if "model_url" in globals():
  old_model_url = model_url
else:
  old_model_url = None
if "dependencies_installed" not in globals():
  dependencies_installed = False
if "model_file" not in globals():
  model_file = None

# These may be set by other cells, some are legacy
if "custom_dataset" not in globals():
  custom_dataset = None
if "override_dataset_config_file" not in globals():
  override_dataset_config_file = None
if "override_config_file" not in globals():
  override_config_file = None

COMMIT = "fa2427c6b468231e8e270e40fe72add780118dbe"
LOWRAM = False
LOAD_TRUNCATED_IMAGES = True
BETTER_EPOCH_NAMES = True
FIX_DIFFUSERS = True
FIX_WANDB_WARNING = True

#@title ## 🚩 Start Here

#@markdown ### ▶️ Setup
#@markdown El nombre de tu proyecto será el mismo que el de la carpeta que contiene tus imágenes. No se permiten espacios, puedes usar `guión bajo` si el nombre es muy largo.
project_name_param = " " #@param {type:"string"}
project_name = globals().get("project_name", project_name_param).strip()
#@markdown La estructura de carpetas no importa y es puramente por comodidad. Asegúrate de elegir siempre el mismo.  Me gusta organizar por proyecto.
folder_structure = "Organize by project (lora_projects/project_name/dataset)" #@param ["Organize by category (lora_training/datasets/project_name)", "Organize by project (lora_projects/project_name/dataset)"]
#@markdown Decida el modelo que se descargará y utilizará para el entrenamiento. También puedes elegir tu propio modelo pegando su enlace de descarga o proporcionando una ruta dentro de `/teamspace/studios/this_studio`.
training_model_param = "Illustrious_2.0" # @param ["Pony Diffusion V6 XL","Animagine XL V3","animagine_4.0_zero","Illustrious_0.1","Illustrious_2.0","NoobAI-XL0.75","NoobAI-XL0.5","Stable Diffusion XL 1.0 base","NoobAIXL0_75vpred","RouWei_v080vpred"]
training_model = globals().get("training_model", training_model_param)
optional_custom_training_model_param = "" #@param {type:"string"}
optional_custom_training_model = str(globals().get("optional_custom_training_model", optional_custom_training_model_param)).strip()
#@markdown Esto forzara el uso del modelo en formato diffusers, puede ser util en ciertos casos. <p>
#@markdown Manten esto desmarcado para usar un modelo ckpt (.safetensors) para el entrenamiento.
force_load_diffusers_param = False # @param {"type":"boolean"}
force_load_diffusers = globals().get("force_load_diffusers", force_load_diffusers_param)
#@markdown Marca está opción si el modelo custom esta en dicho formato
custom_model_is_diffusers_param = False #@param {type:"boolean"}
custom_model_is_diffusers = bool(globals().get("custom_model_is_diffusers", custom_model_is_diffusers_param))
#@markdown Marca esta opción si tu modelo soporta vpred de lo contrario dejala desmarcada.
custom_model_is_vpred_param = False #@param {type:"boolean"}
custom_model_is_vpred = bool(globals().get("custom_model_is_vpred", custom_model_is_vpred_param))
#@markdown Activa esta opción para utilizar el modelo personalizado descargado.
use_optional_custom_training_model_param = False #@param {type:"boolean"}
use_optional_custom_training_model = bool(globals().get("use_optional_custom_training_model", use_optional_custom_training_model_param))
#@markdown Utilice wandb si desea visualizar el progreso de su entrenamiento a lo largo del tiempo.
wandb_key = "" #@param {type:"string"}

custom_model_selected = use_optional_custom_training_model and len(optional_custom_training_model) > 0
load_diffusers = (custom_model_is_diffusers and custom_model_selected) or (force_load_diffusers and not custom_model_selected)
vpred = custom_model_is_vpred and custom_model_selected

if custom_model_selected:
  model_url = optional_custom_training_model
elif "Pony" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/WhiteAiZ/Pony_diffusion_v6_diffusers_fp16"
  else:
    model_url = "https://huggingface.co/WhiteAiZ/PonyXL/resolve/main/PonyDiffusionV6XL.safetensors"
  model_file = os.path.join(models_dir, "ponyDiffusionV6XL.safetensors")
elif "Animagine" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/cagliostrolab/animagine-xl-3.0"
  else:
    model_url = "https://civitai.com/api/download/models/293564"
  model_file = os.path.join(models_dir, "animagineXLV3.safetensors")
elif "animagine_4.0_zero" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/cagliostrolab/animagine-xl-4.0-zero"
  else:
    model_url = "https://huggingface.co/cagliostrolab/animagine-xl-4.0-zero/resolve/main/animagine-xl-4.0-zero.safetensors"
  model_file = os.path.join(models_dir, "animagine-xl-4.0-zero.safetensors")
elif "Illustrious_0.1" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0"
  else:
    model_url = "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors"
elif "Illustrious_2.0" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/WhiteAiZ/Illustrious_2.0"
  else:
    model_url = "https://huggingface.co/WhiteAiZ/Illustrious_2.0/resolve/main/illustriousXL20_v20.safetensors"
  model_file = os.path.join(models_dir, "illustriousXL20_v20.safetensors")
elif "NoobAI-XL0.75" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/Laxhar/noobai-XL-0.75"
  else:
    model_url = "https://huggingface.co/Laxhar/noobai-XL-0.75/resolve/main/NoobAI-XL-v0.75.safetensors"
elif "NoobAI-XL0.5" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/Laxhar/noobai-XL-0.5"
  else:
    model_url = "https://huggingface.co/Laxhar/noobai-XL-0.5/resolve/main/NoobAI-XL-v0.5.safetensors"
elif "Stable Diffusion XL 1.0 base" in training_model:
  if load_diffusers:
    model_url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/"
  else:
    model_url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
elif "NoobAIXL0_75vpred" in training_model:
  vpred = True
  if load_diffusers:
    model_url = "https://huggingface.co/Laxhar/noobai-XL-Vpred-0.75"
  else:
    model_url = "https://huggingface.co/Laxhar/noobai-XL-Vpred-0.75/resolve/main/NoobAI-XL-Vpred-v0.75.safetensors"
  model_file = os.path.join(models_dir, "NoobAI-XL-Vpred-v0.75.safetensors")
else:
  vpred = True
  if load_diffusers:
    model_url = "https://huggingface.co/John6666/rouwei-v080-vpred-sdxl"
  else:
    model_url = "https://huggingface.co/WhiteAiZ/RouWei/resolve/main/rouwei_v080Vpred.safetensors"
  model_file = os.path.join(models_dir, "rouwei_v080Vpred.safetensors")

if load_diffusers:
  vae_file= "stabilityai/sdxl-vae"
else:
  vae_url = "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"
  vae_file = os.path.join(models_dir, "sdxl_vae.safetensors")

model_url = model_url.strip()

#@markdown ### ▶️ Processing
#@markdown Por defecto la resolución para personajes es 1024. otras resoluciones que puedes usar son 896 (recomendado para personajes o 1024) y 768 (recomendado para estilos, puedes usar más repeticiones con esta resolución).
resolution_param = 1024 #@param {type:"dropdown", min:768, max:1536, step:128}
resolution = globals().get("resolution", resolution_param)
#@markdown Activa `Flip Aug`si tu dataset es pequeño, util en personajes isometricos, volteara todas tus imagenes (modo espejo) para aprender el doble, pero podria afectar a personajes con tatuajes, marcas, cicatrices etc...
flip_aug = False #@param {type:"boolean"}
caption_extension = ".txt" # @param [".txt",".caption"]
#@markdown Mezcla etiquetas de anime, mejora el aprendizaje y las indicaciones.  Una etiqueta de activación va al comienzo de cada archivo de texto y no se mezclará.<p>
shuffle_tags = True #@param {type:"boolean"}
shuffle_caption = shuffle_tags
activation_tags = "1" #@param [0,1,2,3]
keep_tokens = int(activation_tags)

#@markdown ### ▶️ Steps <p>
#@markdown Tus imágenes se repetirán esta cantidad de veces durante el entrenamiento. Te recomiendo que tus imágenes multiplicadas por sus repeticiones esté entre 200 y 400.
num_repeats_param = 2 #@param {type:"number"}
num_repeats = globals().get("num_repeats", num_repeats_param)
#@markdown Elige cuánto tiempo quieres entrenar.  Un buen punto de partida es alrededor de 10 épocas o alrededor de 2000 pasos.<p>
#@markdown Una época es una cantidad de pasos igual a: la cantidad de imágenes multiplicada por sus repeticiones, dividida por el tamaño del lote. <p>
preferred_unit = "Epochs" #@param ["Epochs", "Steps"]
how_many_param = 40 #@param {type:"number"}
how_many = globals().get("how_many", how_many_param)
max_train_epochs = how_many if preferred_unit == "Epochs" else None
max_train_steps = how_many if preferred_unit == "Steps" else None
#@markdown Guardar más épocas te permitirá comparar mejor el progreso de tu Lora.
save_every_n_epochs = 1 #@param {type:"number"}
keep_only_last_n_epochs = 5 #@param {type:"number"}
if not save_every_n_epochs:
  save_every_n_epochs = max_train_epochs
if not keep_only_last_n_epochs:
  keep_only_last_n_epochs = max_train_epochs

#@markdown ### ▶️ Learning
#@markdown La tasa de aprendizaje es lo más importante para tus resultados. Si quieres entrenar más lento con muchas imágenes, o si tu dim y alfa son altos, mueve el unet a 2e-4 o menos.  <p>
#@markdown El codificador de texto ayuda al Lora a aprender conceptos un poco mejor.  Se recomienda hacerlo la mitad o una quinta parte del unet.  Si estás entrenando un estilo, puedes incluso configurarlo en 0.
unet_lr_param = 1e-4 #@param {type:"number"}
unet_lr = globals().get("unet_lr", unet_lr_param)
text_encoder_lr_param = 5e-5 #@param {type:"number"}
text_encoder_lr = globals().get("text_encoder_lr", text_encoder_lr_param)
#@markdown El scheduler es el algoritmo que guía la tasa de aprendizaje. Si no está seguro, elije "constant" e ignore el número. Personalmente recomiendo `cosine_with_restarts` con 3 reinicios.
lr_scheduler_param = "constant_with_warmup" # @param ["constant","cosine","cosine_with_restarts","constant_with_warmup","linear","polynomial","rex"]
lr_scheduler = globals().get("lr_scheduler", lr_scheduler_param)
lr_scheduler_number = 0 #@param {type:"number"}
#@markdown Pasos dedicados a "calentar" la tasa de aprendizaje durante la capacitación para lograr eficiencia. Recomiendo dejarlo al 5%.
lr_warmup_ratio = 0.05 #@param {type:"slider", min:0.0, max:0.2, step:0.01}
lr_warmup_steps = 100 #@param {type:"number"}
#@markdown Estas configuraciones pueden producir mejores resultados.`min_snr_gamma` ajusta la pérdida con el tiempo. `ip_noise_gamma` ajusta el ruido aleatorio.
min_snr_gamma_enabled = True #@param {type:"boolean"}
min_snr_gamma = 8.0 #@param {type:"slider", min:4, max:16.0, step:0.5}
ip_noise_gamma_enabled = True #@param {type:"boolean"}
ip_noise_gamma = 0.05 #@param {type:"slider", min:0.05, max:0.1, step:0.01}
#@markdown Multinoise puede ayudar con el equilibrio del color (negros más oscuros, blancos más claros) no es necesario activarlo si entrenas Lora Vpred.
multinoise = False #@param {type:"boolean"}

#@markdown ### ▶️ Structure
#@markdown LoRA es del tipo clásico y bueno para una variedad de propósitos. LoCon es bueno con los estilos artísticos (también funciona con personajes) ya que tiene más capas para aprender más aspectos del conjunto de datos.
lora_type_param = "LoRA" # @param ["LoRA","LoCon"]
lora_type = globals().get("lora_type", lora_type_param)

#@markdown A continuación se muestran algunos valores XL recomendados para las siguientes configuraciones:

#@markdown | type | network_dim | network_alpha | conv_dim | conv_alpha |
#@markdown | :---: | :---: | :---: | :---: | :---: |
#@markdown | Personaje LoRA | 4 | 16 |   |   |
#@markdown | Regular y Estilo LoRA | 8 | 4 |   |   |
#@markdown | Style LoCon | 16 | 8 | 16 | 8 |

#@markdown Más dim significa un Lora más grande, puede contener más información, pero más no siempre es mejor.
network_dim_param = 16 #@param {type:"number", min:1, max:32, step:1}
network_dim = globals().get("network_dim", network_dim_param)
network_alpha_param = 32 #@param {type:"number", min:1, max:32, step:1}
network_alpha = globals().get("network_alpha", network_alpha_param)
#@markdown Los siguientes dos valores solo se aplican a las capas adicionales de LoCon.
conv_dim_param = 16 #@param {type:"number", min:1, max:32, step:1}
try:
  conv_dim = int(globals().get("conv_dim", conv_dim_param))
except (TypeError, ValueError):
  conv_dim = conv_dim_param
conv_alpha_param = 8 #@param {type:"number", min:1, max:32, step:1}
try:
  conv_alpha = int(globals().get("conv_alpha", conv_alpha_param))
except (TypeError, ValueError):
  conv_alpha = conv_alpha_param

network_module = "networks.lora"
network_args = None
if lora_type.lower() == "locon":
  network_args = [f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}"]

#@markdown ### ▶️ Training
#@markdown Ajuste estos parámetros según la configuración de su colab.

#@markdown El batch size de 4 es el predeterminado pero puedes incrementarlo incluso a 8 usando una resolución baja (768).
#@markdown
#@markdown Un tamaño de lote más alto suele ser más rápido pero utiliza más memoria.
train_batch_size_param = 8 #@param {type:"slider", min:1, max:16, step:1}
train_batch_size = globals().get("train_batch_size", train_batch_size_param)
#@markdown xformers funciona mejor que sdpa con los nuevos scrips.
cross_attention = "xformers" #@param ["sdpa", "xformers"]
#@markdown Utilice `full fp16` para el uso mínimo de memoria. <p>
#@markdown `float, full bf16, full fp16, mixed bf16 y mixed fp16` solo funcionaran con colab pro. <p>
#@markdown El Lora se entrenará con la precisión seleccionada, pero siempre se guardará en formato fp16 por razones de compatibilidad.
precision_param = "fp16" #@param ["float", "full fp16", "full bf16", "mixed fp16", "mixed bf16"]
precision = globals().get("precision", precision_param)
#@markdown El almacenamiento en caché latente en disco agregará un archivo de 250 KB junto a cada imagen, pero usará considerablemente menos memoria.
cache_latents = True #@param {type:"boolean"}
cache_latents_to_disk = False #@param {type:"boolean"}
#@markdown La siguiente opción desactivará shuffle_tags y deshabilitará el entrenamiento del codificador de texto.
cache_text_encoder_outputs  = False  # @param {type:"boolean"}

mixed_precision = "no"
if "fp16" in precision:
  mixed_precision = "fp16"
elif "bf16" in precision:
  mixed_precision = "bf16"
full_precision = "full" in precision

#@markdown ### ▶️ Advanced
#@markdown El optimizador es el algoritmo utilizado para el entrenamiento. Adafactor es el predeterminado y funciona muy bien, mientras que el Prodigy administra la tasa de aprendizaje automáticamente y puede tener varias ventajas, como entrenar más rápido, debido a que necesita menos pasos y funcionan mejor para datasets pequeños.
optimizer_param = "Prodigy" #@param ["AdamW8bit", "Prodigy", "DAdaptation", "DadaptAdam", "DadaptLion", "AdamW", "Lion", "SGDNesterov", "SGDNesterov8bit", "AdaFactor", "Came"]
optimizer = globals().get("optimizer", optimizer_param)
#@markdown Argumentos recomendados para Adafactor: `scale_parameter=False relative_step=False warmup_init=False` <p>
#@markdown Argumentos recomendados para AdamW8bit: `weight_decay=0.1 betas=[0.9,0.99]`<p>
#@markdown Argumentos recomendados para Prodigy: `decouple=True weight_decay=0.01 betas=[0.9,0.999] d_coef=2 use_bias_correction=True safeguard_warmup=True`<p>
#@markdown Argumentos recomendado para CAME: `weight_decay=0.04` <p>
#@markdown Si se selecciona Dadapt o Prodigy y se marca la casilla recomendada, los siguientes valores recomendados anularán cualquier configuración anterior:<p>
#@markdown `unet_lr=0.75`, `text_encoder_lr=0.75`, `network_alpha=network_dim`, `full_precision=True`<p>
#@markdown Si selecciona Prodigy o Dadapt recomiendo usar `mixed fp16`para mejores resultados. <p>
recommended_values = True #@param {type:"boolean"}
#@markdown Alternativamente, establezca sus propios argumentos de optimizador separados por espacios (no comas). `recommended_values` debe estar deshabilitado.
optimizer_args = "" #@param {type:"string"}
optimizer_args = [a.strip() for a in optimizer_args.split(' ') if a]


if recommended_values:
  if any(opt in optimizer.lower() for opt in ["dadapt", "prodigy"]):
    unet_lr = 0.75
    text_encoder_lr = 0.75
    network_alpha = network_dim
    full_precision = False
  if optimizer == "Prodigy":
    optimizer_args = ["decouple=True", "weight_decay=0.01", "betas=[0.9,0.999]", "d_coef=2", "use_bias_correction=True", "safeguard_warmup=True"]
  elif optimizer == "AdamW8bit":
    optimizer_args = ["weight_decay=0.1", "betas=[0.9,0.99]"]
  elif optimizer == "AdaFactor":
    optimizer_args = ["scale_parameter=False", "relative_step=False", "warmup_init=False"]
  elif optimizer == "Came":
    optimizer_args = ["weight_decay=0.04"]

if optimizer == "Came":
  optimizer = "LoraEasyCustomOptimizer.came.CAME"

lr_scheduler_type = None
lr_scheduler_args = None
lr_scheduler_num_cycles = lr_scheduler_number
lr_scheduler_power = lr_scheduler_number

if "rex" in lr_scheduler:
  lr_scheduler = "cosine"
  lr_scheduler_type = "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts"
  lr_scheduler_args = ["min_lr=1e-9", "gamma=0.9", "d=0.9"]

# Misc
seed = 42
gradient_accumulation_steps = 1
bucket_reso_steps = 64
min_bucket_reso = 256
max_bucket_reso = 4096

#@markdown ### ▶️ Ready
#@markdown Ahora puedes ejecutar esta celda para entrenar tu Lora. ¡Buena suerte! <p>

# 👩‍💻 Cool code goes here


for required_dir in (models_dir, downloads_dir):
  os.makedirs(required_dir, exist_ok=True)


def lightning_rel(path):
  try:
    return os.path.relpath(path, root_dir)
  except ValueError:
    return path


venv_python = "/home/zeus/miniconda3/envs/cloudspace/bin/python3"
#venv_pip = os.path.join(kohya_dir, "venv/bin/pip")
train_network = os.path.join(kohya_dir, "sdxl_train_network.py")

if "lora_projects" in folder_structure:
  main_dir      = os.path.join(root_dir, "lora_projects")
  log_folder    = os.path.join(main_dir, "_logs")
  config_folder = os.path.join(main_dir, project_name)
  images_folder = os.path.join(main_dir, project_name, "dataset")
  output_folder = os.path.join(main_dir, project_name, "output")
else:
  main_dir      = os.path.join(root_dir, "lora_training")
  images_folder = os.path.join(main_dir, "datasets", project_name)
  output_folder = os.path.join(main_dir, "output", project_name)
  config_folder = os.path.join(main_dir, "config", project_name)
  log_folder    = os.path.join(main_dir, "log")

config_file = os.path.join(config_folder, "training_config.toml")
dataset_config_file = os.path.join(config_folder, "dataset_config.toml")

def install_trainer():
  global installed
  libtcmalloc_path = os.path.join(root_dir, "libtcmalloc_minimal.so.4")

  if 'installed' not in globals():
    installed = False

  if not os.path.exists(libtcmalloc_path):
    _run_cmd(f"wget -q -c --show-progress https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O {libtcmalloc_path}")

  if not os.path.exists(trainer_dir):
    _run_cmd(f"git clone -b dev https://github.com/gwhitez/LoRA_Easy_Training_scripts_Backend.git {trainer_dir}")
  else:
    os.chdir(trainer_dir)
    _run_cmd("git pull")
    os.chdir(root_dir)

  os.chdir(trainer_dir)
  display(HTML("<h2 style='color: yellow;'>Descargando dependencias</h2>"))
  _run_cmd("chmod 755 /teamspace/studios/this_studio/LoRA_Easy_Training_scripts_Backend/colab_install.sh")
  _run_cmd("/teamspace/studios/this_studio/LoRA_Easy_Training_scripts_Backend/colab_install.sh > install_log.txt 2>&1")

  os.chdir(kohya_dir)
  if LOAD_TRUNCATED_IMAGES:
    _run_cmd("sed -i 's/from PIL import Image/from PIL import Image, ImageFile\nImageFile.LOAD_TRUNCATED_IMAGES=True/g' library/train_util.py")
  if BETTER_EPOCH_NAMES:
    _run_cmd("sed -i 's/{:06d}/{:02d}/g' library/train_util.py")
    train_network_path = Path(kohya_dir) / "train_network.py"
    try:
      text = train_network_path.read_text()
    except FileNotFoundError:
      text = None
    if text is not None:
      pattern = re.compile(r'"-\{:[0-9]+d\}\."\.format\(([^)]+)\)\s*\+\s*args\.save_model_as')

      def _repl(match: re.Match) -> str:
        expr = match.group(1).strip()
        return 'f"-{' + expr + ':02d}." + args.save_model_as'

      new_text, count = pattern.subn(_repl, text)
      if count:
        train_network_path.write_text(new_text)
      else:
        print("⚠️ No se pudo parchear train_network.py para los nuevos nombres de checkpoint.")
  if FIX_DIFFUSERS:
    deprecation_utils = os.path.join(kohya_dir, "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/diffusers/utils/deprecation_utils.py")
    _run_cmd(f"sed -i 's/if version.parse/if False:#/g' {deprecation_utils}")
  if FIX_WANDB_WARNING:
    _run_cmd("sed -i 's/accelerator.log(logs, step=epoch + 1)//g' train_network.py")
    _run_cmd("sed -i 's/accelerator.log(logs, step=epoch + 1)//g' sdxl_train.py")

  os.environ["LD_PRELOAD"] = libtcmalloc_path
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["BITSANDBYTES_NOWELCOME"] = "1"
  os.environ["SAFETENSORS_FAST_GPU"] = "1"
  os.environ["PYTHONWARNINGS"] = "ignore"
  os.chdir(root_dir)

def validate_dataset():
  global lr_warmup_steps, lr_warmup_ratio, caption_extension, keep_tokens, model_url
  supported_types = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

  print("\n💿 Checking dataset...")
  if not project_name.strip() or any(c in project_name for c in " .()\"'\\/"):
    print("💥 Error: Elija un nombre de proyecto válido.")
    return

  # Find the folders and files
  if custom_dataset:
    try:
      datconf = toml.loads(custom_dataset)
      datasets = [d for d in datconf["datasets"][0]["subsets"]]
    except:
      print(f"💥 Error: El conjunto de datos personalizado no es válido o contiene un error. Por favor, compruebe la plantilla original.")
      return
    reg = [d.get("image_dir") for d in datasets if d.get("is_reg", False)]
    datasets_dict = {d["image_dir"]: d["num_repeats"] for d in datasets}
    folders = datasets_dict.keys()
    files = [f for folder in folders for f in os.listdir(folder)]
    images_repeats = {folder: (len([f for f in os.listdir(folder) if f.lower().endswith(supported_types)]), datasets_dict[folder]) for folder in folders}
  else:
    reg = []
    folders = [images_folder]
    files = os.listdir(images_folder)
    images_repeats = {images_folder: (len([f for f in files if f.lower().endswith(supported_types)]), num_repeats)}

  # Validation
  for folder in folders:
    if not os.path.exists(folder):
      print(f"💥 Error: La carpeta {lightning_rel(folder)} no existe.")
      return
  for folder, (img, rep) in images_repeats.items():
    if not img:
      print(f"💥 Error: tú {lightning_rel(folder)} La carpeta está vacía.")
      return
  test_files = []
  for f in files:
    if not f.lower().endswith((caption_extension, ".npz")) and not f.lower().endswith(supported_types):
      print(f"💥 Error: Archivo no válido en el conjunto de datos: \"{f}\". Abortar.")
      return
    for ff in test_files:
      if f.endswith(supported_types) and ff.endswith(supported_types) \
          and os.path.splitext(f)[0] == os.path.splitext(ff)[0]:
        print(f"💥 Error: Los archivos {f} y {ff} no puede tener el mismo nombre. Abortar.")
        return
    test_files.append(f)

  if caption_extension and not [txt for txt in files if txt.lower().endswith(caption_extension)]:
    caption_extension = ""

  # Show estimations to the user

  pre_steps_per_epoch = sum(img*rep for (img, rep) in images_repeats.values())
  steps_per_epoch = pre_steps_per_epoch/train_batch_size
  total_steps = max_train_steps or int(max_train_epochs*steps_per_epoch)
  estimated_epochs = int(total_steps/steps_per_epoch)
  lr_warmup_steps = int(total_steps*lr_warmup_ratio)

  for folder, (img, rep) in images_repeats.items():
    print("📁" + lightning_rel(folder) + (" (Regularization)" if folder in reg else ""))
    print(f"📈 Se encontró {img} imágenes con {rep} repeticiones, igual {img*rep} pasos.")
  print(f"📉 Divide {pre_steps_per_epoch} pasos por {train_batch_size} batch size para obtener {steps_per_epoch} pasos por epoch.")
  if max_train_epochs:
    print(f"🔮 Habrá {max_train_epochs} epochs, por alrededor de {total_steps} total de pasos.")
  else:
    print(f"🔮 Habrá {total_steps} pasos, divididos en {estimated_epochs} epochs y algo más.")

  if total_steps > 10000:
    print("💥 Error: El total de pasos es demasiado alto. Probablemente cometiste un error. Abortar...")
    return

  return True

def create_config():
  global dataset_config_file, config_file, model_file

  if override_config_file:
    config_file = override_config_file
    print(f"\n⭕ Using custom config file {config_file}")
  else:
    config_dict = {
      "network_arguments": {
        "unet_lr": unet_lr,
        "text_encoder_lr": text_encoder_lr if not cache_text_encoder_outputs else 0,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "network_module": network_module,
        "network_args": network_args,
        "network_train_unet_only": text_encoder_lr == 0 or cache_text_encoder_outputs,
      },
      "optimizer_arguments": {
        "learning_rate": unet_lr,
        "lr_scheduler": lr_scheduler,
        "lr_scheduler_type": lr_scheduler_type,
        "lr_scheduler_args": lr_scheduler_args,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
        "lr_warmup_steps": lr_warmup_steps if lr_scheduler not in ("cosine", "constant") else None,
        "optimizer_type": optimizer,
       "optimizer_args": optimizer_args or None,
        "loss_type": "l2",
        "max_grad_norm": 1.0,
      },
      "training_arguments": {
        "lowram": LOWRAM,
        "pretrained_model_name_or_path": model_file,
        "vae": vae_file,
        "max_train_steps": max_train_steps,
        "max_train_epochs": max_train_epochs,
        "train_batch_size": train_batch_size,
        "seed": seed,
        "max_token_length": 225,
        "xformers": cross_attention == "xformers",
        "sdpa": cross_attention == "sdpa",
        "min_snr_gamma": min_snr_gamma if min_snr_gamma_enabled else None,
        "ip_noise_gamma": ip_noise_gamma if ip_noise_gamma_enabled else None,
        "no_half_vae": True,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_data_loader_n_workers": 1,
        "persistent_data_loader_workers": True,
        "mixed_precision": mixed_precision,
        "full_fp16": mixed_precision == "fp16" and full_precision,
        "full_bf16": mixed_precision == "bf16" and full_precision,
        "cache_latents": cache_latents,
        "cache_latents_to_disk": cache_latents_to_disk,
        "cache_text_encoder_outputs": cache_text_encoder_outputs,
        "min_timestep": 0,
        "max_timestep": 1000,
        "prior_loss_weight": 1.0,
        "multires_noise_iterations": 6 if multinoise else None,
        "multires_noise_discount": 0.3 if multinoise else None,
        "v_parameterization": vpred or None,
        "scale_v_pred_loss_like_noise_pred": vpred or None,
        "zero_terminal_snr": vpred or None,
      },
      "saving_arguments": {
        "save_precision": "fp16",
        "save_model_as": "safetensors",
        "save_every_n_epochs": save_every_n_epochs,
        "save_last_n_epochs": keep_only_last_n_epochs,
        "output_name": project_name,
        "output_dir": output_folder,
        "log_prefix": project_name,
        "logging_dir": log_folder,
        "wandb_api_key": wandb_key or None,
        "log_with": "wandb" if wandb_key else None,
      }
    }

    for key in config_dict:
      if isinstance(config_dict[key], dict):
        config_dict[key] = {k: v for k, v in config_dict[key].items() if v is not None}

    with open(config_file, "w") as f:
      f.write(toml.dumps(config_dict))
    print(f"\n📄 Config saved to {config_file}")

  if override_dataset_config_file:
    dataset_config_file = override_dataset_config_file
    print(f"⭕ Using custom dataset config file {dataset_config_file}")
  else:
    dataset_config_dict = {
      "general": {
        "resolution": resolution,
        "shuffle_caption": shuffle_caption and not cache_text_encoder_outputs,
        "keep_tokens": keep_tokens,
        "flip_aug": False,
        "caption_extension": caption_extension,
        "enable_bucket": True,
        "bucket_no_upscale": False,
        "bucket_reso_steps": bucket_reso_steps,
        "min_bucket_reso": min_bucket_reso,
        "max_bucket_reso": max_bucket_reso,
      },
      "datasets": toml.loads(custom_dataset)["datasets"] if custom_dataset else [
        {
          "subsets": [
            {
              "num_repeats": num_repeats,
              "image_dir": images_folder,
              "class_tokens": None if caption_extension else project_name
            }
          ]
        }
      ]
    }

    for key in dataset_config_dict:
      if isinstance(dataset_config_dict[key], dict):
        dataset_config_dict[key] = {k: v for k, v in dataset_config_dict[key].items() if v is not None}

    with open(dataset_config_file, "w") as f:
      f.write(toml.dumps(dataset_config_dict))
    print(f"📄 Configuración de dataset guardada en {dataset_config_file}")

def download_model():
  global old_model_url, model_url, model_file, vae_url, vae_file

  def ensure_vae_ready() -> bool:
    if load_diffusers:
      return True

    target = Path(vae_file)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
      print(f"✅ VAE listo: {target}")
      return True

    print(f"🌐 Descargando VAE en {target} ...")
    try:
      _run_cmd(
        f"aria2c '{vae_url}' --console-log-level=warn -c -s 16 -x 16 -k 10M -d {target.parent} -o '{target.name}'"
      )
    except Exception as exc:
      print(f"💥 Error al descargar el VAE: {exc}")
      return False

    if target.exists():
      print(f"✅ VAE listo: {target}")
      return True

    print(f"💥 Error: el VAE {target} no se encontró después de la descarga.")
    return False

  real_model_url = (model_url or "").strip()
  if not real_model_url:
    print("💥 Error: no se especificó ningún modelo base para entrenar.")
    return False

  if load_diffusers:
    if 'huggingface.co' in real_model_url:
      match = re.search(r'huggingface.co/([^/]+)/([^/]+)', real_model_url)
      if match:
        username = match.group(1)
        model_name = match.group(2)
        model_file = f"{username}/{model_name}"
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        existing_folders = set(fs.ls(model_file, detail=False))
        necessary_folders = ["scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae"]
        if all(f"{model_file}/{folder}" in existing_folders for folder in necessary_folders):
          print("🍃 Modelo diffusers identificado; kohya manejará la descarga.")
          return True
    raise ValueError("💥 Failed to load Diffusers model. Si este modelo no es diffusers, desactiva la opción correspondiente.")

  local_candidate = None
  if '://' not in real_model_url:
    candidate = Path(real_model_url)
    if not candidate.is_absolute():
      candidate = Path(root_dir) / real_model_url.lstrip('/')
    if candidate.exists():
      local_candidate = candidate
    else:
      print(f"💥 Error: el modelo local {candidate} no existe. Asegúrate de que esté dentro de {root_dir} o usa una URL.")
      return False

  if local_candidate is not None:
    model_file = str(local_candidate)
    print(f"📁 Usando modelo local: {model_file}")
    if not ensure_vae_ready():
      return False
  else:
    if real_model_url.lower().endswith((".ckpt", ".safetensors")):
      filename = os.path.basename(real_model_url)
    else:
      filename = "downloaded_model.safetensors"

    civitai_match = re.search(r"(?:https?://)?(?:www\.)?civitai\.com/models/([0-9]+)(/[A-Za-z0-9-_]+)?", real_model_url)
    if civitai_match:
      name_hint = civitai_match.group(2)
      if name_hint:
        filename = f"{Path(name_hint).name}.safetensors"
      version_match = re.search(r"modelVersionId=([0-9]+)", real_model_url)
      if version_match:
        real_model_url = f"https://civitai.com/api/download/models/{version_match.group(1)}"
      else:
        raise ValueError("💥 optional_custom_training_model contiene un enlace de Civitai sin modelVersionId válido.")

    model_file = os.path.join(models_dir, filename)
    if os.path.exists(model_file):
      _run_cmd(f"rm '{model_file}'")

    if re.search(r"(?:https?://)?(?:www\.)?huggingface\.co/[^/]+/[^/]+/blob", real_model_url):
      real_model_url = real_model_url.replace("blob", "resolve")

    print(f"🌐 Descargando modelo en {model_file} ...")
    _run_cmd(f"aria2c '{real_model_url}' --console-log-level=warn -c -s 16 -x 16 -k 10M -d {models_dir} -o '{os.path.basename(model_file)}'")

    if not ensure_vae_ready():
      return False

  if model_file.lower().endswith(".safetensors"):
    from safetensors.torch import load_file as load_safetensors
    try:
      test = load_safetensors(model_file)
      del test
    except Exception as exc:
      print(f"💥 Error al validar el archivo safetensors {model_file}: {exc}")
      return False

  if model_file.lower().endswith(".ckpt"):
    from torch import load as load_ckpt
    try:
      test = load_ckpt(model_file)
      del test
    except Exception:
      return False

  return True


def calculate_rex_steps():
  # https://github.com/derrian-distro/LoRA_Easy_Training_scripts_Backend/blob/c34084b0435e6e19bb7a01ac1ecbadd185ee8c1e/utils/validation.py#L268
  global max_train_steps
  print("\n🤔 Calculating Rex steps")
  if max_train_steps:
    calculated_max_steps = max_train_steps
  else:
    from library.train_util import BucketManager
    from PIL import Image
    from pathlib import Path
    import math

    with open(dataset_config_file, "r") as f:
      subsets = toml.load(f)["datasets"][0]["subsets"]

    supported_types = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    res = (resolution, resolution)
    bucketManager = BucketManager(False, res, min_bucket_reso, max_bucket_reso, bucket_reso_steps)
    bucketManager.make_buckets()
    for subset in subsets:
        for image in Path(subset["image_dir"]).iterdir():
            if image.suffix not in supported_types:
                continue
            with Image.open(image) as img:
                bucket_reso, _, _ = bucketManager.select_bucket(img.width, img.height)
                for _ in range(subset["num_repeats"]):
                    bucketManager.add_image(bucket_reso, image)
    steps_before_acc = sum(math.ceil(len(bucket) / train_batch_size) for bucket in bucketManager.buckets)
    calculated_max_steps = math.ceil(steps_before_acc / gradient_accumulation_steps) * max_train_epochs
    del bucketManager

  cycle_steps = calculated_max_steps // (lr_scheduler_num_cycles or 1)
  print(f"  cycle steps: {cycle_steps}")
  lr_scheduler_args.append(f"first_cycle_max_steps={cycle_steps}")

  warmup_steps = round(calculated_max_steps * lr_warmup_ratio) // (lr_scheduler_num_cycles or 1)
  if warmup_steps > 0:
    print(f"  warmup steps: {warmup_steps}")
    lr_scheduler_args.append(f"warmup_steps={warmup_steps}")

def main():
  global dependencies_installed

  for dir in (main_dir, trainer_dir, log_folder, images_folder, output_folder, config_folder, models_dir, downloads_dir):
    os.makedirs(dir, exist_ok=True)

  if not validate_dataset():
    return

  if not dependencies_installed:
    print("🏭 Instalando entrenador...")
    t0 = time.time()
    install_trainer()
    t1 = time.time()
    dependencies_installed = True
    print(f"✅ Instalación terminada en {int(t1 - t0)} segundos.")
  else:
    print("✅ Dependencias ya instaladas.")

  if old_model_url != model_url or not model_file or not os.path.exists(model_file):
    print("🔄 Obteniendo modelo...")
    if not download_model():
      print("💥 Error: el modelo que especificó no es válido o está corrupto. Verifique que la URL sea accesible o que la ruta exista dentro de su espacio Lightning.")
      return
    print()
  else:
    print("🔄 Modelo ya disponible.")

  if lr_scheduler_type:
    create_config()
    os.chdir(kohya_dir)
    calculate_rex_steps()
    os.chdir(root_dir)

  create_config()

  print("⭐ Iniciando Entrenador..")

  os.chdir(kohya_dir)
  _run_cmd(f"{venv_python} {train_network} --config_file={config_file} --dataset_config={dataset_config_file}")
  os.chdir(root_dir)

  if not get_ipython().__dict__.get('user_ns', {}).get('_exit_code', False):
    display(Markdown(f"### ✅ ¡Hecho! Tus archivos se encuentran en `{output_folder}`"))

main()
print("🔵 El cuaderno continuará en ejecución. Detén manualmente la sesión de Lightning cuando termines.")
