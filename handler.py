import os
import io
import base64
import requests
import torch
import runpod

from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image


MODEL_ID = os.getenv(
    "MODEL_ID",
    "SG161222/RealVisXL_V5.0"
)

IP_ADAPTER_REPO = os.getenv(
    "IP_ADAPTER_REPO",
    "h94/IP-Adapter"
)

IP_ADAPTER_SUBFOLDER = os.getenv(
    "IP_ADAPTER_SUBFOLDER",
    "sdxl_models"
)

IP_ADAPTER_WEIGHT = os.getenv(
    "IP_ADAPTER_WEIGHT",
    "ip-adapter-plus-face_sdxl_vit-h.safetensors"
)

# Encoder ViT-H correto para os pesos SDXL do IP-Adapter.
# A pasta padrão "image_encoder" contém o ViT-L (SD 1.5) e causa o erro
# "mat1 and mat2 shapes cannot be multiplied" quando há imagem de referência.
IP_ADAPTER_ENCODER_FOLDER = os.getenv(
    "IP_ADAPTER_ENCODER_FOLDER",
    "models/image_encoder"
)

DEFAULT_IP_ADAPTER_SCALE = float(
    os.getenv("IP_ADAPTER_SCALE", "0.60")
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading model: {MODEL_ID}")
print(f"Device: {DEVICE}")

# --------------------------------------------------
# TEXT TO IMAGE PIPELINE
# --------------------------------------------------
txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    use_safetensors=True
)

txt2img_pipe = txt2img_pipe.to(DEVICE)

print("Loading IP-Adapter...")

txt2img_pipe.load_ip_adapter(
    IP_ADAPTER_REPO,
    subfolder=IP_ADAPTER_SUBFOLDER,
    weight_name=IP_ADAPTER_WEIGHT,
    image_encoder_folder=IP_ADAPTER_ENCODER_FOLDER,
)

txt2img_pipe.set_ip_adapter_scale(DEFAULT_IP_ADAPTER_SCALE)

# --------------------------------------------------
# IMAGE TO IMAGE PIPELINE
# Reaproveita os mesmos componentes do txt2img
# --------------------------------------------------
img2img_pipe = AutoPipelineForImage2Image.from_pipe(txt2img_pipe)
img2img_pipe = img2img_pipe.to(DEVICE)
img2img_pipe.set_ip_adapter_scale(DEFAULT_IP_ADAPTER_SCALE)

print("RealVisXL V5.0 + IP-Adapter loaded successfully.")


def decode_base64_image(raw_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(raw_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def resolve_image(raw):
    """
    Aceita:
    - URL http/https
    - data:image/...;base64,...
    - base64 puro
    - path local (se existir no worker)
    """
    if not raw:
        return None

    try:
        # URL pública
        if isinstance(raw, str) and (
            raw.startswith("http://") or raw.startswith("https://")
        ):
            response = requests.get(raw, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")

        # Data URI
        if isinstance(raw, str) and raw.startswith("data:image"):
            payload = raw.split(",", 1)[1]
            return decode_base64_image(payload)

        # Arquivo local dentro do container
        if isinstance(raw, str) and os.path.exists(raw):
            return Image.open(raw).convert("RGB")

        # Base64 puro
        if isinstance(raw, str):
            return decode_base64_image(raw)

        return None

    except Exception as e:
        print(f"Failed to resolve image: {e}")
        return None


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_generator(seed):
    if seed is None:
        return None

    return torch.Generator(device=DEVICE).manual_seed(int(seed))


def handler(job):
    try:
        data = job.get("input", {})

        prompt = data.get("prompt")
        if not prompt:
            return {"error": "prompt is required"}

        negative_prompt = data.get(
            "negative_prompt",
            "worst quality, low quality, blurry, deformed, bad anatomy, "
            "bad hands, extra fingers, missing fingers, mutated hands, "
            "poorly drawn face, asymmetrical eyes, cgi, 3d render, cartoon, "
            "anime, doll face, plastic skin, overprocessed skin"
        )

        width = int(data.get("width", 768))
        height = int(data.get("height", 1024))

        steps = int(
            data.get("steps", data.get("num_inference_steps", 28))
        )

        guidance_scale = float(data.get("guidance_scale", 5.0))

        seed = data.get("seed")
        generator = build_generator(seed)

        # strength do img2img
        strength = float(data.get("strength", 0.35))

        # intensidade da referência facial no IP-Adapter
        ip_adapter_scale = float(
            data.get("ip_adapter_scale", data.get("reference_strength", DEFAULT_IP_ADAPTER_SCALE))
        )

        txt2img_pipe.set_ip_adapter_scale(ip_adapter_scale)
        img2img_pipe.set_ip_adapter_scale(ip_adapter_scale)

        # aceita vários nomes
        init_image_raw = (
            data.get("init_image")
            or data.get("input_image")
            or data.get("image")
        )

        reference_image_raw = (
            data.get("reference_image")
            or (data["images"][0] if data.get("images") else None)
        )

        init_image = resolve_image(init_image_raw)
        reference_image = resolve_image(reference_image_raw)

        if init_image is not None:
            init_image = init_image.resize(
                (width, height),
                Image.Resampling.LANCZOS
            )

        ip_adapter_image = None
        if reference_image is not None:
            ip_adapter_image = [reference_image]
            print("Reference image loaded for IP-Adapter.")
        else:
            print("No reference image provided.")

        common_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        # --------------------------------------------------
        # IMG2IMG MODE
        # --------------------------------------------------
        if init_image is not None:
            mode = "img2img"

            run_kwargs = {
                **common_kwargs,
                "image": init_image,
                "strength": strength,
            }

            if ip_adapter_image is not None:
                run_kwargs["ip_adapter_image"] = ip_adapter_image

            with torch.inference_mode():
                result = img2img_pipe(**run_kwargs)

        # --------------------------------------------------
        # TEXT2IMG MODE
        # --------------------------------------------------
        else:
            mode = "text2img"

            run_kwargs = {
                **common_kwargs,
                "width": width,
                "height": height,
            }

            if ip_adapter_image is not None:
                run_kwargs["ip_adapter_image"] = ip_adapter_image

            with torch.inference_mode():
                result = txt2img_pipe(**run_kwargs)

        output_image = result.images[0]
        image_base64 = image_to_base64(output_image)

        return {
            "image": image_base64,
            "mode": mode,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "strength": strength if mode == "img2img" else None,
            "seed": seed,
            "used_reference": reference_image is not None,
            "used_init_image": init_image is not None,
            "ip_adapter_scale": ip_adapter_scale,
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"error": str(e)}


runpod.serverless.start({
    "handler": handler
})
