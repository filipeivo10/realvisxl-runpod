import os
import io
import base64
import torch
import runpod

from diffusers import DiffusionPipeline
from diffusers.utils import load_image


MODEL_ID = os.getenv(
    "MODEL_ID",
    "SG161222/RealVisXL_V5.0"
)

IP_ADAPTER_REPO = os.getenv(
    "IP_ADAPTER_REPO",
    "h94/IP-Adapter"
)

IP_ADAPTER_WEIGHT = os.getenv(
    "IP_ADAPTER_WEIGHT",
    "ip-adapter-plus_sdxl_vit-h.safetensors"
)

# 0.0 = ignora a referência | 1.0 = copia demais a imagem
# 0.5-0.6 mantém o rosto e ainda obedece ao prompt
IP_ADAPTER_SCALE = float(os.getenv("IP_ADAPTER_SCALE", "0.55"))

print(f"Loading model: {MODEL_ID}")

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe = pipe.to("cuda")

print("Loading IP-Adapter for face reference...")

pipe.load_ip_adapter(
    IP_ADAPTER_REPO,
    subfolder="sdxl_models",
    weight_name=IP_ADAPTER_WEIGHT,
)
pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

print("RealVisXL V5.0 + IP-Adapter loaded successfully.")


def resolve_reference(raw):
    """Aceita URL http(s) ou data URI base64 enviado pelo backend."""
    if not raw:
        return None
    try:
        if raw.startswith("data:"):
            payload = raw.split(",", 1)[1]
            raw_bytes = base64.b64decode(payload)
            return load_image(io.BytesIO(raw_bytes)).convert("RGB")
        return load_image(raw).convert("RGB")
    except Exception as e:
        print(f"Failed to load reference image: {e}")
        return None


def handler(job):
    try:
        data = job.get("input", {})

        prompt = data.get("prompt")

        if not prompt:
            return {
                "error": "prompt is required"
            }

        negative_prompt = data.get(
            "negative_prompt",
            "worst quality, low quality, blurry, deformed, "
            "bad anatomy, bad hands, extra fingers, missing fingers"
        )

        width = int(data.get("width", 768))
        height = int(data.get("height", 1024))

        # aceita os dois nomes: a rota manda num_inference_steps
        steps = int(
            data.get("steps", data.get("num_inference_steps", 25))
        )

        guidance_scale = float(
            data.get("guidance_scale", 5.0)
        )

        seed = data.get("seed")

        generator = None

        if seed is not None:
            generator = torch.Generator(
                device="cuda"
            ).manual_seed(int(seed))

        # referência facial: init_image, reference_image ou images[0]
        reference_raw = (
            data.get("init_image")
            or data.get("reference_image")
            or (data["images"][0] if data.get("images") else None)
        )

        reference = resolve_reference(reference_raw)

        ip_image = None
        if reference is not None:
            ip_image = [reference]
            print("IP-Adapter enabled with reference image.")
        else:
            print("No usable reference image; text-to-image only.")

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                ip_adapter_image=ip_image,
            )

        image = result.images[0]

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=95
        )

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return {
            "image": image_base64,
            "width": width,
            "height": height,
            "seed": seed,
            "used_reference": reference is not None,
        }

    except Exception as e:
        return {
            "error": str(e)
        }


runpod.serverless.start({
    "handler": handler
})
