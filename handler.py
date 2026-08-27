import os
import io
import base64
import torch
import runpod

from diffusers import DiffusionPipeline


MODEL_ID = os.getenv(
    "MODEL_ID",
    "SG161222/RealVisXL_V5.0"
)

print(f"Loading model: {MODEL_ID}")

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe = pipe.to("cuda")

print("RealVisXL V5.0 loaded successfully.")


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

        width = int(data.get("width", 1024))
        height = int(data.get("height", 1024))
        steps = int(data.get("steps", 25))
        guidance_scale = float(
            data.get("guidance_scale", 5.0)
        )

        seed = data.get("seed")

        generator = None

        if seed is not None:
            generator = torch.Generator(
                device="cuda"
            ).manual_seed(int(seed))

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
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
            "seed": seed
        }

    except Exception as e:
        return {
            "error": str(e)
        }


runpod.serverless.start({
    "handler": handler
})
