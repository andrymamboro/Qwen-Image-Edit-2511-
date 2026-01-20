import gradio as gr
import numpy as np
import random
import torch
import spaces

from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
# from optimization import optimize_pipeline_
# from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
# from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
# from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

import math

# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Scheduler configuration for Lightning
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(5),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Initialize scheduler with Lightning config
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

# Load the model pipeline
pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", 
                                                 scheduler=scheduler,
                                                 torch_dtype=dtype).to(device)
pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Edit-2511-Lightning", 
        weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors"
)
pipe.fuse_lora()

# # Apply the same optimizations from the first version
# pipe.transformer.__class__ = QwenImageTransformer2DModel
# pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

# # --- Ahead-of-time compilation ---
# optimize_pipeline_(pipe, image=[Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))], prompt="prompt")

# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

def use_output_as_input(output_images):
    """Convert output images to input format for the gallery"""
    if output_images is None or len(output_images) == 0:
        return []
    return output_images

# --- Main Inference Function (with hardcoded negative prompt) ---
@spaces.GPU()
def infer(
    image_1,
    image_2,
    image_3,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    height=None,
    width=None,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Run image-editing inference using the Qwen-Image-Edit pipeline.

    Parameters:
        images (list): Input images from the Gradio gallery (PIL or path-based).
        prompt (str): Editing instruction (may be rewritten by LLM if enabled).
        seed (int): Random seed for reproducibility.
        randomize_seed (bool): If True, overrides seed with a random value.
        true_guidance_scale (float): CFG scale used by Qwen-Image.
        num_inference_steps (int): Number of diffusion steps.
        height (int | None): Optional output height override.
        width (int | None): Optional output width override.
        rewrite_prompt (bool): Whether to rewrite the prompt using Qwen-2.5-VL.
        num_images_per_prompt (int): Number of images to generate.
        progress: Gradio progress callback.

    Returns:
        tuple: (generated_images, seed_used, UI_visibility_update)
    """
    
    # Hardcode the negative prompt as requested
    negative_prompt = " "
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Load input images into a list of PIL Images
    pil_images = []
    for item in [image_1, image_2, image_3]:
        if item is None: continue
        pil_images.append(item.convert("RGB"))

    if height==256 and width==256:
        height, width = None, None
    print(f"Calling pipeline with prompt: '{prompt}'")
    print(f"Negative Prompt: '{negative_prompt}'")
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}, Size: {width}x{height}")
    

    # Generate the image
    images = pipe(
        image=pil_images if len(pil_images) > 0 else None,
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
    ).images

    # Return images, seed, and make button visible
    return images[0], seed, gr.update(visible=True)

# --- Examples and UI Layout ---
examples = []

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#logo-title {
    text-align: left;
}
#logo-title img {
    width: 40px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
        <div id="logo-title">
            <img src="https://huggingface.co/spaces/obitouchiha88/image_studio/resolve/main/logo.png" alt="Mamboro Ai Logo" width="40" style="display: block; margin: 0 auto;">
                    <h2 style="font-style: italic;color: #5#FFF;margin-top: -27px !important;margin-left: 96px">MAMBORO AI</h2>
        </div>
        """)
        gr.Markdown("""
        Image Studio Edit Menggunakan Qwen-Image-Edit-2511 dan LoRa,☕ https://github.com/andrymamboro)
        """)
    
        with gr.Row():
            with gr.Column():
                image_1 = gr.Image(label="image 1", type="pil", interactive=True)
                with gr.Accordion("More references", open=False):
                    with gr.Row():
                        image_2 = gr.Image(label="image 2", type="pil", interactive=True)
                        image_3 = gr.Image(label="image 3", type="pil", interactive=True)

            with gr.Column():
                result = gr.Image(label="Result", type="pil", interactive=False)
                # Add this button right after the result gallery - initially hidden
                use_output_btn = gr.Button("↗️ Use as image 1", variant="secondary", size="sm", visible=False)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prompt = gr.Text(
                            label="Prompt",
                            show_label=False,
                            placeholder="describe the edit instruction",
                            container=False,
                            lines=5
                    )
                with gr.Row():
                    run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            # Negative prompt UI element is removed here

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():

                true_guidance_scale = gr.Slider(
                    label="True guidance scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=40,
                    step=1,
                    value=4,
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=None,
                )
                
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=None,
                )

        # gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)

    gr.on(
        triggers=[run_button.click],
        fn=infer,
        inputs=[
            image_1,
            image_2,
            image_3,
            prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            height,
            width,
        ],
        outputs=[result, seed, use_output_btn],  # Added use_output_btn to outputs
    )

    # Add the new event handler for the "Use Output as Input" button
    use_output_btn.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[image_1]
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True, show_error=True, share=True)