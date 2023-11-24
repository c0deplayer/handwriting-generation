import os
import random
from pathlib import Path
from typing import Tuple

import PIL.Image as ImageModule
import gradio as gr
import lightning.pytorch as pl
import torch
import yaml
from PIL.Image import Image
from dataclass_wizard import YAMLWizard

from configs.settings import COLORS, CONFIGS, MODELS, MODELS_APP


def load_model_and_config(selected_model: str) -> Tuple[pl.LightningModule, YAMLWizard]:
    """
    Loads a model and its configuration based on the selected model name.

    Args:
        selected_model (str): Name of the model to be loaded.

    Returns:
        Tuple[pl.LightningModule, YAMLWizard]: A tuple containing the loaded model and its configuration.
    """

    checkpoint_path = f"model_checkpoints/{selected_model}/model.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = f"configs/{selected_model}/base_gpu.yaml"

    return (
        MODELS[selected_model].load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=device,
        ),
        CONFIGS[selected_model].from_yaml_file(
            file=config_file, decoder=yaml.load, Loader=yaml.Loader
        ),
    )


def handwriting_generation(
    prompt: str,
    selected_style: str,
    color: str,
    selected_model: str,
    save_file: int,
    file_format: str,
    seed: int,
) -> Image:
    """
    Generates handwriting based on the given parameters and model.

    Args:
        prompt (str): Text to be generated in handwriting.
        selected_style (str): The style ID for handwriting generation.
        color (str): Color of the handwriting.
        selected_model (str): The model to use for generation.
        save_file (int): Whether to save the file (1) or not (0).
        file_format (str): Format of the saved file, e.g., 'PNG', 'JPEG'.
        seed (int): Seed for random number generation to ensure reproducibility.

    Returns:
        Image: An image object representing the generated handwriting.

    Raises:
        gr.Error: If the style ID is not within the valid range for the selected model.
    """

    if selected_model == "LatentDiffusion" and not (1 <= int(selected_style) <= 330):
        raise gr.Error("Style ID for Latent Diffusion must be between 1 and 330")
    elif selected_model == "Diffusion" and not (
        style_range[0] <= int(selected_style) <= style_range[1]
    ):
        raise gr.Error(
            f"Style ID for Diffusion must be between {style_range[0]} and {style_range[1]}"
        )

    model, config = load_model_and_config(selected_model)
    model.eval()

    selected_style = int(selected_style) - 1
    save_path = (
        None
        if save_file
        else Path(
            f"{os.getcwd()}/images/{selected_model}/{prompt.replace(' ', '-')}.{file_format.lower()}"
        )
    )

    if selected_model == "LatentDiffusion":
        return model.generate(
            prompt,
            vocab=config.vocab,
            max_text_len=config.max_text_len,
            writer_id=selected_style,
            save_path=save_path,
            color=color,
            seed=seed,
        )

    style_path = f"assets/{asset_dir[selected_style]}"
    fig = model.generate(
        prompt,
        save_path=save_path,
        vocab=config.vocab,
        max_text_len=config.max_text_len,
        style_path=style_path,
        color=color,
        seed=seed,
    )

    return ImageModule.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def dynamic_style_ids_update(selected_model: str) -> gr.Number:
    """
    Dynamically updates the style IDs based on the selected model.

    Args:
        selected_model (str): The model based on which the style IDs are updated.

    Returns:
        gr.Number: A Gradio number component with updated values and visibility based on the selected model.
    """

    if selected_model == "Diffusion":
        return gr.Number(
            value=1,
            label=f"Style ID (values: ({style_range[0]}, {style_range[1]}))",
            visible=True,
        )

    elif selected_model == "LatentDiffusion":
        return gr.Number(
            value=1,
            label="Style ID (values: (1, 330))",
            visible=True,
        )


def dynamic_image_save_update(save_image: int) -> gr.Radio:
    """
    Updates the visibility of the image save option based on the user's choice.

    Args:
        save_image (int): Indicates whether the image is to be saved (1) or not (0).

    Returns:
        gr.Radio: A Gradio radio button component with updated visibility.
    """

    return gr.Radio(visible=False) if save_image else gr.Radio(visible=True)


if __name__ == "__main__":
    asset_dir = os.listdir("assets")
    style_range = (1, 1) if len(asset_dir) == 1 else (1, len(asset_dir))

    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("Handwriting Generation | Jakub Kujawa")

        with gr.Row():
            with gr.Column():
                text = gr.Textbox(value="Handwriting Synthesis in Python", label="Text")
                with gr.Row():
                    model_type = gr.Dropdown(
                        choices=MODELS_APP,
                        value="LatentDiffusion",
                        label="Model type",
                    )

                    style = gr.Number(
                        value=1,
                        label="Style ID (values: (1, 340))",
                    )

                    model_type.change(
                        fn=dynamic_style_ids_update, inputs=model_type, outputs=style
                    )

                    color = gr.Dropdown(
                        choices=COLORS,
                        value="black",
                        label="Colors",
                        info="The color in which the text should be",
                    )

                    save_image = gr.Radio(
                        choices=["Yes", "No"],
                        value="Yes",
                        type="index",
                        label="Save handwriting image",
                    )

                with gr.Row():
                    save_type = gr.Radio(
                        choices=["PNG", "JPEG", "SVG"],
                        value="PNG",
                        type="value",
                        label="File extension to save the image",
                    )

                    save_image.change(
                        fn=dynamic_image_save_update,
                        inputs=save_image,
                        outputs=save_type,
                    )

                    seed = gr.Number(value=random.randint(0, 123456789), label="Seed")

                submit = gr.Button("Submit")

            with gr.Column():
                output = gr.Image(interactive=False, show_label=False)
                submit.click(
                    fn=handwriting_generation,
                    inputs=[
                        text,
                        style,
                        color,
                        model_type,
                        save_image,
                        save_type,
                        seed,
                    ],
                    outputs=output,
                )

    app.launch(debug=True)
