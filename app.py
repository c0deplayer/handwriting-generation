import os
from pathlib import Path
from typing import Tuple

import PIL.Image as ImageModule
import gradio as gr
import lightning.pytorch as pl
import torch
from PIL.Image import Image
from dataclass_wizard import YAMLWizard
from ruamel import yaml

from configs.settings import CONFIGS, MODELS, COLORS, MODELS_APP


def load_model_and_config(selected_model: str) -> Tuple[pl.LightningModule, YAMLWizard]:
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
) -> Image:
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
            writer_id=selected_style,
            save_path=save_path,
            color=color,
        )
    # TODO: RNN Model
    elif selected_model == "RNN":
        raise gr.Error("Not implemented")

    style_path = f"assets/{asset_dir[selected_style]}"
    fig = model.generate(
        prompt,
        save_path=save_path,
        vocab=config.vocab,
        style_path=style_path,
        color=color,
    )

    return ImageModule.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def dynamic_style_ids_update(selected_model: str) -> gr.Number:
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
    else:
        return gr.Number(visible=False)


def dynamic_image_save_update(save_image: int) -> gr.Radio:
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

                submit = gr.Button("Submit")

            with gr.Column():
                output = gr.Image(interactive=False, show_label=False)
                submit.click(
                    fn=handwriting_generation,
                    inputs=[text, style, color, model_type, save_image, save_type],
                    outputs=output,
                )

    app.launch(debug=True)
