import gradio as gr
from webui import (
    create_interface_texture_synthesis,
    create_interface_style_t2i,
    create_interface_style_transfer,
)
from webui.runner import Runner

import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"


def main():
    runner = Runner()

    with gr.Blocks(analytics_enabled=False,
                   title='Attention Distillation',
                   ) as demo:

        md_txt = "# Attention Distillation" \
                 "\nOfficial demo of the paper [Attention Distillation: A Unified Approach to Visual Characteristics Transfer](https://arxiv.org/abs/2502.20235)"
        gr.Markdown(md_txt)

        with gr.Tabs(selected='tab_style_transfer'):
            with gr.TabItem("Style Transfer", id='tab_style_transfer'):
                create_interface_style_transfer(runner=runner)

            with gr.TabItem("Style-Specific Text-to-Image Generation", id='tab_style_t2i'):
                create_interface_style_t2i(runner=runner)

            with gr.TabItem("Texture Synthesis", id='tab_texture_syn'):
                create_interface_texture_synthesis(runner=runner)

        # demo.queue().launch()
        demo.launch(share=False, debug=False)


if __name__ == '__main__':
    main()
