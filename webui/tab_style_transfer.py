import os
from PIL import Image
import gradio as gr


def create_interface_style_transfer(runner):
    with gr.Blocks():
        with gr.Row():
            gr.Markdown('1. Upload the content and style images as inputs.\n'
                        '2. (Optional) Customize the configurations below as needed.\n'
                        '3. Cilck `Run` to start transfer.')
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    content_image = gr.Image(label='Input Content Image', type='pil', interactive=True,
                                             value=Image.open('examples/c1.jpg').convert('RGB') if os.path.exists('examples/c1.jpg') else None)
                    style_image = gr.Image(label='Input Style Image', type='pil', interactive=True,
                                           value=Image.open('examples/s1.jpg').convert('RGB') if os.path.exists('examples/s1.jpg') else None)
            
                run_button = gr.Button(value='Run')

                with gr.Accordion('Options', open=True):
                    seed = gr.Number(label='Seed', value=2025, precision=0, minimum=0, maximum=2**31)
                    num_steps = gr.Slider(label='Number of Steps', minimum=1, maximum=1000, value=200, step=1)
                    lr = gr.Slider(label='Learning Rate', minimum=0.01, maximum=0.5, value=0.05, step=0.01)
                    content_weight = gr.Slider(label='Content Weight', minimum=0., maximum=1., value=0.25, step=0.001)
                    mixed_precision = gr.Radio(choices=['bf16', 'no'], value='bf16', label='Mixed Precision')
                    
                    base_model_list = ['stable-diffusion-v1-5/stable-diffusion-v1-5',]
                    model = gr.Radio(choices=base_model_list, label='Select a Base Model', value='stable-diffusion-v1-5/stable-diffusion-v1-5')

            with gr.Column():
                gr.Markdown('#### Output Image:\n')
                result_gallery = gr.Gallery(label='Output', elem_id='gallery', columns=2, height='auto', preview=True)
        
        ips = [content_image, style_image, seed, num_steps, lr, content_weight, mixed_precision, model]

        run_button.click(fn=runner.run_style_transfer, inputs=ips, outputs=[result_gallery])
