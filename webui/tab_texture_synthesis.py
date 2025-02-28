import os
from PIL import Image
import gradio as gr


def create_interface_texture_synthesis(runner):
    with gr.Blocks():
        with gr.Row():
            gr.Markdown('1. Upload the texture image as input.\n'
                        '2. (Optional) Customize the configurations below as needed.\n'
                        '3. Cilck `Run` to start synthesis.')
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    texture_image = gr.Image(label='Input Texture Image', type='pil', interactive=True,
                                           value=Image.open('examples/s1.jpg').convert('RGB') if os.path.exists('examples/s1.jpg') else None)
            
                run_button = gr.Button(value='Run')

                with gr.Accordion('Options', open=True):
                    height = gr.Number(label='Height', value=512, precision=0, minimum=2, maximum=4096)
                    width = gr.Number(label='Width', value=1024, precision=0, minimum=2, maximum=4096)
                    seed = gr.Number(label='Seed', value=2025, precision=0, minimum=0, maximum=2**31)
                    num_steps = gr.Slider(label='Number of Steps', minimum=1, maximum=1000, value=200, step=1)
                    iterations = gr.Slider(label='Iterations', minimum=0, maximum=10, value=2, step=1)
                    lr = gr.Slider(label='Learning Rate', minimum=0.01, maximum=0.5, value=0.05, step=0.01)
                    mixed_precision = gr.Radio(choices=['bf16', 'no'], value='bf16', label='Mixed Precision')
                    num_images_per_prompt = gr.Slider(label='Num Images Per Prompt', minimum=1, maximum=10, value=1, step=1)
                    
                    base_model_list = ['stable-diffusion-v1-5/stable-diffusion-v1-5',]
                    model = gr.Radio(choices=base_model_list, label='Select a Base Model', value='stable-diffusion-v1-5/stable-diffusion-v1-5')
                    synthesis_way = gr.Radio(['Sampling', 'MultiDiffusion'], label='Synthesis Way', value='MultiDiffusion')

            with gr.Column():
                gr.Markdown('#### Output Image:\n')
                result_gallery = gr.Gallery(label='Output', elem_id='gallery', columns=2, height='auto', preview=True)
        
        ips = [texture_image, height, width, seed, num_steps, iterations, lr, mixed_precision, num_images_per_prompt, synthesis_way,model]

        run_button.click(fn=runner.run_texture_synthesis, inputs=ips, outputs=[result_gallery])

