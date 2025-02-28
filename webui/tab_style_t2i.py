import os
from PIL import Image
import gradio as gr


def create_interface_style_t2i(runner):
    with gr.Blocks():
        with gr.Row():
            gr.Markdown('1. Upload the style image and text your prompt.\n'
                        '2. Choose the generative model.\n'
                        '3. (Optional) Customize the configurations below as needed.\n'
                        '4. Cilck `Run` to start generation.')
        
        with gr.Row():
            with gr.Column():
                style_image = gr.Image(label='Input Style Image', type='pil', interactive=True,
                                        value=Image.open('examples/s1.jpg').convert('RGB') if os.path.exists('examples/s1.jpg') else None)
                prompt = gr.Textbox(label='Prompt', value='A rocket')
                negative_prompt = gr.Textbox(label='Negative Prompt', value='')

                base_model_list = ['stable-diffusion-v1-5/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-2-1', 'stabilityai/stable-diffusion-xl-base-1.0']
                model = gr.Radio(choices=base_model_list, label='Select a Base Model', value='stabilityai/stable-diffusion-xl-base-1.0')

                run_button = gr.Button(value='Run')

            with gr.Column():
                with gr.Accordion('Options', open=True):
                    guidance_scale = gr.Slider(label='Guidance Scale', minimum=1., maximum=30., value=7.5, step=0.1)
                    height = gr.Number(label='Height', value=1024, precision=0, minimum=2, maximum=4096)
                    width = gr.Number(label='Width', value=1024, precision=0, minimum=2, maximum=4096)
                    seed = gr.Number(label='Seed', value=2025, precision=0, minimum=0, maximum=2**31)
                    num_steps = gr.Slider(label='Number of Steps', minimum=1, maximum=1000, value=50, step=1)
                    iterations = gr.Slider(label='Iterations', minimum=0, maximum=10, value=2, step=1)
                    lr = gr.Slider(label='Learning Rate', minimum=0.01, maximum=0.5, value=0.015, step=0.001)
                    num_images_per_prompt = gr.Slider(label='Num Images Per Prompt', minimum=1, maximum=10, value=1, step=1)
                    mixed_precision = gr.Radio(choices=['bf16', 'no'], value='bf16', label='Mixed Precision')
                    is_adain = gr.Checkbox(label='Adain', value=True,)
                    
            with gr.Column():
                gr.Markdown('#### Output Image:\n')
                result_gallery = gr.Gallery(label='Output', elem_id='gallery', columns=2, height='auto', preview=True)
        
        ips = [style_image, prompt, negative_prompt, guidance_scale, height, width, seed, num_steps, iterations, lr, num_images_per_prompt, mixed_precision, is_adain, model]

        run_button.click(fn=runner.run_style_t2i_generation, inputs=ips, outputs=[result_gallery])

