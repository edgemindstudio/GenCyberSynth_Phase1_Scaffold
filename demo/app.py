# demo/app.py — tiny viewer for latest grids (CPU friendly)
from pathlib import Path
import gradio as gr

ART = Path("artifacts/preview_grids")
MODELS = ["gan","vae","gaussianmixture","diffusion","autoregressive","restrictedboltzmann","maskedautoflow"]

def list_grids():
    items = []
    for m in MODELS:
        p = ART / f"{m}_grid.png"
        if p.exists():
            items.append((m, str(p)))
    return items or [("no grids yet", None)]

def ui_load(model):
    if not model or model == "no grids yet":
        return None
    p = ART / f"{model}_grid.png"
    return str(p) if p.exists() else None

with gr.Blocks() as demo:
    gr.Markdown("# Synth Preview Grids")
    choices = [m for m, _ in list_grids()]
    dd = gr.Dropdown(choices=choices, value=(choices[0] if choices else None), label="Model")
    img = gr.Image(label="Grid", interactive=False)
    dd.change(fn=ui_load, inputs=dd, outputs=img)
    # initial render
    if choices:
        img.value = ui_load(choices[0])

if __name__ == "__main__":
    demo.launch()
