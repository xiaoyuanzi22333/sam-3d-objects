import sys
from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian

# import inference code
sys.path.append("notebook")
try:
    from inference import Inference, load_image, load_single_mask
except ImportError:
    from .notebook.inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"/mnt/data2/yiming/sam3d_ckpt/sam-3d-objects/checkpoints/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")
