import sys
import os

# import inference code
sys.path.append("notebook")
try:
    from inference import Inference, load_image, load_masks, make_scene, ready_gaussian_for_video_rendering
except ImportError:
    from src_sam3d_notebook.inference import Inference, load_image, load_masks, make_scene, ready_gaussian_for_video_rendering


# load model
tag = "hf"
config_path = f"/mnt/data2/yiming/sam3d_ckpt/sam-3d-objects/checkpoints/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
IMAGE_PATH = "notebook/images/shutterstock_1980085646/image.png"
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

image = load_image(IMAGE_PATH)
masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")

# run model
outputs = [inference(image, mask, seed=42) for mask in masks]

# export gaussian splat

scene_gs = make_scene(*outputs)
scene_gs = ready_gaussian_for_video_rendering(scene_gs)

# export gaussian splatting (as point cloud)
scene_gs.save_ply(f"demo_multi_object2.ply")
print("Your reconstruction has been saved to splat.ply")
