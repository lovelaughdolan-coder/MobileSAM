import torch
import numpy as np
from mobile_sam import sam_model_registry

model = sam_model_registry['vit_t']()
# FORCE CPU!
model.to('cpu')
model.eval()

img = np.zeros((1024, 1024, 3), dtype=np.uint8)
from mobile_sam import SamPredictor
predictor = SamPredictor(model)
print("Setting image...")
predictor.set_image(img)
print("Image set. Predicting...")
out = predictor.predict(
    point_coords=np.array([[500, 500]]),
    point_labels=np.array([1]),
    multimask_output=True,
)
print("Predict success!")
