# %%
import cv2
import cellpose
from cellpose import models
import numpy as np
from skimage.color import rgb2gray
from skimage.measure import regionprops_table, regionprops
# import matplotlib.pyplot as plt
import torch
import pandas as pd
import argparse
import os

# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %%

parser = argparse.ArgumentParser(description="Process a video file for cell analysis.")
parser.add_argument("input_video_path", type=str, help="Path to the input video file")
args = parser.parse_args()

input_video_path = args.input_video_path
print(f'Analysising {input_video_path}')

cap = cv2.VideoCapture(input_video_path)
imgs = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # print(frame, ret)
        # plt.imshow(frame[:, :, 0])
        # plt.show()
        imgs.append(rgb2gray(frame))
    else:
        break

cap.release()

# %%
imgs = np.stack(imgs, axis=0)

# %%
imgs.shape

# %%
cp = models.CellposeModel(gpu=True, device=torch.device(device), pretrained_model=r'Data\Videos\human_20241212_500e')
sz = models.SizeModel(cp, device=torch.device(device))

model = models.Cellpose(gpu=True, device=torch.device(device))
model.cp = cp
model.sz = sz

# %%
masks = np.zeros_like(imgs)

for t in range(len(masks)):
    results = model.eval(imgs[t])
    mask, flow, style, diams = results[0], results[1][0], results[1][1], results[2]
    masks[t] = mask.astype(int)


# %%
# for t in range(len(masks)):
#     plt.imshow(masks[t])
#     plt.show()

# %%
values_to_measure = ['label', 'area', 'centroid', 'bbox']
df = None


for t in range(len(masks)):
    results = regionprops_table(masks[t].astype(int), imgs[t], values_to_measure)
    results['time'] = t
    
    if df is None:
        df = pd.DataFrame(results)
    else:
        df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)

# %%
df

# %%
output_csv_path = os.path.splitext(input_video_path)[0] + '_sperm_measurements.csv'
df.to_csv(output_csv_path)

# %%




