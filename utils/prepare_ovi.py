import torch
import os
import glob
import random
from tqdm.auto import tqdm
import numpy as np
import cv2
import shutil
from tqdm.contrib.concurrent import process_map, thread_map


img_size = 1024
ignore_label = 255
pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

cls_label_names = ['real', 'full_synthetic', 'tampered']

step_real = 25
step_fake = 25
base_video_dir = "/home/peter/faigc/data/ovi"
output_data_dir = '/home/peter/faigc/SIDA/data/ovi'


def prepare_videos_and_masks_threaded(video_fps, split_name):
    def worker(video_fp):
        video_frames = list()
        video_name = os.path.basename(video_fp).split('.')[0]
        video_cap = cv2.VideoCapture(video_fp)
        video_ret, video_frame = video_cap.read()

        frame_idx = 0
        while video_ret:
            video_frames.append((video_name + '_' + str(frame_idx) + '.png', video_frame))
            frame_idx += step_fake
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            video_ret, video_frame = video_cap.read()

        video_cap.release()
        
        for image_fn, image in video_frames:
            cv2.imwrite(os.path.join(output_data_dir, split_name, 'full_synthetic', image_fn), image)

        del video_frames

    
    thread_map(worker, video_fps, max_workers=8)


video_fps = glob.glob(os.path.join(base_video_dir, "**/*.mp4"))

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)

for cls_label_name in cls_label_names:
    os.makedirs(os.path.join(output_data_dir, 'train', cls_label_name), exist_ok=True)
    os.makedirs(os.path.join(output_data_dir, 'test', cls_label_name), exist_ok=True)
os.makedirs(os.path.join(output_data_dir, 'train', 'masks'), exist_ok=True)
os.makedirs(os.path.join(output_data_dir, 'test', 'masks'), exist_ok=True)


prepare_videos_and_masks_threaded(video_fps, 'test')

print(f"train reals: {len(os.listdir(os.path.join(output_data_dir, 'train', 'real')))}")
print(f"train fakes: {len(os.listdir(os.path.join(output_data_dir, 'train', 'full_synthetic')))}")
print(f"train fakes: {len(os.listdir(os.path.join(output_data_dir, 'train', 'tampered')))}")
print(f"train masks: {len(os.listdir(os.path.join(output_data_dir, 'train', 'masks')))}")

print(f"test reals: {len(os.listdir(os.path.join(output_data_dir, 'test', 'real')))}")
print(f"test fakes: {len(os.listdir(os.path.join(output_data_dir, 'test', 'full_synthetic')))}")
print(f"test fakes: {len(os.listdir(os.path.join(output_data_dir, 'test', 'tampered')))}")
print(f"test masks: {len(os.listdir(os.path.join(output_data_dir, 'test', 'masks')))}")
