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

cls_label_names = ['real', 'full_synthetic', 'tampered']

step_real = 50
step_fake = 70
base_video_dir = '/home/peter/faigc/data/faigc_v4'
output_data_dir = '/home/peter/faigc/SIDA/data/faigc_v4'


def prepare_videos_and_masks_threaded(video_fps, split_name):
    def worker(video_fp):
        if 'real' in video_fp:
            classname = 'real'
            step = step_real
        elif 'fake' in video_fp:
            classname = 'full_synthetic'
            step = step_fake
        else:
            raise NotImplementedError()

        video_frames = list()
        video_name = os.path.basename(video_fp).split('.')[0]
        video_cap = cv2.VideoCapture(video_fp)
        video_ret, video_frame = video_cap.read()

        total_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)

        frame_idx = 0
        while video_ret and frame_idx < total_frames:
            ext = '.jpg' if classname == 'real' else '.png'
            video_frames.append((video_name + '_' + str(frame_idx) + ext, video_frame))
            frame_idx += step
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            video_ret, video_frame = video_cap.read()

        video_cap.release()
        
        for image_fn, image in video_frames:
            cv2.imwrite(os.path.join(output_data_dir, split_name, classname, image_fn), image)

        del video_frames

    
    thread_map(worker, video_fps, max_workers=8)


video_fps = glob.glob(os.path.join(base_video_dir, "**/*.mp4"), recursive=True)
random.shuffle(video_fps)

train_fps = video_fps[:int(len(video_fps)*0.88)]
validation_fps = video_fps[int(len(video_fps)*0.88):int(len(video_fps)*0.91)]
test_fps = video_fps[int(len(video_fps)*0.91):]

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)

for cls_label_name in cls_label_names:
    os.makedirs(os.path.join(output_data_dir, 'train', cls_label_name), exist_ok=True)
    os.makedirs(os.path.join(output_data_dir, 'validation', cls_label_name), exist_ok=True)
    os.makedirs(os.path.join(output_data_dir, 'test', cls_label_name), exist_ok=True)
os.makedirs(os.path.join(output_data_dir, 'train', 'masks'), exist_ok=True)
os.makedirs(os.path.join(output_data_dir, 'validation', 'masks'), exist_ok=True)
os.makedirs(os.path.join(output_data_dir, 'test', 'masks'), exist_ok=True)


prepare_videos_and_masks_threaded(train_fps, 'train')
prepare_videos_and_masks_threaded(validation_fps, 'validation')
prepare_videos_and_masks_threaded(test_fps, 'test')

print(f"train reals: {len(os.listdir(os.path.join(output_data_dir, 'train', 'real')))}")
print(f"train fakes: {len(os.listdir(os.path.join(output_data_dir, 'train', 'full_synthetic')))}")
print(f"train fakes: {len(os.listdir(os.path.join(output_data_dir, 'train', 'tampered')))}")
print(f"train masks: {len(os.listdir(os.path.join(output_data_dir, 'train', 'masks')))}")

print(f"validation reals: {len(os.listdir(os.path.join(output_data_dir, 'validation', 'real')))}")
print(f"validation fakes: {len(os.listdir(os.path.join(output_data_dir, 'validation', 'full_synthetic')))}")
print(f"validation fakes: {len(os.listdir(os.path.join(output_data_dir, 'validation', 'tampered')))}")
print(f"validation masks: {len(os.listdir(os.path.join(output_data_dir, 'validation', 'masks')))}")

print(f"test reals: {len(os.listdir(os.path.join(output_data_dir, 'test', 'real')))}")
print(f"test fakes: {len(os.listdir(os.path.join(output_data_dir, 'test', 'full_synthetic')))}")
print(f"test fakes: {len(os.listdir(os.path.join(output_data_dir, 'test', 'tampered')))}")
print(f"test masks: {len(os.listdir(os.path.join(output_data_dir, 'test', 'masks')))}")
