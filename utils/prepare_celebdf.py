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
base_video_dir = "/home/peter/faigc/data/Celeb-DF-v2"
output_data_dir = '/home/peter/faigc/SIDA/data/Celeb-DF-v2'

fake_subdir = 'Celeb-synthesis'
real_subdir = 'Celeb-real'

youtube_real_subdir = 'YouTube-real'


def prepare_videos_and_masks_threaded(real_fake_pairs, youtube_real_fps, split_name):
    def worker1(real_fakes_pair):
        real_fp, fake_fps = real_fakes_pair

        real_video_frames = list()
        real_video_name = os.path.basename(real_fp).split('.')[0]
        real_video_cap = cv2.VideoCapture(real_fp)
        video_ret, video_frame = real_video_cap.read()

        frame_idx = 0
        while video_ret:
            real_video_frames.append((real_video_name + '_' + str(frame_idx) + '.jpg', video_frame))
            frame_idx += step_fake
            real_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            video_ret, video_frame = real_video_cap.read()

        real_video_cap.release()

        fake_video_frames = list()
        masks = list()

        for fake_fp in fake_fps:
            fake_video_frames_ = list()
            
            fake_video_name = os.path.basename(fake_fp).split('.')[0]
            fake_video_cap = cv2.VideoCapture(fake_fp)
            video_ret, video_frame = fake_video_cap.read()

            frame_idx = 0
            while video_ret:
                fake_video_frames_.append((fake_video_name + '_' + str(frame_idx) + '.jpg', video_frame))
                frame_idx += step_fake
                fake_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                video_ret, video_frame = fake_video_cap.read()

            fake_video_cap.release()

            frame_idx = 0
            for (real_fp, real_frame), (fake_fp, fake_frame) in zip(real_video_frames, fake_video_frames_):
                if real_frame.shape != fake_frame.shape:
                    continue

                mask = (fake_frame != real_frame).all(axis=-1).astype(np.uint8) * 255
                mask_dilated = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=3)

                mask_ratio = mask_dilated.sum() / 255 / mask_dilated.size

                if mask_ratio > 0.1:
                    continue    

                fake_video_frames.append((fake_video_name + '_' + str(frame_idx) + '.png', fake_frame))
                masks.append((fake_video_name + '_' + str(frame_idx) + '_mask' + '.png', mask_dilated))

                frame_idx += step_fake

        assert len(fake_video_frames) == len(masks)

        for (image_fn, image) in real_video_frames:
            cv2.imwrite(os.path.join(output_data_dir, split_name, 'real', image_fn), image)
        
        for (image_fn, image), (mask_fn, mask) in zip(fake_video_frames, masks):
            cv2.imwrite(os.path.join(output_data_dir, split_name, 'tampered', image_fn), image)
            cv2.imwrite(os.path.join(output_data_dir, split_name, 'masks', mask_fn), mask)

        del real_video_frames
        del fake_video_frames
        del masks

    def worker2(youtube_real_fp):
        real_video_name = os.path.basename(youtube_real_fp).split('.')[0]
        real_video_cap = cv2.VideoCapture(youtube_real_fp)
        video_ret, video_frame = real_video_cap.read()

        frame_idx = 0
        while video_ret:
            fn = real_video_name + '_' + str(frame_idx) + '.jpg'
            cv2.imwrite(os.path.join(output_data_dir, split_name, 'real', fn), video_frame)
            frame_idx += step_real
            real_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            video_ret, video_frame = real_video_cap.read()

        real_video_cap.release()
    
    thread_map(worker1, real_fake_pairs, max_workers=8)
    thread_map(worker2, youtube_real_fps, max_workers=8)


def match_real_to_fake_fp(real_fp, fake_fp):
    real_fn = os.path.basename(real_fp)
    fake_fn = os.path.basename(fake_fp)
    fake_fn_parts = fake_fn.split('_')
    fake_fn_transformed = fake_fn_parts[0] + '_' + fake_fn_parts[2]

    return real_fn == fake_fn_transformed


with open(os.path.join(base_video_dir, 'List_of_testing_videos.txt')) as f:
    test_fns = list(map(lambda x: os.path.basename(x.strip()), f.readlines()))


fake_fps = glob.glob(os.path.join(base_video_dir, fake_subdir, "*.mp4"))
real_fps = glob.glob(os.path.join(base_video_dir, real_subdir, "*.mp4"))

reals_and_fakes = list()

for real_fp in tqdm(real_fps, desc='matching reals and fakes'):
    this_real_fakes = list(filter(lambda fake_fp: match_real_to_fake_fp(real_fp, fake_fp), fake_fps))
    if len(this_real_fakes) > 0:
        real_fn = os.path.basename(real_fp)
        fake_fns = list(map(lambda x: os.path.basename(x), this_real_fakes))

        reals_and_fakes.append((real_fp, this_real_fakes))

random.shuffle(reals_and_fakes)
train_reals_and_fakes = reals_and_fakes[:int(len(reals_and_fakes) * 0.9)]
test_reals_and_fakes = reals_and_fakes[int(len(reals_and_fakes) * 0.9):]

youtube_real_fps = glob.glob(os.path.join(base_video_dir, youtube_real_subdir, "*.mp4"))

random.shuffle(youtube_real_fps)
train_youtube_real_fps = youtube_real_fps[:int(len(youtube_real_fps) * 0.9)]
test_youtube_real_fps = youtube_real_fps[int(len(youtube_real_fps) * 0.9):]

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)

for cls_label_name in cls_label_names:
    os.makedirs(os.path.join(output_data_dir, 'train', cls_label_name), exist_ok=True)
    os.makedirs(os.path.join(output_data_dir, 'test', cls_label_name), exist_ok=True)
os.makedirs(os.path.join(output_data_dir, 'train', 'masks'), exist_ok=True)
os.makedirs(os.path.join(output_data_dir, 'test', 'masks'), exist_ok=True)


prepare_videos_and_masks_threaded(test_reals_and_fakes, test_youtube_real_fps, 'test')
prepare_videos_and_masks_threaded(train_reals_and_fakes, train_youtube_real_fps, 'train')

print(f"train reals: {len(os.listdir(os.path.join(output_data_dir, 'train', 'real')))}")
print(f"train fakes: {len(os.listdir(os.path.join(output_data_dir, 'train', 'tampered')))}")
print(f"train masks: {len(os.listdir(os.path.join(output_data_dir, 'train', 'masks')))}")

print(f"test reals: {len(os.listdir(os.path.join(output_data_dir, 'test', 'real')))}")
print(f"test fakes: {len(os.listdir(os.path.join(output_data_dir, 'test', 'tampered')))}")
print(f"test masks: {len(os.listdir(os.path.join(output_data_dir, 'test', 'masks')))}")
