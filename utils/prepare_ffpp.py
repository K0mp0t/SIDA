import torch
import os
import glob
import random
from tqdm.auto import tqdm
import numpy as np
import cv2
import shutil
from tqdm.contrib.concurrent import thread_map


img_size = 1024
ignore_label = 255
pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


step = 25
base_video_dir = "/home/peter/faigc/data/ff++"
output_data_dir = '/home/peter/faigc/SIDA/data/ff++'

subsets_to_use = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'original']
cls_label_names = ['real', 'full_synthetic', 'tampered']


def prepare_videos_and_masks_threaded(video_fps, split_name):
    def worker(video_fp):
        if 'original' in video_fp:
            label = 0
            extension = '.jpg'
        else:
            label = 2
            extension = '.png'
        
        video_frames = list()
        video_subset = list(filter(lambda ss: ss in video_fp, subsets_to_use))[0]
        video_name = video_subset + '_' + os.path.basename(video_fp).split('.')[0]

        video_cap = cv2.VideoCapture(video_fp)
        video_ret, video_frame = video_cap.read()

        frame_idx = 0
        while video_ret:
            video_frames.append((video_name + '_' + str(frame_idx) + extension, video_frame))
            frame_idx += step
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            video_ret, video_frame = video_cap.read()

        video_cap.release()

        if label == 0 or label == 1:
            mask_frames = list([(None, None) for _, f in video_frames])
        elif label == 2:
            mask_frames = list()
            mask_fp = video_fp.replace('videos', 'masks')

            mask_cap = cv2.VideoCapture(mask_fp)
            mask_ret, mask_frame = mask_cap.read()

            frame_idx = 0
            while mask_ret:
                mask_frames.append((video_name + '_' + str(frame_idx) + '_mask' + '.png', mask_frame))
                frame_idx += step
                mask_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                mask_ret, mask_frame = mask_cap.read()
            mask_cap.release()
        else:
            raise NotImplementedError()

        assert len(video_frames) == len(mask_frames)

        for (image_fn, image), (mask_fn, mask) in zip(video_frames, mask_frames):
            cls_label_name = cls_label_names[label]

            cv2.imwrite(os.path.join(output_data_dir, split_name, cls_label_name, image_fn), image)
            if label == 2:
                cv2.imwrite(os.path.join(output_data_dir, split_name, 'masks', mask_fn), mask)

        del video_frames
        del mask_frames
    
    thread_map(worker, video_fps, max_workers=8)


def read_videos_and_masks(video_fps):
    images = list()
    masks = list()
    cls_labels = list()

    for video_fp in tqdm(video_fps, desc='reading videos'):
        if 'original' in video_fp:
            label = 0
            extension = '.jpg'
        else:
            label = 2
            extension = '.png'
        
        video_frames = list()
        video_subset = list(filter(lambda ss: ss in video_fp, subsets_to_use))[0]
        video_name = video_subset + '_' + os.path.basename(video_fp).split('.')[0]

        video_cap = cv2.VideoCapture(video_fp)
        video_ret, video_frame = video_cap.read()

        frame_idx = 0
        while video_ret:
            video_frames.append((video_name + '_' + str(frame_idx) + extension, video_frame))
            frame_idx += step
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            video_ret, video_frame = video_cap.read()

        video_cap.release()

        if label == 0 or label == 1:
            mask_frames = list([np.zeros_like(f) for _, f in video_frames])
        elif label == 2:
            mask_frames = list()

            mask_fp = video_fp.replace('videos', 'masks')

            mask_cap = cv2.VideoCapture(mask_fp)
            mask_ret, mask_frame = mask_cap.read()

            frame_idx = 0
            while mask_ret:
                mask_frames.append((video_name + '_' + str(frame_idx) + '_mask' + '.png', mask_frame))
                frame_idx += step
                mask_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                mask_ret, mask_frame = mask_cap.read()
            mask_cap.release()
        else:
            raise NotImplementedError()

        assert len(video_frames) == len(mask_frames)

        images.extend(video_frames)
        masks.extend(mask_frames)
        cls_labels.extend([label] * len(video_frames))

    return images, masks, cls_labels


def save_dataset_split(base_save_dir, split_name, images, masks, cls_labels):
    if os.path.exists(base_save_dir):
        shutil.rmtree(base_save_dir)

    for cls_label_name in cls_label_names:
        os.makedirs(os.path.join(base_save_dir, split_name, cls_label_name))
    os.makedirs(os.path.join(base_save_dir, split_name, 'masks'))

    for (image_fn, image), (mask_fn, mask), cls_label in zip(images, masks, cls_labels):
        cls_label_name = cls_label_names[cls_label]

        cv2.imwrite(os.path.join(base_save_dir, split_name, cls_label_name, image_fn), image)
        if cls_label == 2:
            cv2.imwrite(os.path.join(base_save_dir, split_name, 'masks', mask_fn), mask)


video_fps = glob.glob(os.path.join(base_video_dir, '**/videos/*.mp4'))
video_fps = list(filter(lambda fp: any(sn in fp for sn in subsets_to_use), video_fps))

random.shuffle(video_fps)
train_video_fps = video_fps[:int(len(video_fps) * 0.9)]
test_video_fps = video_fps[int(len(video_fps) * 0.9):]

# if os.path.exists(output_data_dir):
#     shutil.rmtree(output_data_dir)

for cls_label_name in cls_label_names:
    os.makedirs(os.path.join(output_data_dir, 'train', cls_label_name))
    os.makedirs(os.path.join(output_data_dir, 'test', cls_label_name))
os.makedirs(os.path.join(output_data_dir, 'train', 'masks'))
os.makedirs(os.path.join(output_data_dir, 'test', 'masks'))

# train_images, train_masks, train_cls_labels = read_videos_and_masks(train_video_fps)
# test_images, test_masks, test_cls_labels = read_videos_and_masks(test_video_fps)

prepare_videos_and_masks_threaded(train_video_fps, 'train')
prepare_videos_and_masks_threaded(test_video_fps, 'test')
