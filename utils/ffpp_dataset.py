import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from tqdm.auto import tqdm

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)

def ffpp_collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, cls_token_idx=None
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    cls_labels_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    has_text_description = []
    
    # Process batch items
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        cls_labels,
        resize,
        questions,
        sampled_classes,
        inference,
        has_text,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        masks_list.append(masks.float())
        label_list.append(label)
        cls_labels_list.append(torch.tensor(cls_labels))
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        has_text_description.append(has_text)

    # Handle image tokens
    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            conversation_list[i] = conversation_list[i].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    # Pre-calculate original lengths before padding
    original_input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    original_lengths = [len(ids) for ids in original_input_ids]

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        original_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Process targets using original lengths
    targets = []
    for i, conversation in enumerate(conversation_list):
        if has_text_description[i]:
            target = input_ids[i].clone()
        else:
            target = torch.full_like(input_ids[i], IGNORE_INDEX)
        targets.append(target)

    targets = torch.stack(targets)
    conv = conversation_lib.default_conversation.copy()
    
    # Set separator based on conversation type
    sep = conv.sep + conv.roles[1] + ": " if conv_type == "llava_v1" else "[/INST] "
    
    # Process each conversation using original lengths
    for idx, (conversation, target, orig_len) in enumerate(zip(conversation_list, targets, original_lengths)):
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        
        for i, rou in enumerate(rounds):
            if rou == "":
                break
                
            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"Warning: Unexpected format in conversation {idx}")
                continue
                
            parts[0] += sep
            
            # Calculate lengths
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        
        # Use original length for verification
        total_len = orig_len
        
        if cur_len != total_len:
            print(f"Length mismatch in conversation {idx}:")
            print(f"cur_len: {cur_len}, total_len: {total_len}")
            print(f"conversation: {conversation}")
            
        # Keep the assertion as a safety check
        assert cur_len == total_len, f"Length mismatch: cur_len={cur_len}, total_len={total_len}"
        
        target[cur_len:] = IGNORE_INDEX

    # Handle truncation for non-inference cases
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 255
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "cls_labels": torch.stack(cls_labels_list).view(-1),
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "cls_labels_list": cls_labels_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }

class FFPPCustomDataset(torch.utils.data.Dataset):
    img_size = 1024
    ignore_label = 255
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __init__(
        self,
        base_video_dir,  # Root directory containing real/full_synthetic/tampered
        tokenizer,
        vision_tower,
        split="test",
        precision: str = "fp32",
        image_size: int = 224,
    ):
        self.base_video_dir = base_video_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.split = split
        # Image processing
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        
        # Load images and verify
        self.images = []
        self.masks = []
        self.cls_labels = []

        subsets_to_use = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'original']

        # Load images and verify counts
        video_fps = glob.glob(os.path.join(base_video_dir, '**/videos/*.mp4'))
        video_fps = list(filter(lambda fp: any(sn in fp for sn in subsets_to_use), video_fps))

        random.shuffle(video_fps)
        if split == 'train':
            video_fps = video_fps[:int(len(video_fps) * 0.9)]
        elif split == 'test':
            video_fps = video_fps[int(len(video_fps) * 0.998):]
        else:
            raise ValueError(f'split \'{split}\' is not supported')

        for video_fp in tqdm(video_fps, desc='reading videos'):
            video_frames = list()
            
            video_cap = cv2.VideoCapture(video_fp)
            video_ret, video_frame = video_cap.read()
            while video_ret:
                video_frames.append(video_frame)
                video_ret, video_frame = video_cap.read()

            if 'original' in video_fp:
                label = 0
            else:
                label = 2

            if label == 0 or label == 1:
                mask_frames = list([np.zeros_like(f) for f in video_frames])
            elif label == 2:
                mask_frames = list()

                mask_fp = video_fp.replace('videos', 'masks')

                mask_cap = cv2.VideoCapture(mask_fp)
                mask_ret, mask_frame = mask_cap.read()
                while mask_ret:
                    mask_frames.append(mask_frame)
                    mask_ret, mask_frame = mask_cap.read()
            else:
                raise NotImplementedError()

            assert len(video_frames) == len(mask_frames)

            step = len(video_frames) // 10
            video_frames = video_frames[::step]
            mask_frames = mask_frames[::step]

            self.images.extend(video_frames)
            self.masks.extend(mask_frames)
            self.cls_labels.extend([label] * len(video_frames))

        # Print dataset statistics
        print(f"\nDataset Statistics for {split} split:")
        print(f"Real images: {len(list(filter(lambda l: l==0, self.cls_labels)))}")
        print(f"Full synthetic images: {len(list(filter(lambda l: l==1, self.cls_labels)))}")
        print(f"Tampered images: {len(list(filter(lambda l: l==2, self.cls_labels)))}")
        
    def __len__(self):
        return len(self.images)
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def _generate_response(self, cls_label):
        """Generate appropriate response based on image type and available description"""
        if cls_label == 0:
            return "[CLS] The image is real"
        elif cls_label == 1:
            return "[CLS] The image is full synthetic"
        else:  # cls_label == 2 (tampered)
            return "[CLS] The image is tampered [SEG]"
    
    def __getitem__(self, idx):
        image = self.images[idx]
        cls_labels = self.cls_labels[idx]
        # Load and process image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process for CLIP
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # Process image for model
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # Initialize mask
        # mask = torch.zeros((1, resize[0], resize[1]))

        mask_img = self.masks[idx]
        mask_img = self.transform.apply_image(mask_img)
        mask_img = mask_img / 255.0
        mask = torch.from_numpy(mask_img).max(dim=-1).values.unsqueeze(0)

        # Generate conversation
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], 
            f"{DEFAULT_IMAGE_TOKEN}\nCan you identify if this image is real, full synthetic, or tampered image? Please mask the tampered regions if it is tampered.")
        
        response = self._generate_response(cls_labels)
        conv.append_message(conv.roles[1], response)
        conversation = conv.get_prompt()
        has_text = None
        labels = torch.ones(mask.shape[1], mask.shape[2]) * self.ignore_label
        
        return None, image, image_clip, [conversation], mask, labels, cls_labels, resize, None, None, False, has_text
