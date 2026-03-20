import argparse
import os
import shutil
import sys
import time
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from tqdm.auto import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from utils.SID_Set import collate_fn, CustomDataset
from utils.ffpp_dataset import ffpp_collate_fn, FFPPCustomDataset
from utils.batch_sampler import BatchSampler
import torch.distributed as dist
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
import random
import torch.nn.functional as F
import warnings
import cv2


warnings.filterwarnings("ignore")

def parse_args(args):
    parser = argparse.ArgumentParser(description="SIDA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--test_dataset", default="test", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="sida", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)

    parser.add_argument("--num_classes", type=int, default=3,
                       help="Number of classes for classification in stage 1")
    parser.add_argument("--use_stage1_cls", action="store_true", default=True,
                   help="Whether to use Stage 1 CLS token in Stage 2")
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    parser.add_argument("--cls_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=1.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--no_test", action="store_true", default=False)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--weight", default=None, type=str, required=False)

    return parser.parse_args(args)
def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    num_added_token = tokenizer.add_tokens("[CLS]")
    num_added_token = tokenizer.add_tokens("[SEG]")
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "cls_loss_weight": args.cls_loss_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "cls_token_idx": args.cls_token_idx,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = SIDAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print("\nChecking specific components:")
    for component in [ "cls_head", "sida_fc1", "attention_layer", "text_hidden_fcs"]:
        matching_params = [n for n, _ in model.named_parameters() if component in n]
        if matching_params:
            print(f"Found {component} in parameters: {matching_params}")
        else:
            print(f"Component not found: {component}")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.test_only:
        model.get_model().initialize_sida_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False

    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "cls_head",
                                "sida_fc1",
                                "attention_layer",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
                model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = False

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["embed_tokens", "mask_decoder", "text_hidden_fcs","cls_head", "sida_fc1","attention_layer"]
            ]
        ):
            p.requires_grad = True

    # print("Checking trainable parameters:")
    total_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            # print(f"Trainable: {n} with {p.numel()} parameters")
            total_params += p.numel()
    print(f"Total trainable parameters: {total_params}")

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    test_dataset = CustomDataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,  
        split="test", 
        precision=args.precision, 
        image_size=args.image_size, 
    )
    print('Number of samples to test on:', len(test_dataset))

    if args.weight is not None:
        state_dict = torch.load(args.weight, map_location="cpu")
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        model.load_state_dict(state_dict, strict=True)
    model = model.to(torch_dtype).to('cuda:0')

    collate_fn_kwargs = {
        "tokenizer": tokenizer,
        "conv_type": args.conv_type,
        "use_mm_start_end": args.use_mm_start_end,
        "local_rank": args.local_rank
        }

    visualize_masks(test_dataset, model, 0, writer, args, collate_fn_kwargs) 


def visualize_masks(test_dataset, model_engine, epoch, writer, args, collate_fn_kwargs, sample_ratio=None, num_masks_to_visualize=24):
    output_dir = './vis_out/SIDset_CelebDF'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    model_engine.eval()

    data_idxs = [i for i, e in enumerate(test_dataset.cls_labels) if e == 2]

    for item_idx in tqdm(random.choices(data_idxs, k = num_masks_to_visualize)):
        item = test_dataset[item_idx]
        input_dict = collate_fn([item], **collate_fn_kwargs)
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        input_dict['inference'] = True
        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        # Get predictions
        cls_labels = input_dict["cls_labels"]
        
        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int() 

        pred_mask = ((pred_masks[0][0] > 0).to(torch.uint8) * 255).cpu().numpy()
        gt_mask = (masks_list[0].to(torch.uint8) * 255).cpu().numpy()

        pred_mask[gt_mask[gt_mask == 255]] = 255

        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

        image = cv2.imread(input_dict["image_paths"][0])
        coef = min(pred_mask.shape[0] / image.shape[0], pred_mask.shape[1] / image.shape[1])
        image = cv2.resize(image, None, fx=coef, fy=coef, interpolation=cv2.INTER_CUBIC)
        image = cv2.copyMakeBorder(image, 0, pred_mask.shape[0] - image.shape[0], 0, pred_mask.shape[1] - image.shape[1], cv2.BORDER_CONSTANT, value=255)

        vis = np.hstack([image, pred_mask, gt_mask])
        cv2.imwrite(os.path.join(output_dir, os.path.basename(input_dict["image_paths"][0])), vis)


if __name__ == "__main__":
    main(sys.argv[1:])
