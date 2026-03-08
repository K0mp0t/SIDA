import argparse
import os
import shutil
import sys
import time
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
import tqdm
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

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
    )

    model = model.to(torch_dtype).to('cuda:0')

    test(test_loader, model, 0, writer, args) 


def test(test_loader, model_engine, epoch, writer, args, sample_ratio=None):
    model_engine.eval()
    correct = 0
    total = 0
    num_classes = 3
    confusion_matrix = torch.zeros(num_classes, num_classes, device='cuda')
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    # Calculate total number of batches and samples to use
    total_batches = len(test_loader)
    if sample_ratio is not None:
        num_batches = max(1, int(total_batches * sample_ratio))
        # Generate random indices for sampling
        sample_indices = set(random.sample(range(total_batches), num_batches))
        print(f"\ntest on {num_batches}/{total_batches} randomly sampled batches...")

    for batch_idx, input_dict in enumerate(tqdm.tqdm(test_loader)):
        # Skip batches not in our sample if sampling is enabled
        if sample_ratio is not None and batch_idx not in sample_indices:
            continue

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
        logits = output_dict["logits"]
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        cls_labels = input_dict["cls_labels"]
        correct += (preds == cls_labels).sum().item()
        total += cls_labels.size(0)

        for t, p in zip(cls_labels, preds):
            confusion_matrix[t.long(), p.long()] += 1
        if cls_labels[0] == 2:
            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] if hasattr(iou_class, 'len') and len(iou_class) > 1 else 0.0
    giou = acc_iou_meter.avg[1] if hasattr(acc_iou_meter.avg, 'len') and len(acc_iou_meter.avg) > 1 else 0.0
    accuracy = correct / total * 100.0
    confusion_matrix = confusion_matrix.cpu()
    class_names = ['Real', 'Full Synthetic', 'Tampered']
    per_class_metrics = {}
    for i in range(num_classes):
        tp = confusion_matrix[i, i]  
        fp = confusion_matrix[:, i].sum() - tp  
        fn = confusion_matrix[i, :].sum() - tp 
        tn = confusion_matrix.sum() - (tp + fp + fn)  

        total_class_samples = confusion_matrix[i, :].sum()

        class_accuracy = float(tp / total_class_samples) if total_class_samples > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_metrics[class_names[i]] = {
            'accuracy': class_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    pixel_correct = intersection_meter.sum[1] if hasattr(intersection_meter.sum, 'len') and len(intersection_meter.sum) > 1 else 0.0
    pixel_total = union_meter.sum[1] if hasattr(union_meter.sum, 'len') and len(union_meter.sum) > 1 else 0.0
    pixel_accuracy = pixel_correct / (pixel_total + 1e-10) * 100.0

    iou = ciou
    f1_score = 2 * (iou * accuracy / 100) / (iou + accuracy / 100 + 1e-10) if (iou + accuracy / 100) > 0 else 0.0

    avg_precision = np.mean([metrics['precision'] for metrics in per_class_metrics.values()])
    avg_recall = np.mean([metrics['recall'] for metrics in per_class_metrics.values()])

    auc_approx = avg_precision * avg_recall

    if args.local_rank == 0:
        writer.add_scalar("test/accuracy", accuracy, epoch)
        writer.add_scalar("test/giou", giou, epoch)
        writer.add_scalar("test/ciou", ciou, epoch)
        writer.add_scalar("test/pixel_accuracy", pixel_accuracy, epoch)
        writer.add_scalar("test/iou", iou, epoch)
        writer.add_scalar("test/f1_score", f1_score, epoch)
        writer.add_scalar("test/auc_approx", auc_approx, epoch)
        for class_name, metrics in per_class_metrics.items():
         for metric_name, value in metrics.items():
             writer.add_scalar(f"test/{class_name.lower().replace('/', '_')}_{metric_name}", value, epoch)

        test_type = "Full" if sample_ratio is None else f"Sampled ({sample_ratio*100}%)"
        print(f"\n{test_type} test Results:")
        print(f"giou: {giou:.4f}, ciou: {ciou:.4f}")
        print(f"Classification Accuracy: {accuracy:.4f}%")
        print(f"Pixel Accuracy: {pixel_accuracy:.4f}%")
        print(f"IoU: {iou:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Approximate AUC: {auc_approx:.4f}")
        print(f"Total correct classifications: {correct}")
        print(f"Total classification samples: {total}")
        print("\nPer-Class Metrics:")
        for class_name, metrics in per_class_metrics.items():
            print(f"\n{class_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")

        print("\nConfusion Matrix:")
        print("Predicted ")
        print("Actual ")
        print(f"{'':20}", end="")  
        for name in class_names:
            print(f"{name:>12}", end="") 
        print()  

        for i, class_name in enumerate(class_names):
            print(f"{class_name:20}", end="") 
            for j in range(num_classes):
                print(f"{confusion_matrix[i, j]:12.0f}", end="")
            print()  

    return accuracy, giou, ciou, per_class_metrics

if __name__ == "__main__":
    main(sys.argv[1:])
