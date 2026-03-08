deepspeed --include localhost:0 --master_port=24999 test.py \
  --version="./weights/SIDA-7B" \
  --dataset_dir='/home/peter/faigc/data/ff++/FaceForensics++_C23' \
  --vision_pretrained="./weights/sam_vit_h_4b8939.pth"