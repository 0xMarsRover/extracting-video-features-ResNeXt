#!/bin/sh
#sbatch --job-name=KQ_feat1 --gres=gpu:1 --mem=131072 --cpus-per-task=4 --output=/data/d14122793/zsar/log/log_extractFeat_resnxt101_6704.out ext_feat_6704.sh

python main.py --input ./ucf101_input_filename_6704.txt --video_root /data/d14122793/zsar/ucf101_main --output /data/d14122793/zsar/output_ucf101_6704.json --model ./resnext-101-kinetics.pth --batch_size 16 --model_depth 101 --model_name resnext --resnet_shortcut B --mode feature


# test on CPU (mac)
#python main.py --input ./ucf101_input_filename.txt --video_root /Volumes/Kellan/datasets/ucf101/ucf101_main_val --output ./output_ucf101.json --model ./resnet-34-kinetics-cpu.pth --mode feature --no_cuda
