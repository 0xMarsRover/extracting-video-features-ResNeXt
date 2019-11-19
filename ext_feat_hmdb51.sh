#!/bin/sh
#sbatch --job-name=KQ_feat3 --gres=gpu:1 --mem=131072 --cpus-per-task=4 --output=/data/d14122793/zsar/log/log_extractFeat_resnxt101_hmdb51.out ext_feat_hmdb51.sh

python main.py --input ./hmdb51_input_filename.txt --video_root /data/d14122793/zsar/hmdb51_main --output /data/d14122793/zsar/output_hmdb51.json --model ./resnext-101-kinetics.pth --batch_size 16 --model_depth 101 --model_name resnext --resnet_shortcut B --mode feature


# test on CPU (mac)
#python main.py --input ./ucf101_input_filename.txt --video_root /Volumes/Kellan/datasets/ucf101/ucf101_main_val --output ./output_ucf101.json --model ./resnet-34-kinetics-cpu.pth --mode feature --no_cuda
