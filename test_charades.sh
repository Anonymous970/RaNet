
export CUDA_VISIBLE_DEVICES=4,5

# test
python moment_localization/test.py --cfg experiments/charades/vgg-dynamic+cc.yaml --verbose --split test
# python moment_localization/test.py --cfg experiments/charades/vgg-dot+8conv.yaml --verbose --split test
# python moment_localization/test.py --cfg experiments/charades/i3d-raw-dynamic+cc.yaml --verbose --split test
# python moment_localization/test.py --cfg experiments/charades/i3d-finetune-dynamic+cc.yaml --verbose --split test

