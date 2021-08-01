export CUDA_VISIBLE_DEVICES=0,1,2,3

python moment_localization/train.py --cfg experiments/charades/vgg-dynamic+cc.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/vgg-dot+8conv.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/vgg-dot+cc.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/vgg-dynamic+8conv.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/i3d-raw-dynamic+cc.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/i3d-finetune-dynamic+cc.yaml --verbose
# python moment_localization/train.py --cfg experiments/charades/c3d-dynamic+cc.yaml --verbose