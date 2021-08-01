
export CUDA_VISIBLE_DEVICES=4,5,6,7

# test
python moment_localization/test.py --cfg experiments/activitynet/dynamic+cc.yaml --verbose --split test
# python moment_localization/test.py --cfg experiments/activitynet/dot+4conv.yaml --verbose --split test



