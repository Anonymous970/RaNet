 
export CUDA_VISIBLE_DEVICES=0,1,2,3

python moment_localization/train.py --cfg experiments/activitynet/dynamic+cc.yaml --verbose
# python moment_localization/train.py --cfg experiments/activitynet/dot+4conv.yaml --verbose


