for seed in 2022 2021 2020; do for gr in 0.5 0.3 0.1; do for gv in 0.35 0.4 0.45 0.5; do python3 train.py --arch resnet9 --data-dir /data/yihua/dataset/celeba/celeba.hdf5 --target-attrs Blond_Hair --gv ${gv} --gr ${gr} --seed ${seed} | tee log/resnet9_BlondHair_gv${gv}_gr${gr}_seed${seed}.log; done; done; done