python train.py \
        llama \
        occ_Net \
        /home/wangzj/occ-svdd/logs/test \
        /home/wangzj/occ-svdd/data/data.pt \
        /home/wangzj/occ-svdd/data/test \
        --objective soft-boundary \
        --lr 0.0001 \
        --n_epochs 200 \
        --lr_milestone 50 \
        --batch_size 1024 \
        --weight_decay 0.5e-6 \
        --pretrain True \
        --train True \
        --ae_lr 0.0001 \
        --ae_n_epochs 400 \
        --ae_lr_milestone 100 \
        --ae_batch_size 1024 \
        --ae_weight_decay 0.5e-6 \