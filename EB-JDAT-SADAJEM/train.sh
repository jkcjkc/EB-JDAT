export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES="1" python train_sadajem.py --dataset cifar10 \
 --lr 0.01 --optimizer sgd \
 --px 1 --pyx 1 --p_x_xadv_weight 0.1 --p_y_given_xadv_weight 1 \
 --sigma .03 --width 10 --depth 28 \
 --plot_uncond --warmup_iters 1000 \
 --model wrn \
 --norm batch \
 --print_every 100 \
 --n_epochs 200 --decay_epochs 50 100 150 \
 --n_steps 10   \
 --in_steps 5  \
 --sgld_lr 1.0  \
 --sgld_std 0.0 \
 --gpu-id 1
