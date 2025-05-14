export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES="1" python train_jempp.py --dataset=cifar10 \
 --lr=0.01 --optimizer=sgd \
 --p_x_weight=1 --p_y_given_x_weight=1 --p_x_xadv_weight=0.1 --p_y_given_xadv_weight=1 \
 --sigma=.03 --width=10 --depth=28 \
 --plot_uncond --warmup_iters=1000 \
 --log_arg=JEMPP-n_steps-in_steps-pyld_lr \
 --model=yopo \
 --norm batch \
 --print_every=100 \
 --n_epochs=150 --decay_epochs 40 80 120 \
 --n_steps=10 \
 --in_steps=5 \
 --pyld_lr=0.2 \
 --gpu-id=1
