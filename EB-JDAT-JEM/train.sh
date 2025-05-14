CUDA_VISIBLE_DEVICES="2"  python train_wrn_ebm.py --lr .00001 \
 --dataset cifar10 --optimizer adam \
 --p_x_weight 0.45 --p_y_given_x_weight 0.45 --p_x_xadv_weight 0.05 --p_y_given_xadv_weight 0.45 --p_x_y_weight 0.0 \
 --sigma .03 --width 10 --depth 28 --save_dir ./CKP --plot_uncond --warmup_iters 1000