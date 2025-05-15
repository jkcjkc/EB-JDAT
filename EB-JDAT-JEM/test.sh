CUDA_VISIBLE_DEVICES="1"  python eval_jem.py --eval test_clf --load_path ./ckp/your_model.pt   \
 --dataset=cifar_test  --gpu-id=1   \

CUDA_VISIBLE_DEVICES="1"  python eval_jem.py --eval PGD --load_path ./ckp/your_model.pt   \
 --dataset=cifar_test  --gpu-id=1   \
 
CUDA_VISIBLE_DEVICES="1"  python eval_jem.py --eval AA --load_path ./ckp/your_model.pt   \
 --dataset=cifar_test  --gpu-id=1   \

CUDA_VISIBLE_DEVICES="1"  python eval_jem.py --eval gen --load_path ./ckp/your_model.pt   \
 --dataset=cifar_test  --gpu-id=1 --ratio 0.9  \

CUDA_VISIBLE_DEVICES="1"  python eval_jem.py --eval fid --load_path ./ckp/your_model.pt   \
 --dataset=cifar_test  --gpu-id=1 --ratio 0.9  \