# command

pip config --unset global.index-url
pip config --unset install.trusted-host

pip install ftfy
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.4.9
pip install transformers==4.8.1

# python -m torch.distributed.launch --nproc_per_node=8 --use_env Retrieval.py --config ./configs/Retrieval_coco.yaml --output_dir output/small_v2 --checkpoint /mnt/workspace/Project/Project/ALBEF/checkpoints/ALBEF_4M.pth
python -m torch.distributed.launch --nproc_per_node=8 --use_env Pretrain.py --config ./configs/Pretrain.yaml --output_dir output/Pretrain_all_data_on_dlc_sim_base_without_queue_v14 --checkpoint /mnt/workspace/Project/Project/ALBEF/output/Pretrain_all_data_on_dlc_sim_base_without_queue_v12/checkpoint_29.pth --resume True