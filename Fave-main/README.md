  python src/auto_main.py --stage_mode pretrain --dataset ml-1m --device cuda --ckpt_path "./best for ml-1m/pretrain_best.pt"

  python src/auto_main.py --stage_mode finetune --dataset ml-1m --device cuda --ckpt_path "./best for ml-1m/pretrain_best.pt"