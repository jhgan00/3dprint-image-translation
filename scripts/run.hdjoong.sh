DATASET="hdjoong"
python main.py \
--netG resnet \
--dataset $DATASET \
--batch_size 4 \
--input_size 160 672 \
--n_epochs 10 \
--n_epochs_decay 20 \
--input_nc 1 \
--output_nc 1 \
--n_layers_D 2 \
--ndf 32 \
--src_dir "./data/$DATASET/Blueprint" \
--dst_dir "./data/$DATASET/Target" \
--csv_fpath "./data/$DATASET/data.csv" \
--lr_decay_iters 10 \
--output_dir ./experiments/$DATASET/outputs \
--log_dir ./experiments/$DATASET/logs \
--checkpoint_dir ./experiments/$DATASET/checkpoints \
--max_grad_norm 1
