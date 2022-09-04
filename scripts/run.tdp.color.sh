DATASET="tdp-color"
python main.py \
--dataset $DATASET \
--n_epochs 50 \
--n_epochs_decay 100 \
--input_nc 1 \
--output_nc 3 \
--n_layers_D 2 \
--src_dir "./data/data/Blueprint" \
--dst_dir "./data/data/Mash" \
--csv_fpath "./data/data/Metadata/data.csv" \
--lr_decay_iters 25 \
--output_dir ./experiments/$DATASET/outputs \
--log_dir ./experiments/$DATASET/logs \
--checkpoint_dir ./experiments/$DATASET/checkpoints
