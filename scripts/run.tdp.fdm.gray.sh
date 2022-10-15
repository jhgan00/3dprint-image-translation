DATASET="fdm-gray"
python main.py \
--dataset $DATASET \
--n_epochs 100 \
--n_epochs_decay 200 \
--input_nc 1 \
--output_nc 1 \
--batch_size 4 \
--n_layers_G 9 \
--n_layers_D 3 \
--ndf 32 \
--dataset "fdm-gray" \
--src_dir "./data/tdp-fdm/Blueprint" \
--dst_dir "./data/tdp-fdm/Mash" \
--csv_fpath "./data/tdp-fdm/Metadata/data.fdm.latest.scaled.csv" \
--lr_decay_iters 50 \
--output_dir ./experiments/$DATASET/outputs \
--log_dir ./experiments/$DATASET/logs \
--checkpoint_dir ./experiments/$DATASET/checkpoints \
--dropout 0.5 \
--expr_name regression-strain-reg-10-gc-0-fl-3 \
--lambda_VGG 50 \
--lambda_REG 10 \
--max_grad_norm 0. \
--input_size 512 512 \
--feature_layers 3 \
--device cuda:3
