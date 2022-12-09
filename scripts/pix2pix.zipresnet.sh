DATASET="fdm-color"
python main.py \
--netG zip_resnet \
--dataset $DATASET \
--n_epochs 50 \
--n_epochs_decay 100 \
--input_nc 1 \
--output_nc 3 \
--batch_size 4 \
--n_layers_G 9 \
--n_layers_D 3 \
--ndf 32 \
--dataset "fdm-color" \
--src_dir "./data/g-fdm/Blueprint" \
--dst_dir "./data/g-fdm/Mash" \
--csv_fpath "./data/g-fdm/Metadata/data.G-FDM.latest.csv" \
--lr_decay_iters 25 \
--output_dir ./experiments/$DATASET/outputs \
--log_dir ./experiments/$DATASET/logs \
--checkpoint_dir ./experiments/$DATASET/checkpoints \
--dropout-D 0.75 \
--dropout-G 0.5 \
--expr_name ZipBlockResNet \
--lambda_VGG 10 \
--lambda_REG 1 \
--max_grad_norm 0. \
--input_size 512 512 \
--feature_layers 0 1 2 3 \
--device cuda:1 \
--weight_decay_G 1e-4 \
--weight_decay_D 1e-2


# vgg 300
# reg 100
# adv 10
