sudo update-alternatives --config gcc
sudo ln -sfT /usr/local/cuda-10.2/ /usr/local/cuda
pip install torch==1.9.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.1+cu102.html

# Case Study
python main.py \
--sagemaker False \
--num_node_types 3 \
--source_types 0,1,2 \
--num_train 7200000 \
--sampling_size 3200 \
--batch_s 16 \
--mini_batch_s 16 \
--eval_size 100000 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.01 \
--train_iter_n 100 \
--random_seed 36 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 31 \
--out_embed_s 64 \
--hidden_channels 64 \
--num_hidden_conv_layers 3 \
--embed_activation sigmoid \
--tolerance 5 \
--augmentation_method all \
--main_loss svdd \
--weighted_loss ignore \
--loss_weight 0 \
--eval_method svdd \
--model_path ../model_save_case_study \
--data_path ../tpg_case_study_data \
--job_prefix test_case_study