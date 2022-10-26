python main.py \
--sagemaker False \
--num_node_types 8 \
--num_edge_types 4 \
--num_train 65000 \
--source_types 0,1,2,3,4,5,6,7 \
--sampling_size 320 \
--batch_s 32 \
--mini_batch_s 32 \
--eval_size 10 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 500 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 7 \
--out_embed_s 300 \
--hidden_channels 300 \
--num_hidden_conv_layers 1 \
--augmentation_method all \
--edge_ratio_percentile 0.9 \
--edge_mutate_prob 0.1 \
--subgraph_ratio 0.01 \
--insertion_iteration 1 \
--swap_node_pct 0.1 \
--swap_edge_pct 0.1 \
--svdd_loss_weight 0.8 \
--model_path ../model_save_tralog_gcn11_all \
--data_path ../ProcessedData_HetGCN


sudo update-alternatives --config gcc
sudo ln -sfT /usr/local/cuda-10.2/ /usr/local/cuda
pip install torch==1.9.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.1+cu102.html


# StreamSpot Data
python main.py \
--sagemaker False \
--num_node_types 8 \
--num_train 375 \
--source_types 0,1 \
--sampling_size 375 \
--batch_s 25 \
--mini_batch_s 25 \
--eval_size 25 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 500 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 26 \
--out_embed_s 32 \
--hidden_channels 32 \
--num_hidden_conv_layers 1 \
--model_path ../model_save_streamspot_gcn11 \
--data_path ../ProcessedData_streamspot


# DeepTraLog Baseline
python main.py \
--sagemaker False \
--num_train 65000 \
--sampling_size 320 \
--batch_s 160 \
--mini_batch_s 160 \
--eval_size 10 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 200 \
--model_version 12 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 7 \
--out_embed_s 300 \
--hidden_channels 300 \
--num_hidden_conv_layers 3 \
--model_path ../model_save_tralog_gcn12 \
--data_path ../ProcessedData_HetGCN

# StreamSpot Baseline
python main.py \
--sagemaker False \
--num_train 375 \
--sampling_size 375 \
--batch_s 25 \
--mini_batch_s 25 \
--eval_size 25 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 200 \
--model_version 9 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 26 \
--out_embed_s 32 \
--hidden_channels 32 \
--num_hidden_conv_layers 3 \
--model_path ../model_save_streamspot_gcn12 \
--data_path ../ProcessedData_streamspot