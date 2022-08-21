python main.py \
--sagemaker False \
--num_node_types 8 \
--num_train 3200 \
--batch_s 32 \
--mini_batch_s 32 \
--num_eval 1000 \
--seed 10 \
--unzip False \
--s3_stage False \
--split_data True \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 200 \
--model_version 4 \
--model_sub_version 0 \
--source_types 0,1,2,3,4,5,6,7 \
--dataset_id 0 \
--input_type batch \
--feature_size 7 \
--out_embed_s 300 \
--hidden_channels 300 \
--model_path ../model_save_HetGCN_gcn4 \
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
--batch_s 25 \
--mini_batch_s 25 \
--num_eval 225 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.001 \
--train_iter_n 100 \
--model_version 7 \
--model_sub_version 0 \
--source_types 0,1 \
--dataset_id 0 \
--input_type batch \
--feature_size 26 \
--out_embed_s 26 \
--hidden_channels 26 \
--num_hidden_conv_layers 2 \
--model_path ../model_save_HetGCN_gcn7 \
--data_path ../ProcessedData_streamspot