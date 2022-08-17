python main.py \
--sagemaker False \
--num_node_types 14 \
--num_train 360 \
--batch_s 360 \
--mini_batch_s 360 \
--num_eval 125 \
--hidden_channels 26 \
--seed 10 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.005 \
--train_iter_n 200 \
--model_version 4 \
--dataset_id 0 \
--feature_size 26 \
--out_embed_s 26 \
--model_path ../model_save_HetGCN_streamspot_hetgnn_paper \
--data_path ../custom_data_simple


sudo update-alternatives --config gcc
sudo ln -sfT /usr/local/cuda-10.2/ /usr/local/cuda

pip install torch==1.9.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.1+cu102.html