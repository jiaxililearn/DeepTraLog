python main.py \
--sagemaker False \
--num_node_types 8 \
--num_train 375 \
--batch_s 125 \
--mini_batch_s 125 \
--num_eval 225 \
--hidden_channels 26 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.005 \
--train_iter_n 200 \
--model_version 3 \
--feature_size 26 \
--out_embed_s 26 \
--model_path ../model_save_HetGCN_streamspot_weight \
--data_path ../ProcessedData_streamspot


sudo update-alternatives --config gcc
sudo ln -sfT /usr/local/cuda-10.2/ /usr/local/cuda

pip install torch==1.9.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.1+cu102.html