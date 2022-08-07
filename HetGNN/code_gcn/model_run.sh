python main.py \
--sagemaker False \
--num_train 200 \
--batch_s 100 \
--mini_batch_s 100 \
--num_eval 100 \
--unzip False \
--s3_stage False \
--split_data False \
--test_set False \
--save_model_freq 2 \
--lr 0.005 \
--train_iter_n 200 \
--model_version 3 \
--feature_size 7 \
--out_embed_s 32 \
--model_path ../model_save_HetGCN_act \
--data_path ../ProcessedData_HetGCN


sudo update-alternatives --config gcc
sudo ln -sfT /usr/local/cuda-10.2/ /usr/local/cuda

pip install torch==1.9.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.1+cu102.html