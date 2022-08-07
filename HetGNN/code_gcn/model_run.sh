python main.py \
--sagemaker False \
--num_train 10 \
--batch_s 10 \
--mini_batch_s 10 \
--num_eval 10 \
--unzip False \
--s3_stage False \
--save_model_freq 2 \
--lr 0.005 \
--train_iter_n 200 \
--model_version 3 \
--test_set False \
--feature_size 10 \
--out_embed_s 16 \
--model_path ../model_save_clean_synth_3 \
--data_path ../ProcessedData_synth \
--split_data False


sudo update-alternatives --config gcc
sudo ln -sfT /usr/local/cuda-10.2/ /usr/local/cuda

pip install torch==1.9.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.1+cu102.html