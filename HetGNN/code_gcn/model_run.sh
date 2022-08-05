python main.py \
--sagemaker False \
--num_train 200 \
--batch_s 200 \
--mini_batch_s 100 \
--num_eval 100 \
--unzip False \
--s3_stage False \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 250 \
--model_version 2 \
--model_path ../model_save_clean_gcnconv


sudo update-alternatives --config gcc
sudo ln -sfT /usr/local/cuda-10.2/ /usr/local/cuda

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu102.html