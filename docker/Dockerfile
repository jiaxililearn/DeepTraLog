FROM 763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.11.0-gpu-py38

ENV PATH="/opt/ml/code:${PATH}"

RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

RUN pip install cytoolz
# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
# COPY HetGNN/code_gcn/ /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
# ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
# ENV SAGEMAKER_PROGRAM main.py