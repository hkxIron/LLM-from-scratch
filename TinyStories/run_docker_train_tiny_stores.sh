#!/usr/bin/env bash
. ./shell_utils.sh

pushd .
cd $HOME
sh scp_proj.sh
popd


echo `date`
start_time=$(date +%s)
time_str="$(date +%Y%m%d-%H-%M-%S)"

test_min_gpu_num 1
# 同步数据
#ip="192.168.0.1"
#rsync -av -e ssh --exclude='*.git' --exclude='.*' --exclude='__pycache__/' --exclude='wandb/' hkx@$ip:/media/hkx/win/hkx/ubuntu/work/open/nanoGPT $HOME/work/open/


root_path="$HOME/work"
project_path="${root_path}/open/project/LLM-from-scratch/TinyStories/"
data_dir="${root_path}/open/hf_data_and_model/datas/TinyStoriesV2/"
tokenizer_path="${root_path}/open/hf_data_and_model/models/NousResearch/Llama-2-7b-hf/"
output_dir="${root_path}/open/model_output/TinyStoriesV2/" # 包括模型，数据

#image="YOUR DOCKER IMAGE"

img1="icr.cl"
img2="d.m"
img3="ice.cn/w"
image="m${img1}ou${img2}ioff${img3}sw/large-lm:1.0.15-2"

wandb_key="bdfc8b674cd322f967699975e89d431e82fcd317" # hkx wandb
device_list="0"

nohup docker run -i --rm --gpus '"device='${device_list}'"' --name train_llm_tiny_stories --network=host --shm-size=16gb \
    -v /etc/localtime:/etc/localtime:ro \
    -v ${project_path}:/docker_workspace \
    -v ${data_dir}:/docker_data_dir \
    -v ${output_dir}:/docker_output_dir \
    -v ${tokenizer_path}:/docker_tokenizer_path \
    -w /docker_workspace \
    ${image} \
    bash -c "\
export PYTHONPATH=/docker_workspace && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 && \
export WANDB_DISABLED=false && \
export WANDB_PROJECT=llm_tiny_stories && \
export WANDB_API_KEY=${wandb_key} && \
wandb login ${wandb_key} && \
python train_tiny_stores.py \
--enable_wandb True \
--is_debug False \
--tokenizer_path /docker_tokenizer_path \
--dataset_path /docker_data_dir \
--output_path /docker_output_dir " 2>&1 |tee logs/log_${time_str}.txt

echo "`date` 训练结束"

