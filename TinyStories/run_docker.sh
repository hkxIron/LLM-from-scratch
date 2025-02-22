#!/usr/bin/env bash
. ./shell_utils.sh

echo `date`
start_time=$(date +%s)
time_str="$(date +%Y%m%d-%H-%M-%S)"

root_path="$HOME/work"
project_path="${root_path}/open/project/LLM-from-scratch/TinyStories/"
data_dir="${root_path}/open/hf_data_and_model/datas/TinyStoriesV2/"
tokenizer_path="${root_path}/open/hf_data_and_model/models/NousResearch/Llama-2-7b-hf/"
output_dir="${root_path}/open/model_output/TinyStoriesV2/" # 包括模型，数据

# docker image
img1="icr"
img2=".m"
img3="ice.cn"
img4='wsw/large-lm:1.0.15-2'
image="m${img1}.cloud${img2}ioff${img3}/${img4}"
echo $image

wandb_key="bdfc8b674cd322f967699975e89d431e82fcd317" # hkx wandb
max_gpu_num=1 # 最大限制多少个gpu
test_max_gpu_num $max_gpu_num
#device_list="3"

set -x
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

set +x
echo "`date` 训练结束"

