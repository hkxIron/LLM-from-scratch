echo "HOME:$HOME"
# 同步数据
remote_project_path="/media/hkx/win/hkx/ubuntu/work/open/LLM-from-scratch/"
local_project_path="$HOME/work/open/project/LLM-from-scratch"
user_ip="hkx@10.239.6.137"
rsync -av -e ssh --exclude='*.git' --exclude='.*' --exclude='*checkpoints*' --exclude='__pycache__/' --exclude='wandb/' ${user_ip}:${remote_project_path} $local_project_path
