#!/bin/bash
#SBATCH --job-name=verify-embeddings
#SBATCH --partition=gpu_p
#SBATCH --reservation=zeynep_users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_normal
#SBATCH --output=/lustre/groups/eml/projects/huang/multi_image/lmms-eval/slurm_logs/out/verify-embeddings.out
#SBATCH --error=/lustre/groups/eml/projects/huang/multi_image/lmms-eval/slurm_logs/error/verify-embeddings.err
#SBATCH --constraint="[a100_40gb|a100_80gb|h100_80gb]"
#SBATCH --exclude=gpusrv09,gpusrv10,gpusrv11,gpusrv12,gpusrv13,gpusrv14,gpusrv15,gpusrv16,gpusrv17,gpusrv18,gpusrv22,gpusrv23,gpusrv24,gpusrv25,gpusrv26,gpusrv27,gpusrv28,gpusrv29,gpusrv30,gpusrv31,gpusrv32,gpusrv33,gpusrv34,gpusrv35,gpusrv38,gpusrv39,gpusrv40,gpusrv41,gpusrv42,gpusrv43,gpusrv44,gpusrv45,gpusrv46,gpusrv47,gpusrv48,gpusrv49,gpusrv50,gpusrv51,gpusrv52,gpusrv53,gpusrv54,gpusrv55,gpusrv56,gpusrv57,gpusrv58,gpusrv59,gpusrv60,gpusrv61,gpusrv62,gpusrv63,gpusrv64,gpusrv65,gpusrv66,gpusrv67,gpusrv68,gpusrv69,gpusrv70,gpusrv71,gpusrv72,gpusrv73,gpusrv74,gpusrv75,gpusrv76,gpusrv77

export PYTHONHASHSEED=0

cd /lustre/groups/eml/projects/huang/multi_image/lmms-eval

echo "=== V1 & V3: Token classification + instruction extraction ==="
srun --ntasks=1 --exclusive --gres=gpu:1 uv run python scripts/extract_embeddings.py \
    --verify --compare_prompt \
    --datasets mantis muirbench mirb blink \
    --model_path Qwen/Qwen2.5-VL-3B-Instruct \
    --max_pixels 12845056
