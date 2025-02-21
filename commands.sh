## TACO
# uv run evaluation.py \
#     --dataset taco \
#     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#     --random_forcing \
#     --max_tokens 4096 \
#     --num_random_forces 2 \
#     --split test \
#     --concurrency 50 \
#     --num_samples 8

uv run evaluation.py \
    --dataset taco \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --random_forcing \
    --max_tokens 4096 \
    --num_random_forces 8 \
    --split test \
    --concurrency 50 \
    --num_samples 1 \
    --approach_force

uv run evaluation.py \
    --dataset taco_medium \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --random_forcing \
    --max_tokens 16000 \
    --num_random_forces 2 \
    --split test \
    --concurrency 50 \
    --num_samples 8 \
    --approach_force

# uv run evaluation.py \
#     --dataset taco \
#     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#     --random_forcing \
#     --max_tokens 4096 \
#     --num_random_forces 2 \
#     --split test \
#     --concurrency 50 \
#     --num_samples 8 \
#     --wait_only

# uv run evaluation.py \
#     --dataset taco \
#     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#     --max_tokens 4096 \
#     --split test \
#     --concurrency 50 \
#     --num_samples 8 \
#     --no_forcing

## AIME
uv run evaluation.py \
    --dataset aime24 \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --random_forcing \
    --max_tokens 2048 \
    --num_random_forces 4 \
    --split test \
    --concurrency 50 \
    --num_samples 8

uv run evaluation.py \
    --dataset aime24 \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --random_forcing \
    --max_tokens 4096 \
    --num_random_forces 8 \
    --split test \
    --concurrency 50 \
    --num_samples 1 \
    --approach_force

## GPQA
# uv run evaluation.py \
#     --dataset gpqa_diamond \
#     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#     --random_forcing \
#     --max_tokens 4096 \
#     --num_random_forces 4 \
#     --split test \
#     --concurrency 50 \
#     --num_samples 8

# uv run evaluation.py \
#     --dataset gpqa_diamond \
#     --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#     --random_forcing \
#     --max_tokens 4096 \
#     --num_random_forces 4 \
#     --split test \
#     --concurrency 50 \
#     --num_samples 8 \
#     --approach_force