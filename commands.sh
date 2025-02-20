uv run evaluation.py \
    --dataset taco \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --random_forcing \
    --max_tokens 4096 \
    --num_random_forces 2 \
    --split test \
    --concurrency 50 \
    --num_samples 8



uv run evaluation.py \
    --dataset taco \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --random_forcing \
    --max_tokens 4096 \
    --num_random_forces 2 \
    --split test \
    --concurrency 50 \
    --num_samples 8 \
    --wait_only

uv run evaluation.py \
    --dataset taco \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --max_tokens 4096 \
    --split test \
    --concurrency 50 \
    --num_samples 8 \
    --no_forcing