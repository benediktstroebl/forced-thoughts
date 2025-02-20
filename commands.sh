uv run evaluation.py \
    --dataset taco \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --random_forcing \
    --max_tokens 8192 \
    --num_random_forces 3 \
    --split test \
    --concurrency 50 \
    --num_samples 10

uv run evaluation.py \
    --dataset taco \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --random_forcing \
    --max_tokens 8192 \
    --num_random_forces 3 \
    --split test \
    --concurrency 50 \
    --num_samples 10 \
    --wait_only

uv run evaluation.py \
    --dataset taco \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --max_tokens 8192 \
    --split test \
    --concurrency 50 \
    --num_samples 10 \
    --no_forcing