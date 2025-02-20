import argparse
import asyncio
import copy
import json
import os
from openai import AsyncOpenAI
from inference import InferenceEngine
from forces import Force
from datasets import load_dataset
from prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPTS
from tqdm import tqdm

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["taco"], help="Name of the Hugging Face dataset to use for evaluation")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Model name")
    parser.add_argument("--num_random_forces", type=int, default=3, help="Number of random forces to apply")
    parser.add_argument("--random_forcing", action="store_true", help="Enable random forcing")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top p")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrency limit for processing samples")
    parser.add_argument("--wait_only", action="store_true", help="Only apply wait force")
    parser.add_argument("--no_forcing", action="store_true", help="Do not apply any forcing")
    args = parser.parse_args()

    if args.dataset == "taco":
        dataset = load_dataset("BAAI/TACO", trust_remote_code=True)[args.split].filter(lambda x: x["difficulty"] == "HARD")
        from prompts import generate_prompt
        system_prompt = DEFAULT_SYSTEM_PROMPT
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")
    
    if args.wait_only:
        forces_pool = [
            Force("wait_force", "Wait", 1),
        ]
    else:
        forces_pool = [
        Force("wait_force", "Wait", 1),
        Force("chinese_force", "Let's think about this in Chinese.", 1),
        Force("check_force", "Hmm, let's check whether this is correct.", 1),
        Force("funamentally_different_force", "Let me think about this from a fundamentally different approach.", 1),
        Force("five_approaches_force", "Let me think of 5 different approaches and continue with the one that I think will work the best.", 1),
    ]
        

    base_url = f"http://localhost:{args.port}/v1"
    instruct_client = AsyncOpenAI(api_key="dummy-key", base_url=base_url)
    
    output_dir = f"results/{args.dataset}/{args.model.replace('/', '_')}_{args.split}_nrofrandomforces_{args.num_random_forces}_randomforcing_{args.random_forcing}_waitonly_{args.wait_only}_noforcing_{args.no_forcing}_maxtokens_{args.max_tokens}_temp_{args.temperature}_topp_{args.top_p}"
    os.makedirs(output_dir, exist_ok=True)
    
    sem = asyncio.Semaphore(args.concurrency)

    async def process_sample(idx, sample, sample_idx, conversation):
        async with sem:
            conv_copy = copy.deepcopy(conversation)
            engine = InferenceEngine(
                client=instruct_client,
                model=args.model,
                base_messages=conv_copy,
                forces=forces_pool,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                random_forcing=args.random_forcing,
                num_random_forces=args.num_random_forces,
            )
            output, applied_forces = await engine.run(no_forcing=args.no_forcing)
            
            result = {
                "dataset": args.dataset,
                "question_id": idx,
                "sample_index": sample_idx, 
                "conversation": conversation,
                "output": output, 
                "applied_forces": [f.to_dict() for f in applied_forces],
                "sample": sample,
                "config": args.__dict__
            }
    
            with open(os.path.join(output_dir, f"question_{idx}_sample_{sample_idx}.json"), "w") as f:
                json.dump(result, f, indent=4)

    tasks = []
    for idx, sample in enumerate(dataset):
        if args.dataset == "taco":
            test_case = json.loads(sample["input_output"])
            starter_code = sample["starter_code"]
            prompt = generate_prompt(test_case, sample["question"], starter_code)
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "<think>"}
            ]
        
        for sample_idx in range(args.num_samples):
            output_file = os.path.join(output_dir, f"question_{idx}_sample_{sample_idx}.json")
            if os.path.exists(output_file):
                print(f"Skipping question {idx} sample {sample_idx} because it already exists")
                continue
            tasks.append(asyncio.create_task(process_sample(idx, sample, sample_idx, conversation)))
    
    if tasks:
        pbar = tqdm(total=len(tasks), desc="Processing samples")
        for task in asyncio.as_completed(tasks):
            await task
            pbar.update(1)
        pbar.close()

if __name__ == "__main__":
    asyncio.run(main()) 