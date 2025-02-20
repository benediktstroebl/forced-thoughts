import argparse
import asyncio
import json
import os
from openai import AsyncOpenAI
from inference import InferenceEngine
from forces import Force
from datasets import load_dataset
from prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPTS

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=["taco"], help="Name of the Hugging Face dataset to use for evaluation")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Model name")
    parser.add_argument("--num_random_forces", type=int, default=1, help="Number of random forces to apply")
    parser.add_argument("--random_forcing", action="store_true", help="Enable random forcing")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top p")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.dataset_name == "taco":
        dataset = load_dataset("BAAI/TACO", trust_remote_code=True)[args.split].filter(lambda x: x["difficulty"] == "HARD")
        from prompts import generate_prompt
        system_prompt = DEFAULT_SYSTEM_PROMPT
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

    forces_pool = [
        Force("wait_force", "Wait", 1),
        Force("chinese_force", "Let's think about this in Chinese.", 1),
        Force("check_force", "Hmm, let's check whether this is correct.", 1),
        Force("funamentally_different_force", "Let me think about this from a fundamentally different approach.", 1),
        Force("five_approaches_force", "Let me think of 5 different approaches and continue with the one that will work the best.", 1),
    ]

    base_url = f"http://localhost:{args.port}/v1"
    instruct_client = AsyncOpenAI(api_key="dummy-key", base_url=base_url)
    
    output_dir = f"results/{args.dataset_name}/{args.model}_{args.split}_nrofrandomforces_{args.num_random_forces}_randomforcing_{args.random_forcing}_maxtokens_{args.max_tokens}_temp_{args.temperature}_topp_{args.top_p}"
    os.makedirs(output_dir, exist_ok=True)
    for idx, sample in enumerate(dataset):
        if args.dataset_name == "taco":
            test_case = json.loads(sample["input_output"])
            starter_code = sample["starter_code"]
            prompt = generate_prompt(test_case, sample["question"], starter_code)
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
        # Force begin of thought in the first message
        conversation[1]['content'] = "<|begin_of_thought|>" + conversation[1]['content']
        
        for sample_idx in range(args.num_samples):
            engine = InferenceEngine(
                client=instruct_client,
                model=args.model,
                base_messages=conversation,
                forces=forces_pool,
                max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            random_forcing=args.random_forcing,
            num_random_forces=args.num_random_forces,
            )
            output, applied_forces = await engine.run()
            
            result = {
                "sample_index": sample_idx, 
                "prompt": sample["prompt"], 
                "output": output, 
                "applied_forces": applied_forces
                }
    
        with open(os.path.join(output_dir, f"question_{idx}_sample_{idx}.json"), "w") as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main()) 