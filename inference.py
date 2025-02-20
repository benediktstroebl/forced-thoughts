import asyncio
import openai
from forces import remove_final_solution, Force

class InferenceEngine:
    def __init__(self, client, model, base_messages, forces, max_tokens, temperature=0.7, top_p=0.7, random_forcing=False, num_random_forces=0):
        self.client = client
        self.model = model
        self.messages = base_messages  # list of dicts with "role" and "content"
        self.forces = forces           # list of Force objects
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.random_forcing = random_forcing
        self.num_random_forces = num_random_forces
        self.applied_forces = []

    async def generate(self, extra_body=None):
        extra_body = extra_body or {}
        chat_completion = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body=extra_body,
        )
        return chat_completion.choices[0].message.content

    async def run(self):
        current_trace = await self.generate(extra_body={"continue_final_message": True, "add_generation_prompt": False})
        if self.random_forcing:
            import random
            for _ in range(self.num_random_forces):
                current_trace = remove_final_solution(current_trace)
                force = random.choice(self.forces)
                self.applied_forces.append(force)
                forced_trace = force.append_force(current_trace)
                self.messages[-1]['content'] = forced_trace
                current_trace = await self.generate(extra_body={"continue_final_message": True, "add_generation_prompt": False})
        else:
            for force in self.forces:
                while force.repetitions > 0:
                    current_trace = remove_final_solution(current_trace)
                    forced_trace = force.append_force(current_trace)
                    self.messages[-1]['content'] = forced_trace
                    current_trace = await self.generate(extra_body={"continue_final_message": True, "add_generation_prompt": False})
                    force.repetitions -= 1
                    self.applied_forces.append(force)
        current_trace = remove_final_solution(current_trace)
        final_trace = await self.generate(extra_body={"continue_final_message": True, "add_generation_prompt": False})
        return final_trace, self.applied_forces

# Example usage:
# async def main():
#     client = openai  # assume openai API is configured to point to the local vllm server
#     base_messages = [{"role": "assistant", "content": "Initial reasoning trace..."}]
#     forces = [Force("Please continue reasoning without finalizing the answer.", 2)]
#     engine = InferenceEngine(client, "your-model-name", base_messages, forces, max_tokens=100)
#     final_output = await engine.run()
#     print(final_output)
#
# asyncio.run(main()) 