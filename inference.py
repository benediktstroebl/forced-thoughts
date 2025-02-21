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
        self.final_force = Force("final_force", "Ok, now it's time to conclude and produce the final solution.</think >\n<answer>", 1)

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

    async def run(self, no_forcing=False):
        current_trace = await self.generate(extra_body={"continue_final_message": True, "add_generation_prompt": False})
        current_trace = self.messages[-1]['content'] + current_trace.replace("</think>", "</think >")
        
        if no_forcing:
            return current_trace, []
        
        if self.random_forcing:
            import random
            remaining_forces = [f for f in self.forces if f.max_repetitions > 0]
            for _ in range(self.num_random_forces):
                if not remaining_forces:
                    break
                    
                current_trace = remove_final_solution(current_trace)
                # print(f"Current trace: {current_trace}")
                force = random.choice(remaining_forces)
                self.applied_forces.append(force)
                # print(f"Applied force: {force.name}")
                forced_trace = force.append_force(current_trace)
                # print(f"Forced trace: {forced_trace}")
                self.messages[-1]['content'] = forced_trace
                # print(f"Messages: {self.messages}")
                current_trace = await self.generate(extra_body={"continue_final_message": True, "add_generation_prompt": False})
                current_trace = self.messages[-1]['content'] + current_trace.replace("</think>", "</think >")
                # print(f"New current trace: {current_trace}")
                
                force.max_repetitions -= 1
                if force.max_repetitions <= 0:
                    remaining_forces.remove(force)
        else:
            for force in self.forces:
                while force.max_repetitions > 0:
                    current_trace = remove_final_solution(current_trace)
                    forced_trace = force.append_force(current_trace)
                    self.messages[-1]['content'] = forced_trace
                    current_trace = await self.generate(extra_body={"continue_final_message": True, "add_generation_prompt": False})
                    current_trace = self.messages[-1]['content'] + current_trace.replace("</think>", "</think >")
                    force.max_repetitions -= 1
                    self.applied_forces.append(force)
        current_trace = remove_final_solution(current_trace)
        self.applied_forces.append(self.final_force)
        forced_trace = self.final_force.append_force(current_trace)
        self.messages[-1]['content'] = forced_trace
        final_trace = await self.generate(extra_body={"continue_final_message": True, "add_generation_prompt": False})
        final_trace = self.messages[-1]['content'] + final_trace.replace("</think>", "</think >")
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