from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from pydantic import BaseModel

def remove_final_solution(text: str) -> str:
    marker = "<answer>"
    idx = text.find(marker)
    if idx != -1:
        return text[:idx].strip()
    return text

class Force(ABC):
    def __init__(self, name: str, force_string: str, max_repetitions: int):
        self.name = name
        self.force_string = force_string
        self.max_repetitions = max_repetitions

    def append_force(self, reasoning: str) -> str:
        return reasoning + "\n\n" + self.force_string
    
    def to_dict(self):
        return {"name": self.name, "force_string": self.force_string, "max_repetitions": self.max_repetitions}

# Example modular forces that can be extended
class BudgetForce(Force):
    pass

class ApproachForce(Force):
    pass

class LanguageForce(Force):
    pass

class ApproachForce(Force):
    def __init__(self, approach: str):
        # Convert approach into a force string that prompts model to consider this approach
        force_string = f"Wait, before returning the final solution, let's consider this approach: {approach}. Perhabs this may help me solve the problem."
        super().__init__(name=f"approach_force_{'_'.join(approach.lower().replace(' ', '_').split('_')[:5])}", force_string=force_string, max_repetitions=1)


class ApproachesResponse(BaseModel):
    approaches: list[str]

class ApproachForceGenerator:
    def __init__(self, model,task: str, max_tokens=2048, temperature=0.7):
        self.client = AsyncOpenAI()
        self.task = task
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        if self.task in ["taco_hard", "taco_medium"]:
            self.system_prompt = """You are an expert Python programmer. You will be given a competitive programming question.
            Return several useful, non-obvious approaches of how one could go about solving the problem.
            
            Be creative and go beyond intuitive approaches. Do not number your approaches."""
        elif self.task == "aime24":
            self.system_prompt = """You are an expert mathematician. You will be given a competitive mathematics question.
            Return several useful, non-obvious approaches of how one could go about solving the problem.
            
            Be creative and go beyond intuitive approaches. Do not number your approaches."""
        elif self.task == "gpqa_diamond":
            self.system_prompt = """You are an expert in reasoning and question answering. You will be given a question.
            Return several useful, non-obvious things to consider (approaches) and relevant information that could help answer the question.
            
            Be creative and go beyond intuitive or obvious approaches to the question. Do not number your approaches."""
    
    async def generate_forces(self, problem_spec: str) -> list[ApproachForce]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem_spec}
        ]
        
        approaches = []
        while len(approaches) < 3:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
            messages=messages,
            # max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format=ApproachesResponse
            )
        
            approaches_response = response.choices[0].message.parsed
            approaches = approaches_response.approaches
        
        return [ApproachForce(approach) for approach in approaches if approach] 