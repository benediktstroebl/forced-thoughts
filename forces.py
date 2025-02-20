from abc import ABC, abstractmethod

def remove_final_solution(text: str) -> str:
    marker = "<answer>"
    idx = text.find(marker)
    if idx != -1:
        return text[:idx].strip()
    return text

class Force(ABC):
    def __init__(self, name: str, force_string: str, repetitions: int):
        self.name = name
        self.force_string = force_string
        self.repetitions = repetitions

    def append_force(self, reasoning: str) -> str:
        return reasoning + "\n\n" + self.force_string
    
    def to_dict(self):
        return {"name": self.name, "force_string": self.force_string, "repetitions": self.repetitions}

# Example modular forces that can be extended
class BudgetForce(Force):
    pass

class ApproachForce(Force):
    pass

class LanguageForce(Force):
    pass 