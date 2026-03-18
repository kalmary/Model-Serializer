from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelTracker:
    number_of_models_to_track: int
    model_names: list[str]
    min_or_max: Literal["min", "max"]

    def __post_init__(self):
        if self.min_or_max == "max":
            self.best_objective_value: float = float('-inf')
        else:
            self.best_objective_value: float = float('inf')

    def update_if_better(self, objective:float):
        if self.min_or_max == "min" and objective < self.best_objective:
                self.best_objective = objective
                return True
        elif self.min_or_max == "max" and objective > self.best_objective:
                self.best_objective = objective
                return True
        return False
    
    def update_model_names(self, model_name: str):
         pass
         
         
        








"""
we want to choose how many models we save (between 1-3)
We save the optuna model and training configs.
Every time there is a better model (based on min/max of some metric) we save it as a new step with:
- metrics
- configs as artifacts - that are in a folder with that models name
- 
"""