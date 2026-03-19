from dataclasses import dataclass
from typing import Literal
from collections import deque
import random

@dataclass
class Model:
     model_name: str
     #metrics: dict
     #config: dict
     best_val: float
 
class ModelTracker ():
    def __init__(self, number_of_models_to_track: int, min_or_max: Literal["min", "max"]) -> None:
        self.number_of_models_to_track = number_of_models_to_track
        self.min_or_max = min_or_max
        self.models_list = deque(maxlen=self.number_of_models_to_track)

        if self.min_or_max == "max":
            self.best_objective: float = float('-inf')
        else:
            self.best_objective: float = float('inf')

    def update_if_better(self, objective:float):

        if self.min_or_max == "min" and objective < self.best_objective:
                self.best_objective = objective
                return True
        elif self.min_or_max == "max" and objective > self.best_objective:
                self.best_objective = objective
                return True
        return False
    
    def update_models_tracked(self, model: Model, objective: float):

        if self.update_if_better(objective):
            self.models_list.append(model)
             
        
         

def test_classes():
     i = 0
     model_tracker = ModelTracker(3, "min")
     while i<10:
          model_name = f"model_{i}"
          acc_value = random.uniform(1 - (i+1)/10, 1 - i/10)
          model = Model(model_name, acc_value)
          model_tracker.update_models_tracked(model, model.best_val)
          print(model_tracker.models_list)
          print(f"Epoch:{i}")
          i += 1

test_classes()









"""
we want to choose how many models we save (between 1-3)
We save the optuna model and training configs.
Every time there is a better model (based on min/max of some metric) we save it as a new step with:
- metrics
- configs as artifacts - that are in a folder with that models name
- 
"""