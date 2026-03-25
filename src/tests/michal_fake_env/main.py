import torch.nn as nn
from src.config import MLFlowConfig
from src.tracker import MLFlowTracker, Model
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s", force=True)
logging.getLogger("mlflow").setLevel(logging.ERROR)


accuracy = [0.1, 0.4, 0.35, 0.6, 0.8, 0.90, 0.85]
loss = [0.9, 0.8, 0.75, 0.6, 0.5, 0.4, 0.35]



def main():

    # TRAINING RUN
    optuna_model_config_path = "config/config_model_randlanet_0.json"
    optuna_training_config_path = "config/config_train_randlanet.json"

    config = MLFlowConfig.from_json('src/tests/michal_fake_env/mlflow_config.json')
    client = config.apply("Training Run")

    #logging initial configs and dataset
    logger = MLFlowTracker(client=client, config=config, min_or_max="max")
    logger.log_config(config_path=optuna_model_config_path, save_config_for_model=False, save_as_parameters=True)
    logger.log_config(config_path=optuna_training_config_path, save_config_for_model=False, save_as_parameters=False)
    logger.log_dataset(path="config/wynik 1.las")

    for idx, i in enumerate(accuracy):
        model_config_path = "config/config_model_randlanet_1.json"
        training_config_path = "config/config_train_single_randlanet.json"
        model = Model(model=nn.Sequential(nn.Linear(2,1)),
                      metrics={"accuracy": i, "loss": loss[idx]},
                      metrics_art={"cm": [[50, 2], [1, 47]]},
                      configs=[model_config_path, training_config_path],
                      best_val=i)
        print(model) 
        logger.log_training(model=model, model_name="Resnet", number_of_models_to_track=2, step=idx)

    config.end_run()

    # EVALUATION RUN

    eval_model_config_path = "config/config_model_randlanet_1.json"
    client = config.apply("Evaluation Run")
    logger = MLFlowTracker(client=client, config=config, min_or_max=None)
    logger.log_dataset(path="config/wynik 1.las")

    test_metrics = {"accuracy": 0.88, "loss": 0.38}
    model = Model(model=nn.Sequential(nn.Linear(2,1)),
                    metrics=test_metrics,
                    metrics_art={"cm": [[50, 2], [1, 47]]},
                    configs=[eval_model_config_path],
                    best_val=test_metrics["accuracy"])
    logger.log_evaluation(model=model, model_name="Resnet_Eval")




if __name__ == "__main__":
    main()
