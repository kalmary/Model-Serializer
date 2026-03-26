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


def training_run():

    # TRAINING RUN
    optuna_model_config_path = "config/config_model_randlanet_0.json"
    optuna_training_config_path = "config/config_train_randlanet.json"

    config = MLFlowConfig.from_json('src/tests/example/mlflow_config.json')
    client = config.apply(run_name="Training Run")

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

def evaluation_run(best_model_name: str):

    # EVALUATION RUN
    config = MLFlowConfig.from_json('src/tests/example/mlflow_config.json')
    client = config.apply("Evaluation Run")
    eval_logger = MLFlowTracker(client=client, config=config, min_or_max=None)
    eval_logger.log_dataset(path="config/wynik 1.las")

    # Load the best model from training by name
    loaded_model, config_paths = eval_logger.load_model(model_name=best_model_name)

    # Run evaluation with loaded model (fake inference here)
    test_metrics = {"accuracy": 0.88, "loss": 0.38}
    model = Model(model=loaded_model,
                    metrics=test_metrics,
                    metrics_art={"cm": [[50, 2], [1, 47]]},
                    configs=config_paths,
                    best_val=test_metrics["accuracy"])
    eval_logger.log_evaluation(model=model, model_name=best_model_name)
    config.end_run()

def main():

    training_run()
    evaluation_run(best_model_name="Resnet_2026-03-26_11-43-08_5")



if __name__ == "__main__":
    main()
