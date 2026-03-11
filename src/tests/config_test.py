from src.ml_flow_utils.config import MLFlowConfig

config = MLFlowConfig.from_json('/home/michal/code/Model-Serializer/config/mlflow_config.json')

config.apply()
