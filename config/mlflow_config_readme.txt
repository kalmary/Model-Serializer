tracking_uri - location of the MLflow tracking server
registry_uri - location of the model registry backend
experiment_name - name of the experiment
create_experiment_if_missing - create experiment if it doesn't create_experiment_if_missing
run_name_template - template of the run name
nested_runs - Controls whether runs can be children of other runs (not sure if needed)
experiment_description - description of the experiment (what if the description already exist plus can we have run descriptions?)
tags - metadata attached to runs that allows to filter tehm in UI
register_model - add trained model to model registry
registered_model_name - name of name of the model i the registry
registry_stage - lifecycle stage to the model version like staging or production
artifact_root - Base directory where artifacts are stored inside a run.
checkpoint_dir - Folder for intermediate training weights
plot_dir - Stores visualization outputs
model_dir - Directory where the final MLflow model package is stored



Architecture of projects:
Experiments
    vision-project-Kuba
    vision-project-Michal