from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

# TODO: Include the hyperparameters your script will need over here.
hyperparameters = {
    "epochs": 2,
    "batch-size": 64,
    "test-batch-size": 100,
    "learning-rate": 0.001,
}

# TODO: Create your estimator here. You can use Pytorch or any other framework.
estimator = PyTorch(
    entry_point="scripts/pytorch_cifar.py",
    role=get_execution_role(),
    hyperparameters=hyperparameters,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    framework_version="1.6.0",
    py_version="py3",
)

#TODO: Start Training
estimator.fit()