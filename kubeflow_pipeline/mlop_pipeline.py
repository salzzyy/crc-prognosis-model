import kfp
from kfp import dsl
import kfp.compiler


def data_processing_op():
    return dsl.ContainerOp(
        name="Data Processing",
        image="salzzy/crc-app-mlop:latest",
        command=["python", "src/data_processing.py"],
    )


def model_training_op():
    return dsl.ContainerOp(
        name="Model Training",
        image="salzzy/crc-app-mlop:latest",
        command=["python", "src/model_training.py"],
    )


@dsl.pipeline(
    name="MLOPs Pipeline",
    description="This is to connect model processing to model training",
)
def mlop_pipeline():
    data_processing = data_processing_op()
    model_training = model_training_op().after(data_processing)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(mlop_pipeline, "mlops_pipeline.yaml")
