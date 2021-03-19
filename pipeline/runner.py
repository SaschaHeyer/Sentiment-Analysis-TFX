import os
from absl import logging


from pipeline import create_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.utils import telemetry_utils

PIPELINE_NAME = 'sentiment3'
GOOGLE_CLOUD_PROJECT = 'sascha-playground-doit'
GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-kubeflowpipelines-default'
OUTPUT_DIR = os.path.join('gs://', GCS_BUCKET_NAME)
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, PIPELINE_NAME)


def run():
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    tfx_image = "gcr.io/sascha-playground-doit/sentiment-pipeline"
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config, tfx_image=tfx_image)

    kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
                                              create_pipeline(
                                                  pipeline_name=PIPELINE_NAME,
                                                  pipeline_root=PIPELINE_ROOT))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()
