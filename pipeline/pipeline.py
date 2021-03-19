from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfx.orchestration import pipeline

import tfx
import os
import tensorflow as tf
from tfx import components
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.proto import example_gen_pb2
from tfx.utils.dsl_utils import external_input
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.components import Evaluator
from tfx.components import Pusher
from tfx.proto import  pusher_pb2
import tensorflow_model_analysis as tfma

from tfx.components import Trainer
from tfx.dsl.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.proto import trainer_pb2
from tfx.components.trainer import executor as trainer_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor

#resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver

#evaluator
from tfx.components import Evaluator
import tensorflow_model_analysis as tfma

#pusher
from tfx.components import Pusher
from tfx.proto import  pusher_pb2

def create_pipeline(pipeline_name, pipeline_root):

    components = []

    # ExampleGen
    output = example_gen_pb2.Output(split_config=example_gen_pb2.SplitConfig(
        splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=3),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
        ]))

    examples = external_input(
        "gs://machine-learning-samples/datasets/sentiment/imdb/tfx")
    example_gen = CsvExampleGen(input=examples, output_config=output)
    components.append(example_gen)

    # StatisticsGen
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    components.append(statistics_gen)

    # SchemaGen
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
                           infer_feature_shape=True)
    components.append(schema_gen)

    # ExampleValidator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])
    components.append(example_validator)

    # Transform
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file="transform.py")
    components.append(transform)
   

    # Trainer
    TRAINING_STEPS = 2060
    EVALUATION_STEPS = 1000

    trainer = Trainer(
    module_file="trainer.py",
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=TRAINING_STEPS),
    eval_args=trainer_pb2.EvalArgs(num_steps=EVALUATION_STEPS))
    
    components.append(trainer)

    # Resolver
    model_resolver =  Resolver(
        instance_name= "latest_blessed_model_resolver", 
        strategy_class= latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    )
    components.append(model_resolver)

    # Evaluator
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='label')],
        slicing_specs=[tfma.SlicingSpec()], # empty slice indidacate whole dataset
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.7}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 0.0001})))
            ])
        ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )
    components.append(evaluator)
    
    # Pusher
    location = "gs://doit-sascha-data/sentiment/model"

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'], #disabled due to reduced steps for demonstration purposes
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=location))
    )
    components.append(pusher)

    return pipeline.Pipeline(pipeline_name=pipeline_name,
                             pipeline_root=pipeline_root,
                             components=components)
