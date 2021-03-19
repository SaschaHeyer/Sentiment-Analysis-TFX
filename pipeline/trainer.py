import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from typing import Text

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
from tfx.components.trainer.executor import TrainerFnArgs
import os
from typing import Dict, List, Text

LABEL_KEY = 'label'
BERT_TFHUB_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def load_bert_layer(model_url=BERT_TFHUB_URL):
    bert_layer = hub.KerasLayer(handle=model_url, trainable=False)
    return bert_layer


def get_model(tf_transform_output, max_seq_length=128):
    feature_spec = tf_transform_output.transformed_feature_spec()
    feature_spec.pop(LABEL_KEY)

    inputs = {
        key: tf.keras.layers.Input(shape=(max_seq_length),
                                   name=key,
                                   dtype=tf.int64)
        for key in feature_spec.keys()
    }

    input_word_ids = tf.cast(inputs["input_word_ids"], dtype=tf.int32)
    input_mask = tf.cast(inputs["input_mask"], dtype=tf.int32)
    input_type_ids = tf.cast(inputs["input_type_ids"], dtype=tf.int32)

    bert_layer = load_bert_layer()
    encoder_inputs = dict(
        input_word_ids=tf.reshape(input_word_ids, (-1, max_seq_length)),
        input_mask=tf.reshape(input_mask, (-1, max_seq_length)),
        input_type_ids=tf.reshape(input_type_ids, (-1, max_seq_length)),
    )
    outputs = bert_layer(encoder_inputs)

    x = tf.keras.layers.Dense(256, activation='relu')(outputs["pooled_output"])
    dense = tf.keras.layers.Dense(64, activation='relu')(x)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    keras_model = tf.keras.Model(inputs=[
        inputs['input_word_ids'], inputs['input_mask'],
        inputs['input_type_ids']
    ],
                                 outputs=pred)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)

    keras_model.compile(optimizer=optimizer,
                        loss='binary_crossentropy',
                        metrics=metrics)

    return keras_model


# serve function for raw text
def _get_serve_tf_fn(model, tf_transform_output):

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(text):
        reshaped_text = tf.reshape(text, [-1, 1])
        transformed_features = model.tft_layer({"review": reshaped_text})

        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _input_fn(file_pattern: Text,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 32) -> tf.data.Dataset:
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=LABEL_KEY)

    return dataset


def run_fn(fn_args: TrainerFnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 32)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 32)

    # Mirrored Strategy is useful when running on multiple GPU's on a single machine.
    # This is the most common default strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = get_model(tf_transform_output=tf_transform_output)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir,
        update_freq='batch',
        profile_batch='50,60')

    model.fit(train_dataset,
              epochs=5,
              steps_per_epoch=fn_args.train_steps,
              validation_data=eval_dataset,
              validation_steps=fn_args.eval_steps,
              callbacks=[tensorboard_callback])

    # testing {"instances": ["the book was bad"]}
    signatures = {
        'serving_default':
        _get_serve_tf_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='text')),
    }

    print(fn_args.serving_model_dir)
    model.save(fn_args.serving_model_dir,
               save_format='tf',
               signatures=signatures)
