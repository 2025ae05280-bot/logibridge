import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.config import REPO_ROOT
from data_pipeline.preprocessing import load_stats, normalise


def representative_data(dataset, mean, std):
    with Path(dataset).open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"][:250]
    for row in rows:
        values = [float(row[name]) for name in ("temp_mean", "temp_std", "temp_roc_c_per_min", "vib_rms", "vib_peak", "vib_kurtosis")]
        yield [normalise(values, mean, std).reshape(1, 6)]


def convert(model_path, output_path, dataset, mean, std):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data(dataset, mean, std)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    Path(output_path).write_bytes(converter.convert())


def main():
    mean, std = load_stats(REPO_ROOT / "data_pipeline" / "training_stats.npy")
    model_dir = REPO_ROOT / "training" / "models"
    dataset = REPO_ROOT / "training" / "data" / "dataset.csv"
    convert(model_dir / "m1_fp32.keras", model_dir / "m2_ptq_int8.tflite", dataset, mean, std)
    import tensorflow as tf
    float_model = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(model_dir / "m1_fp32.keras")).convert()
    (model_dir / "m1_fp32.tflite").write_bytes(float_model)


if __name__ == "__main__":
    main()
