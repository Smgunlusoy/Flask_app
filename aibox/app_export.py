# Utilities to export the hand and object detection model as tflite models enriched with metadate for use in Android Studio

import argparse
import tensorflow as tf

# Library for tflite metadata as recommended in: https://www.tensorflow.org/lite/models/convert/metadata
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

from six import text_type
from pathlib import Path


def get_hand_model_meta() -> _metadata_fb.ModelMetadataT:
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "YOLOv5 Hand Detection"
    model_meta.description = "A YOLOv5 model trained on the Egohands dataset."
    model_meta.version = "v1"
    model_meta.author = "Optivist Research Group"
    return model_meta


def get_object_model_meta() -> _metadata_fb.ModelMetadataT:
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "YOLOv5 Object Detection"
    model_meta.description = "A YOLOv5 object detection model."
    model_meta.version = "v1"
    model_meta.author = "Optivist Research Group"
    return model_meta


def get_input_meta() -> _metadata_fb.TensorMetadataT:
    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"
    input_meta.description = "Input image to be classified."
    input_meta.content = _metadata_fb.ContentT()
    input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
    input_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB
    input_meta.content.contentPropertiesType = (
        _metadata_fb.ContentProperties.ImageProperties
    )
    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (
        _metadata_fb.ProcessUnitOptions.NormalizationOptions
    )
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = [127.5]
    input_normalization.options.std = [127.5]
    input_meta.processUnits = [input_normalization]
    input_stats = _metadata_fb.StatsT()
    input_stats.max = [255]
    input_stats.min = [0]
    input_meta.stats = input_stats
    return input_meta


def get_hand_output_meta() -> _metadata_fb.TensorMetadataT:
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "output tensor"
    output_meta.description = "A float32 tensor consisting of 4 one-hot-encoded classes [myleft, myright, yourleft, yourright], the bounding box described by [x, y, width, height], and the confidence (also called objectness)."

    return output_meta


def get_object_output_meta() -> _metadata_fb.TensorMetadataT:
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.name = "output tensor"
    output_meta.description = "A float32 tensor for 80 hot-encoded classes of the COCO dataset, the bounding box described by [x, y, width, height], and the confidence (also called objectness)."
    return output_meta


def generate_metadata(tflite_model, model_type: str) -> bytes | text_type:
    if model_type == "hands":
        model_meta = get_hand_model_meta()
        output_meta = get_hand_output_meta()
    elif model_type == "objects":
        model_meta = get_object_model_meta()
        output_meta = get_object_output_meta()
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'")

    # Create input, output and graph metadata
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [get_input_meta()]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    # Put the metadata into the .tflite binary object
    b = flatbuffers.Builder(0)
    b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()
    populator = _metadata.MetadataPopulator.with_model_buffer(tflite_model)
    populator.load_metadata_buffer(metadata_buf)
    populator.populate()
    return populator.get_model_buffer()


def export_tflite_model(
    tf_saved_model_dir: str | Path, storage_path: str | Path, model_type: str
) -> None:
    tf_saved_model_dir = Path(tf_saved_model_dir)
    storage_path = Path(storage_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_saved_model_dir))

    # Apply post-training float16 quantization for reduced model size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]

    tflite_model = converter.convert()

    # Create and add the metadata to the model
    tflite_model = generate_metadata(tflite_model, model_type)

    with storage_path.open("wb") as save_loc:
        save_loc.write(tflite_model)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hands", type=str, required=False, help="hand saved_model path"
    )
    parser.add_argument(
        "--objects", type=str, required=False, help="object saved_model path"
    )
    parser.add_argument("--output", type=str, required=True, help="output path")
    return parser.parse_args()


def main(opts) -> None:
    commands = vars(opts)

    if commands["hands"] and commands["objects"]:
        return None
    elif commands["hands"] is not None:
        print(f"Exporting tensorflow-lite model from: {commands['hands']}.")
        save_model_dir = commands["hands"]
        model_type = "hands"
    elif commands["objects"] is not None:
        print(f"Hands option was set {commands['objects']}.")
        save_model_dir = commands["objects"]
        model_type = "objects"
    else:
        return None

    export_tflite_model(save_model_dir, commands["output"], model_type)

    print(f"Done. Exported model to {commands['output']}.")


if __name__ == "__main__":
    opts = parse_opt()
    main(opts)

    # Assumes yolo models already exported as tf-saved_model

    # displayer = _metadata.MetadataDisplayer.with_model_file("test.tflite")
    # print(displayer.get_metadata_json())

    # python app_export --hands <saved_model_path> --output <outputfile_name>
    # python app_export --objects <saved_model_path> --output <outputfile_name>