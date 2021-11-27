from sys import argv
from math import exp
from pandas import read_csv
from os import mkdir
from os.path import join, exists
from shutil import rmtree
from t5.models import MtfModel
from seqio import SentencePieceVocabulary
from glob import glob

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(csv_path):
    STEPS = 43045
    MODEL_DIR = "./model"
    OUT_DIR = "./out"
    TEMP_DIR = "./temp"
    TOKENIZER_PATH = "./tokenizer/tokenizer.model"

    method_df = read_csv(csv_path)
    method_codes = method_df["method_code"].tolist()
    method_names = method_df["method_name"].tolist()

    if not exists(OUT_DIR):
        mkdir(OUT_DIR)
    mkdir(TEMP_DIR)
    inputs_path = join(TEMP_DIR, "inputs.txt")
    targets_path = join(TEMP_DIR, "targets.txt")
    predictions_path = join(TEMP_DIR, "predictions.txt")
    scores_path_prefix = join(TEMP_DIR, "predictions_score")

    with open(inputs_path, "w") as inputs_file:
        inputs_file.write('\n'.join(method_codes))

    with open(targets_path, "w") as targets_file:
        targets_file.write('\n'.join(method_names))

    # Load the Vocabulary
    VOCABULARY = SentencePieceVocabulary(TOKENIZER_PATH, extra_ids=100)

    # Load the Model
    model = MtfModel(model_dir=MODEL_DIR, tpu=None, batch_size=256)

    # Model predicts inputs
    model.predict(inputs_path, predictions_path, checkpoint_steps=STEPS, vocabulary=VOCABULARY)

    # Compute scores of predictions
    prediction_files = glob(f"{predictions_path}-*")
    prediction_file = sorted(prediction_files, key=lambda path: int(path.split('-')[-1]))[-1]

    # Change binary representation into string
    modify_file_representation(prediction_file)

    # Add predictions to frame
    predictions = open(prediction_file, "r")
    method_df["predicted_name"] = [line[:-1] for line in predictions.readlines()]
    method_df["perfect_prediction"] = method_df["method_name"] == method_df["predicted_name"]

    try:
        model.score(inputs=inputs_path, targets=prediction_file, scores_file=scores_path_prefix,
                    checkpoint_steps=STEPS, vocabulary=VOCABULARY)
    except AttributeError:
        pass
    finally:
        # Compute confidence
        confidence_strings = open(scores_path_prefix + ".scores", "r")
        confidence_scores = [round(exp(float(confidence))*100.0, 3) for confidence in confidence_strings.readlines()]
        method_df["confidence"] = confidence_scores

        # Compute accuracy
        total_predictions = method_df["perfect_prediction"].size
        perfect_predictions = method_df["perfect_prediction"].sum()
        model_accuracy = perfect_predictions * 100.0 / total_predictions
        print("=" * 50)
        print(f"Instances: {total_predictions}\t\tModel Accuracy: {model_accuracy:.2f}% (pp={perfect_predictions})")
        print("=" * 50)

        # Export to CSV
        columns = ["file", "method_name", "predicted_name", "perfect_prediction", "confidence", "method_code"]
        method_df[columns].to_csv(join(OUT_DIR, "predictions.csv"), index=False)
        rmtree(TEMP_DIR)


def modify_file_representation(prediction_file):
    with open(prediction_file, 'r') as file:
        list_of_lines = file.readlines()
        list_of_lines = [line[2:-2] + '\n' for line in list_of_lines]

    with open(prediction_file, "w") as file:
        file.writelines(list_of_lines)


if __name__ == "__main__":
    main(argv[1])
