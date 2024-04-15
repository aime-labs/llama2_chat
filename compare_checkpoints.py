import torch
from pathlib import Path
from deepdiff import DeepDiff
import json
import os

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compare_models(checkpoint_path_1, checkpoint_path_2):
    checkpoint_1 = torch.load(checkpoint_path_1, map_location="cpu")
    checkpoint_2 = torch.load(checkpoint_path_2, map_location="cpu")
    checkpoint_1 = dict(sorted(checkpoint_1.items()))
    checkpoint_2 = dict(sorted(checkpoint_2.items()))
    models_differ = False
    k=1
    missing_keys_checkpoint_2 = set(checkpoint_1.keys()) - set(checkpoint_2.keys())
    if missing_keys_checkpoint_2:
        models_differ = True
        print(f"Keys missing in checkpoint_2: {missing_keys_checkpoint_2}")

    missing_keys_checkpoint_1 = set(checkpoint_2.keys()) - set(checkpoint_1.keys())
    if missing_keys_checkpoint_1:
        models_differ = True
        print(f"Keys missing in checkpoint_1: {missing_keys_checkpoint_1}")

    for key, value_1 in checkpoint_1.items():
        if key not in checkpoint_2:
            continue  # Skip keys that are missing in checkpoint_2

        value_2 = checkpoint_2[key]

        try:
            if value_1.dtype != value_2.dtype:
                models_differ = True
                print(f'Precision difference for key: {key} in file: {checkpoint_path_1}')
        except AttributeError:
            pass

        try:
            if value_1.shape != value_2.shape:
                models_differ = True
                print(f'Shape difference for key: {key} in file: {checkpoint_path_1}')
        except AttributeError:
            pass

        try:
            if not torch.equal(value_1, value_2):
                indices = torch.nonzero(value_1 != value_2)
                if indices.numel() == 0:
                    print(f'Tensors are equal for key: {key} in file: {checkpoint_path_1}.')
                else:
                    print('\n#####', key, value_1, '\n####', value_2)
                    print(f'Tensors differ at indices {indices} for key: {key} in file: {checkpoint_path_1}.')
                models_differ = True
        except TypeError:
            if value_1 != value_2:
                print(f'Mismatch found at key: {key} in file: {checkpoint_path_1}. Value 1: {value_1}, Value 2: {value_2}')
                models_differ = True

    if not models_differ:
        print(f'Checkpoints {checkpoint_path_1} and {checkpoint_path_2} match perfectly! :)')


def main():

    ckpt_dir_1='/data/models/llama2/llama-2-70b-chat/'
    ckpt_dir_2='/data/models/llama2/llama-2-70b-chat/'
    checkpoints_1 = sorted(Path(ckpt_dir_1).glob(f'merged.2GPUs.*.pth'))
    checkpoints_2 = sorted(Path(ckpt_dir_2).glob(f'merged_git.2GPUs.*.pth'))
    print('checkpoints_1', checkpoints_1)
    print('checkpoints_2', checkpoints_2)

    for checkpoint_path_1, checkpoint_path_2 in zip(checkpoints_1, checkpoints_2):
        compare_models(checkpoint_path_1, checkpoint_path_2)

if __name__ == "__main__":
    main()
