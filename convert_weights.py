import argparse
import json
import os
import shutil
import torch
from pathlib import Path
"""
Sample usage:
    ```
    python convert_weights.py --input_dir /models/llama/ --model_size 13B
    ```
"""


NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "70B": 8,
}


def read_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())

def write_model(input_base_path, model_size, num_gpus):

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    print('num_shards', num_shards)
    n_layers = params["n_layers"]
    print('n_layers', n_layers)
    n_heads = params["n_heads"]
    print('n_heads', n_heads)
    n_heads_per_shard = n_heads // num_shards
    print('n_heads_per_shard', n_heads_per_shard)
    dim = params["dim"]
    print('dim', dim)
    
    if "n_kv_heads" in params:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim



    dims_per_head = dim // n_heads
    print('dims_per_head', dims_per_head)
    # Load weights
    if model_size == "7B":
        loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
    else:
        #loaded = [
        #    torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
        #    for i in range(num_shards-4)
        #]
        loaded = []
        for i in range(num_shards):
            print('loading ', os.path.join(input_base_path, f"consolidated.{i:02d}.pth"))      
            loaded.append(torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu"))

    #print('shape: ', loaded[1][f"layers.{6}.attention.wq.weight"].shape)
    state_dicts = [{} for i in range(num_gpus)]
    for layer_i in range(n_layers):
        print('layer: ', layer_i)
        if model_size == "7B":
            for state_dict in state_dicts:
                state_dict.update({
                    f"layers.{layer_i}.attention.wq.weight": loaded[
                        f"layers.{layer_i}.attention.wq.weight"
                    ],
                    f"layers.{layer_i}.attention.wk.weight": loaded[
                        f"layers.{layer_i}.attention.wk.weight"
                    ],
                    f"layers.{layer_i}.attention.wv.weight": loaded[
                        f"layers.{layer_i}.attention.wv.weight"
                    ],
                    f"layers.{layer_i}.attention.wo.weight": loaded[
                        f"layers.{layer_i}.attention.wo.weight"
                    ],
                    f"layers.{layer_i}.feed_forward.w1.weight": loaded[
                        f"layers.{layer_i}.feed_forward.w1.weight"
                    ],
                    f"layers.{layer_i}.feed_forward.w2.weight": loaded[
                        f"layers.{layer_i}.feed_forward.w2.weight"
                    ],
                    f"layers.{layer_i}.feed_forward.w3.weight": loaded[
                        f"layers.{layer_i}.feed_forward.w3.weight"
                    ],
                    f"layers.{layer_i}.attention_norm.weight": loaded[
                        f"layers.{layer_i}.attention_norm.weight"
                    ],
                    f"layers.{layer_i}.ffn_norm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
                })
        else:
            for gpu_id, state_dict in enumerate(state_dicts):
                state_dict.update({
                    f"layers.{layer_i}.attention_norm.weight": loaded[0][
                        f"layers.{layer_i}.attention_norm.weight"
                    ].clone(),
                    f"layers.{layer_i}.ffn_norm.weight": loaded[0][f"layers.{layer_i}.ffn_norm.weight"].clone(),
                })
                state_dict[f"layers.{layer_i}.attention.wq.weight"] = torch.cat(
                    [
                        loaded[2*gpu_id+i][f"layers.{layer_i}.attention.wq.weight"].clone().view(n_heads_per_shard, dims_per_head, dim).clone()
                        for i in range(int(num_shards/num_gpus))
                    ],
                    dim=0,
                ).reshape(int(dim/num_gpus), dim).clone()
                state_dict[f"layers.{layer_i}.attention.wk.weight"] = torch.cat(
                    [
                        loaded[2*gpu_id+i][f"layers.{layer_i}.attention.wk.weight"].clone().view(num_local_key_value_heads, dims_per_head, dim).clone()
                        for i in range(int(num_shards/num_gpus))
                    ],
                    dim=0,
                ).reshape(int(key_value_dim/num_gpus), dim).clone()
                state_dict[f"layers.{layer_i}.attention.wv.weight"] = torch.cat(
                    [
                        loaded[2*gpu_id+i][f"layers.{layer_i}.attention.wv.weight"].clone().view(num_local_key_value_heads, dims_per_head, dim).clone()
                        for i in range(int(num_shards/num_gpus))
                    ],
                    dim=0,
                ).reshape(int(key_value_dim/num_gpus), dim).clone()
                state_dict[f"layers.{layer_i}.attention.wo.weight"] = torch.cat(
                    [loaded[2*gpu_id+i][f"layers.{layer_i}.attention.wo.weight"].clone() for i in range(int(num_shards/num_gpus))], dim=1
                ).clone()
                state_dict[f"layers.{layer_i}.feed_forward.w1.weight"] = torch.cat(
                    [loaded[2*gpu_id+i][f"layers.{layer_i}.feed_forward.w1.weight"].clone() for i in range(int(num_shards/num_gpus))], dim=0
                ).clone()
                state_dict[f"layers.{layer_i}.feed_forward.w2.weight"] = torch.cat(
                    [loaded[2*gpu_id+i][f"layers.{layer_i}.feed_forward.w2.weight"].clone() for i in range(int(num_shards/num_gpus))], dim=1
                ).clone()
                state_dict[f"layers.{layer_i}.feed_forward.w3.weight"] = torch.cat(
                    [loaded[2*gpu_id+i][f"layers.{layer_i}.feed_forward.w3.weight"].clone() for i in range(int(num_shards/num_gpus))], dim=0
                ).clone()

    if model_size == "7B":
        for gpu_id, state_dict in enumerate(state_dicts):
            state_dict.update({
                "tok_embeddings.weight": loaded["tok_embeddings.weight"],
                "norm.weight": loaded["norm.weight"],
                "output.weight": loaded["output.weight"],
            })
            torch.save(state_dict, Path(input_base_path) / f'merged.{num_gpus}GPUs.{gpu_id:02d}.pth')
    else:
        for gpu_id, state_dict in enumerate(state_dicts):
            print('saving... : gpu_id: ', gpu_id)
            state_dict.update({
                "norm.weight": loaded[0]["norm.weight"].clone(),
                "tok_embeddings.weight": torch.cat(
                    [loaded[2*gpu_id+i]["tok_embeddings.weight"].clone() for i in range(int(num_shards/num_gpus))], dim=1
                ).clone(),
                "output.weight": torch.cat([loaded[2*gpu_id+i]["output.weight"].clone() for i in range(int(num_shards/num_gpus))], dim=0).clone(),
            })
            torch.save(state_dict, Path(input_base_path) / f'merged.{num_gpus}GPUs.{gpu_id:02d}.pth')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "70B"],
    )
    parser.add_argument(
        "--num_gpus", type=int
    )
    args = parser.parse_args()
    write_model(
        input_base_path=os.path.join(args.input_dir, f'llama-2-{args.model_size.lower()}-chat'),
        model_size=args.model_size, num_gpus = args.num_gpus
    )


if __name__ == "__main__":
    main()