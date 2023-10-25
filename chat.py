# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

from llama import Llama, Dialog

import argparse
from pathlib import Path
import os
import torch

WORKER_JOB_TYPE = "llama2"
WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d1"

def main():
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    args = load_flags()
    if not args.tokenizer_path:
        args.tokenizer_path = str(Path(args.ckpt_dir).parent / 'tokenizer.model')
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.api_server:
        from api_worker_interface import APIWorkerInterface
        api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, WORKER_AUTH_KEY, args.gpu_id, world_size=world_size, rank=local_rank)
        callback = ProcessOutputCallback(local_rank, api_worker)
        ctx = ""
        while True:
            prompts = []
            
            job_data = api_worker.job_request()
            if local_rank == 0:
                #print('job_data: ', job_data)
                callback.job_data = job_data
                ctx = job_data['text']
                prompts.append(ctx)
            else:
                prompts.append("")

        #        torch.distributed.barrier()    # not useable! Does active CPU waiting and times out with an error after about 30 minutes!

            torch.distributed.broadcast_object_list(prompts, 0)
            top_p = get_parameter('top_p', float, 0.9, args, job_data, local_rank)
            top_k = get_parameter('top_k', int, 40, args, job_data, local_rank)
            temperature = get_parameter('temperature', float, 0.8, args, job_data, local_rank)

            results = generator.generate(
                callback.process_output, prompts, max_gen_len=512, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=args.repetition_penalty
            )

            ctx = results[0]
    else:
        callback = ProcessOutputToShellCallback()
        ctx = """A dialog, where User interacts with an helpful, kind, obedient, honest and very reasonable assistant called Dave.
User: Hello, Dave.
Dave: How can I assist you today?
"""
        print(ctx)
        while True:
            
            prompt = input(f'User: ')
            if ctx != "":
                ctx = ctx + "User: " + prompt + "\n"
            else:
                ctx = prompt + "\n"
            
            prompts = [ctx]
            if not args.temperature:
                args.temperature = 0.8
            if not args.top_p:
                args.top_p = 0.9
            if not args.top_k:
                args.top_k = 40
            results = generator.generate(
                callback.process_output, prompts, max_gen_len=512, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty
            )
            ctx = callback.ctx
            



def load_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir", type=str, required=False,
        help="Location of LLama weights",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=False,
        help="Location of tokenizer"
    )
    parser.add_argument(
        '--temperature', type=float, required=False,
    help='Temperature'
                    )
    parser.add_argument(
        "--top_p", type=float, required=False,
        help="Top_p, 0=<top_p<=1"
    )
    parser.add_argument(
        "--top_k", type=int, required=False,
        help="Top_k, 0=<top_k<=1",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=2048, required=False,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=1, required=False,
        help="Maximum batch size",
    )    
    parser.add_argument(
        "--repetition_penalty", type=float, default=(1.0/0.85), required=False,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--api_server", type=str, required=False,
        help="Address of the API server"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, required=False,
        help="ID of the GPU to be used"
    )

    
    return parser.parse_args()

def get_parameter(parameter_name, parameter_type, default_value, args, job_data, local_rank):
    parameter = default_value
    if local_rank == 0:
        if getattr(args, parameter_name) is not None:
            parameter = getattr(args, parameter_name)
        elif parameter_type(job_data[parameter_name]) is not None:
            parameter = parameter_type(job_data[parameter_name]) 
    parameter_list = [parameter]
    torch.distributed.broadcast_object_list(parameter_list, 0)
    return parameter_list[0]



class ProcessOutputCallback():
    def __init__(self, local_rank, api_worker):
        self.local_rank = local_rank
        self.api_worker = api_worker
        self.job_data = None


    def process_output(self, output, num_generated_tokens, finished):
        if self.local_rank == 0:
            results = {'text': output}
            if finished:
                self.job_data['num_generated_tokens'] = num_generated_tokens
                return self.api_worker.send_job_results(self.job_data, results)
            else:#if self.api_worker.progress_data_received:
                return self.api_worker.send_progress(self.job_data, num_generated_tokens, results)


class ProcessOutputToShellCallback():
    def __init__(self):
        self.ctx = ""

    def process_output(self, output, final):
        if final:
            self.ctx = output
            print(f'\n{output}')


if __name__ == "__main__":
    main()
