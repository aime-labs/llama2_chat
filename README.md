# Mixtral 8x7B Chat Implementation

Inference code to run Mixtral 8x7B, a high-quality sparse mixture of experts model (SMoE) with open weights. 

More Information about the Mixtral Model: https://mistral.ai/news/mixtral-of-experts/

## Download

The Mixtral 8x7B Model is licensed under Apache 2.0 and downloadable at: https://models.mistralcdn.com/mixtral-8x7b-v0-1/Mixtral-8x7B-v0.1-Instruct.tar

The code is based on the original Llama codebase.

## Hardware Requirements

The hardware requirements to run Mixtral 8x7B model in the default fp16 resolution:

- 2x 80 GB GPUs (NVIDIA A100/H100)
- 4x 48 GB GPUs (NVIDIA RTX A6000 / 6000 Ada)
- 8x 24 GB GPUs (NVIDIA RTX 3090 / A5000)


## Setup with [AIME MLC](https://github.com/aime-team/aime-ml-containers)

Easy installation within an [AIME ML-Container](https://github.com/aime-team/aime-ml-containers).

Create an AIME ML container:
```mlc-create mycontainer Pytorch 2.1.2```
Once done open the container with:
```mlc-open mycontainer```
Navigate to the destination of the repo and install the required pip packages
```
cd mixtral_chat
pip install -r requirements.txt
```

## Start Mixtral as Command Line Chat
Run the chat mode in the command line with following command:
```
python3 chat.py --ckpt_dir <destination_of_checkpoints> --num_gpus <number of GPUs to use>
```

## Start Mixtral Chat as AIME API Worker

To run Mixtral Chat as HTTP/HTTPS API with [AIME API Server](https://github.com/aime-team/aime-api-server) start the chat command with following command line:

```
python3 chat.py --ckpt_dir <destination_of_checkpoints> --num_gpus <number of GPUs to use> --api_server <url to api server>
```
It will start Mixtral Chat as worker, waiting for job request through the AIME API Server. Use the --max_batch_size option to control how many parallel job requests can be handled (depending on the available GPU memory). 
