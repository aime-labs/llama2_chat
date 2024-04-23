# Llama 2

We are unlocking the power of large language models. Our latest version of Llama is now accessible to individuals, creators, researchers and businesses of all sizes so that they can experiment, innovate and scale their ideas responsibly. 

This release includes model weights and starting code for pretrained and fine-tuned Llama language models — ranging from 7B to 70B parameters.

This repository is intended to run LLama 2 models as worker for the [AIME API Server](https://github.com/aime-team/aime-api-server) also an interactive console chat for testing purpose is available.

Llama 2 API demo server running at: [https://api.aime.info/llama2-chat/](https://api.aime.info/llama2-chat/)

## Features

* Realtime interactive console chat example
* Llama 2 13B support for 1 GPU with at least 40GB memory (e.g. RTX A6000/6000 Ada/A100) setups
* Llama 2 70B support for 2 GPU (e.g. 2x A100/H100 80 GB) and 4 GPU (e.g. 4x A100 40GB/RTX A6000/6000 Ada) setups
* Worker mode for AIME API server to use Llama 2 as HTTP/HTTPS API endpoint
* Batch job aggreation support for AIME API server for higher GPU throughput with multi-user chat

## Quick Start

You can follow the steps below to quickly get up and running with Llama 2 models. These steps will let you run quick inference locally. For more examples, see the [Llama 2 recipes repository](https://github.com/facebookresearch/llama-recipes). 

### Setup with Conda Environment

1. In a conda env with PyTorch / CUDA available clone and download this repository.

2. In the top level directory run:
    ```bash
    pip install -e .
    ```

### Alternative setup with [AIME MLC](https://github.com/aime-team/aime-ml-containers)

Easy installation within an [AIME ML-Container](https://github.com/aime-team/aime-ml-containers).

1. Create and open an AIME ML container:

```mlc-create mycontainer Pytorch 2.0.1```
Once done open the container with:
```mlc-open mycontainer```

2. Navigate to the destination of the repo and install the required pip packages
```
cd llama2_chat
pip install -r requirements.txt
```

### Download Model

3. Visit the [Meta AI website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and register to download the model/s.

4. Once registered, you will get an email with a URL to download the models. You will need this URL when you run the download.sh script.

5. Once you get the email, navigate to your downloaded llama repository and run the download.sh script. 
    - Make sure to grant execution permissions to the download.sh script
    - During this process, you will be prompted to enter the URL from the email. 
    - Do not use the “Copy Link” option but rather make sure to manually copy the link from the email.

Pre-requisites: Make sure you have `wget` and `md5sum` installed. Then to run the script: `./download.sh`.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as `403: Forbidden`, you can always re-request a link.

#### 6. Convert 13B models for 1 GPU or 70B models for 2 or 4 GPU configuration (if required)
The default sharding configuration of the downloaded Llama 3 70B model weights is for 8 GPUs (with 24 GB memory). The weights for a 4 or 2 GPU setups can be converted with the 'convert_weights.py' script.

To do so run following command:

```
python convert_weights.py --input_dir /data/models/llama-2-70b-chat/ --model_size 70B --num_gpus <num_gpus>
```

For Llama2 13B model <num_gpus> can be:

- 1 for 1x at least 40 GB memory per GPU

For Llama2 70B model <num_gpus> can be:

- 4 for 4x at least 40 GB memory per GPU
- 2 for 2x at least 80 GB memory per GPU

#### 7a. Start Chat in Command Line
Run the chat mode in the command line with following command:
```
torchrun --nproc_per_node <num_gpus> chat.py --ckpt_dir <destination_of_checkpoints>
```

#### 7b. Start Llama2 Chat as AIME API Worker

To run Llama2 Chat as HTTP/HTTPS API with [AIME API Server](https://github.com/aime-team/aime-api-server) start the chat command with following command line:

```
torchrun --nproc_per_node <num_gpus> chat.py --ckpt_dir <destination_of_checkpoints> --api_server <url to api server>
```
It will start Llama2 as worker, waiting for job request through the AIME API Server. Use the --max_batch_size option to control how many parallel job requests can be handled (depending on the available GPU memory). 
```

**Note**
- Replace  `llama-2-7b-chat/` with the path to your checkpoint directory and `tokenizer.model` with the path to your tokenizer model.
- The `–nproc_per_node` should be set to the [MP](#inference) value for the model you are using.
- Adjust the `max_seq_len` and `max_batch_size` parameters as needed.

## Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2, 1 (for 1 the weights have to be converted with convert_weights.py first) |
| 70B    | 8, 4, 2 (for 4 and 2 the weights have to be converted with convert_weights.py first) |

All models support sequence length up to 4096 tokens, but we pre-allocate the cache according to `max_seq_len` and `max_batch_size` values. So set those according to your hardware.

### Pretrained Models

These models are not finetuned for chat or Q&A. They should be prompted so that the expected answer is the natural continuation of the prompt.

See `example_text_completion.py` for some examples. To illustrate, see the command below to run it with the llama-2-7b model (`nproc_per_node` needs to be set to the `MP` value):

```
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir llama-2-7b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

### Fine-tuned Chat Models

The fine-tuned models were trained for dialogue applications. To get the expected features and performance for them, a specific formatting defined in [`chat_completion`](https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212)
needs to be followed, including the `INST` and `<<SYS>>` tags, `BOS` and `EOS` tokens, and the whitespaces and breaklines in between (we recommend calling `strip()` on inputs to avoid double-spaces).

You can also deploy additional classifiers for filtering out inputs and outputs that are deemed unsafe. See the llama-recipes repo for [an example](https://github.com/facebookresearch/llama-recipes/blob/main/inference/inference.py) of how to add a safety checker to the inputs and outputs of your inference code.

Examples using llama-2-7b-chat:

```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```

Llama 2 is a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
In order to help developers address these risks, we have created the [Responsible Use Guide](Responsible-Use-Guide.pdf). More details can be found in our research paper as well.

## Issues

Please report any software “bug,” or other problems with the models through one of the following means:
- Reporting issues with the model: [github.com/facebookresearch/llama](http://github.com/facebookresearch/llama)
- Reporting risky content generated by the model: [developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- Reporting bugs and security concerns: [facebook.com/whitehat/info](http://facebook.com/whitehat/info)

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md).

## License

Our model and weights are licensed for both researchers and commercial entities, upholding the principles of openness. Our mission is to empower individuals, and industry through this opportunity, while fostering an environment of discovery and ethical AI advancements. 

See the [LICENSE](LICENSE) file, as well as our accompanying [Acceptable Use Policy](USE_POLICY.md)

## References

1. [Research Paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
2. [Llama 2 technical overview](https://ai.meta.com/resources/models-and-libraries/llama)
3. [Open Innovation AI Research Community](https://ai.meta.com/llama/open-innovation-ai-research-community/)

For common questions, the FAQ can be found [here](https://github.com/facebookresearch/llama/blob/main/FAQ.md) which will be kept up to date over time as new questions arise. 

## Original LLaMA
The repo for the original llama release is in the [`llama_v1`](https://github.com/facebookresearch/llama/tree/llama_v1) branch.
