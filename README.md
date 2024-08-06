# HF-Waitress

HF-Waitress is a powerful and flexible server application for deploying and interacting with HuggingFace Transformer models. It simplifies the process of running open-source Large Language Models (LLMs) locally on-device, addressing common pain points in model deployment and usage.

This server enables loading HF-Transformer & AWQ-quantized models directly off the hub, while providing on-the-fly quantization via BitsAndBytes, HQQ and Quanto for the former. It negates the need to manually download any model yourself, simply working off the models name instead. It requires no setup, and provides concurrency and streaming responses all from within a single, easily-portable, platform-agnostic Python script.

## Key Features

- **On-the-fly, in-place quantization**: Supports int8 & int4 quantization via BitsAndBytes, Quanto and HQQ.
- **Model Agnosticism**: Compatible with any HF-Transformers format LLM.
- **Configuration Management**: Uses `config.json` to store settings, allowing for easy configuration and persistence across runs.
- **Error Handling**: Detailed logging and traceback reporting via centralized error-handling functions.
- **Health Endpoint**: Provides valuable information about the loaded model and server health.
- **Concurrency Control**: Uses semaphores for selective concurrency while taking advantage of semaphore-native queueing.
- **Streaming Responses**: Supports both standard and streaming completions.

## Dependencies

1. Python v3.10.x or above

2. PyTorch:

    **If you're planning to use your GPU to run LLMs, make sure to install the GPU drivers and CUDA/ROCm toolkits as appropriate for your setup, and only then proceed with PyTorch setup below**

    Download and install the PyTorch version appropriate for your system: https://pytorch.org/get-started/locally/

3. (optional) If attempting to use Flash Attention 2, specific Nvidia GPUs are required. Check the [official-repo](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) for requirements and installation instructions.

## Installation

1. Clone this repository:

    ```
    git clone https://github.com/abgulati/hf-waitress
    cd hf-waitress
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage

To start the server, run: `python hf_waitress.py [arguments]`

Example:

```
python hf_waitress.py --model_id=mistralai/Mistral-Nemo-Instruct-2407 --quantize=quanto --quant_level=int4 --access_token=<token> --trust_remote_code --use_flash_attention_2 --do_sample
```
*launch-arguments are optional though, even on the first run! See below for first-run defaults.*

### Command-line Arguments

- `--model_id`: The model ID in HF-Transformers format - see below for details.
- `--access_gated`: Set to True if accessing gated models you're approved for.
- `--access_token`: Your Hugging Face Access Token.
- `--gguf`: Add this flag if attempting to load a GGUF model - [For future use, not presently functional](https://huggingface.co/docs/transformers/main/en/gguf)
- `--gguf_model_id`: GGUF repository ID - [For future use, not presently functional](https://huggingface.co/docs/transformers/main/en/gguf)
- `--gguf_filename`: Specific GGUF filename - [For future use, not presently functional](https://huggingface.co/docs/transformers/main/en/gguf)
- `--quantize`: Quantization method ('bitsandbytes', 'quanto', 'hqq' or 'n' for none, see important details below.).
- `--quant_level`: Quantization level (Valid values -  BitsAndBytes: int8 & int4; Quanto: int8, int4 and int2; HQQ: int8, int4, int3, int2, int1).
- `--hqq_group_size`: Specify group_size (default: 64) for HQQ quantization. No restrictions as long as weight.numel() is divisible by the group_size.
- `--push_to_hub`: Push quantized model to Hugging Face Hub.
- `--torch_device_map`: Specify inference device (e.g., 'cuda', 'cpu').
- `--torch_dtype`: Specify model tensor type.
- `--trust_remote_code`: Allow execution of custom code from the model's repository.
- `--use_flash_attention_2`: Attempt to use Flash Attention 2 - [Only for specific Nvidia GPUs](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) 
- `--pipeline_task`: Specify the pipeline task (default: 'text-generation').
- `--max_new_tokens`: Maximum number of tokens to generate.
- `--return_full_text`: Return the full text including the prompt.
- `--temperature`: Set LLM temperature (0.0 to 2.0) - set do_sample to True for temps above 0.0, and False when setting temperature=0.0!
- `--do_sample`: Perform sampling when selecting response tokens - must be set to True for temps above 0.0!
- `--top_k`, `--top_p`, `--min_p`: Token selection parameters - must set do_sample to True!
- `--port`: Specify the server port (default: 9069).
- `--reset_to_defaults`: Reset all settings to default values.

### First-run Defaults (for missing launch-args)
```
{
    'access_gated':False,
    'access_token':"",
    'model_id':"microsoft/Phi-3-mini-4k-instruct",
    'gguf':False,
    'awq':False,
    'gguf_model_id':None,
    'gguf_filename':None,
    'quantize':"quanto",
    'quant_level':"int4",
    'hqq_group_size':64,
    'push_to_hub':False,
    'torch_device_map':"auto", 
    'torch_dtype':"auto", 
    'trust_remote_code':False, 
    'use_flash_attention_2':False, 
    'pipeline_task':"text-generation", 
    'max_new_tokens':500, 
    'return_full_text':False, 
    'temperature':0.0,
    'do_sample':False, 
    'top_k':40, 
    'top_p':0.95, 
    'min_p':0.05, 
    'n_keep':0,
    'port':9069
}
```

### The required `model_id` can typically be obtained one of two ways, both of which involve going to the model's HuggingFace.co page:

1. Simply make use of the copy function provided by HuggingFace:

<p align="center">
<img src="https://github.com/abgulati/hf-server/blob/main/images/hf-copy.png"  align="center">
</p>

2. Or, scroll further down the model card and copy the model_id from the code sample provided by the model creators:

<p align="center">
<img src="https://github.com/abgulati/hf-server/blob/main/images/hf-sample.png"  align="center">
</p>

### Quantizing LLMs

- Several Quantization methods are available in HF-Waitress: BitsAndBytes, Quanto and HQQ, alongside the ability to run HF-Transformer and AWQ models directly off the HF-Hub

- BitsAndBytes:
    - Requires: Nvidia CUDA-supported GPU
    - Supported Quantization Levels: int8 and int4
    - Recommended quant technique for Nvidia GPU owners as this is the best and fastest quantization method available.

- Quanto:
    - Native PyTorch Quantization technique - versatile pytorch quantization toolkit. 
    - The underlying method used is linear quantization. 
    - Supports: CPU, GPU, Apple Silicon
    - Supported Quantization Levels: int8, int4 and int2
    - NOTE: At load time, the model will report a high memory footprint but actual memory-usage will be significantly lower.

- HQQ:
    - Half-Quadratic Quantization (HQQ) implements on-the-fly quantization via fast robust optimization. It doesn’t require calibration data and can be used to quantize any model.
    - Supports: CPU, NvCUDA GPU
    - Supported Quantization Levels: int8, int4, int3, int2 and int1

- AWQ:
    - Activation-aware Weight Quantization (AWQ) doesn’t quantize all the weights in a model, and instead preserves a small percentage of weights that are important for LLM performance. 
    - This significantly reduces quantization loss such that you can run models in 4-bit precision without experiencing any performance degradation.
    - Supports: GPUs - Nvidia CUDA and AMD ROCm compliant GPUs
    - See section below for running these models

- Check the [official HF-docs](https://huggingface.co/docs/transformers/main/en/quantization/overview) for more details and hardware-support matrix.

### Loading AWQ-Quantized Models:

- There are several libraries for quantizing models with the AWQ algorithm, such as llm-awq, autoawq or optimum-intel. 
- Transformers ONLY supports loading models quantized with the llm-awq and autoawq libraries
- For models quantized with `autoawq`, install the AutoAWQ PIP package:
    ```
    pip install autoawq
    ```
- NOTE: As of this writing, AutoAWQ requires Torch 2.3.x. If you have another version of Torch already installed (such as for CUDA-12.4 etc), you can try to run the above with "--no-deps": `pip install --no-deps autoawq`. in my testing, AWQ models work fine this way, but YMMV.

- To run models on the HuggingFace-Hub in the AWQ format, simply specify the model_id and set the `--awq` flag at launch:
    ```
    python .\hf_waitress.py --awq --model_id=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
    ```
- This will auto-set `quantize=n` & `torch_dtype=torch.float16` without overwriting their values in `hf_config.json`


## API Endpoints

### Overview:

1. `/completions` (POST): Generate completions for given messages.
2. `/completions_stream` (POST): Stream completions for given messages.
3. `/health` (GET): Check the health and get information about the loaded model.
4. `/hf_config_reader_api` (POST): Read values from the configuration.
5. `/hf_config_writer_api` (POST): Write values to the configuration.
6. `/restart_server` (GET): Restart the LLM server.


### Details:

1. `/completions` (POST): Generate completions for given messages.

    - **Headers**:
    - `Content-Type: application/json`
    - `X-Max-New-Tokens`: Maximum number of tokens to generate
    - `X-Return-Full-Text`: Whether to return the full text including the prompt
    - `X-Temperature`: Temperature for text generation (0.0 to 2.0)
    - `X-Do-Sample`: Whether to use sampling for text generation
    - `X-Top-K`: Top-K sampling parameter
    - `X-Top-P`: Top-P (nucleus) sampling parameter
    - `X-Min-P`: Minimum probability for token consideration

    - **Body**: Raw JSON
    ```
    {
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
            {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
            {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}
        ]
    }
    ```

    - **Response**: JSON object containing the generated completion

2. `/completions_stream` (POST): Stream completions for given messages.

    - **Headers**: Same as /completions
    - **Body**: Same as /completions
    - **Response**: Server-Sent Events (SSE) stream of generated text

3. `/health` (GET): Check the health and get information about the loaded model.

    - **Body**: None
    - **Response**: JSON object containing model and server health information
        ```
        {
            "model_info": {
                "architecture": "['MistralForCausalLM']",
                "device": "cuda",
                "hidden_activation": "silu",
                "hidden_dimensions": "128",
                "hidden_size": "5120",
                "intermediate_size": "14336",
                "is_quantized": true,
                "max_position_embeddings": "1024000",
                "max_seq_length": "1000000000000000019884624838656",
                "memory_footprint": "8137789440",
                "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
                "model_type": "mistral",
                "model_vocab_size": "131072",
                "number_of_attention_heads": "32",
                "number_of_hidden_layers": "40",
                "number_of_key_value_heads": "8",
                "quant_method": "QuantizationMethod.BITS_AND_BYTES",
                "quantization_config": "BitsAndBytesConfig {
                        \"_load_in_4bit\": true,
                        \"_load_in_8bit\": false,
                        \"bnb_4bit_compute_dtype\": \"float32\",
                        \"bnb_4bit_quant_storage\": \"uint8\",
                        \"bnb_4bit_quant_type\": \"fp4\",
                        \"bnb_4bit_use_double_quant\": false,
                        \"llm_int8_enable_fp32_cpu_offload\": false,
                        \"llm_int8_has_fp16_weight\": false,
                        \"llm_int8_skip_modules\": null,
                        \"llm_int8_threshold\": 6.0,
                        \"load_in_4bit\": true,
                        \"load_in_8bit\": false,
                        \"quant_method\": \"bitsandbytes\"
                    }",
                "tokenizer": "mistralai/Mistral-Nemo-Instruct-2407",
                "tokenizer_vocab_size": "131072",
                "torch_dtype": "torch.bfloat16",
                "transformers_version": "4.43.0.dev0"
            },
            "status": "ok"
        }
        ```

4. `/hf_config_reader_api` (POST): Read values from the configuration.

    - **Body**: JSON object with a keys array specifying which config values to read
        ```
        {
            "keys": [
                "model_id",
                "quantize",
                "quant_level",
                "torch_device_map",
                "torch_dtype",
                "use_flash_attention_2",
                "max_new_tokens"
            ]
        }
        ```
        
    - **Response**: JSON object containing the requested configuration values
        ```
        {
            "success": true,
            "values": {
                "max_new_tokens": 2048,
                "model_id": "microsoft/Phi-3-mini-128k-instruct",
                "quant_level": "int8",
                "quantize": "bitsandbytes",
                "torch_device_map": "cuda",
                "torch_dtype": "auto",
                "use_flash_attention_2": true
            }
        }
        ```

5. `/hf_config_writer_api` (POST): Write values to the configuration.

    - **Body**: JSON object with key-value pairs to update in the configuration
        ```
        {
            "config_updates": {
                "model_id":"microsoft/Phi-3-mini-128k-instruct",
                "quant_level":"int4"
            }
        }
        ```
    - **Response**: JSON object indicating success and whether a restart is required
        ```
        {
            "restart_required": true,
            "success": true
        }
        ```

6. `/restart_server` (GET): Restart the LLM server.

    - **Body**: None
    - **Response**: JSON object indicating success or error
        ```
        {
            "success": true
        }
        ```

## Configuration

The server uses a `hf_config.json` file to store and manage configurations. You can modify this file directly or use the provided API endpoints to update settings.

## Error Handling and Logging

Errors are logged to `hf_server_log.log`. The log file uses a rotating file handler, keeping the most recent logs and discarding older ones.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.