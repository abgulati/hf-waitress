# HF Waitress

HF Waitress is a powerful and flexible server application for deploying and interacting with Hugging Face Transformers models. It simplifies the process of running open-source Language Models (LLMs) locally, addressing common pain points in model deployment and usage.

## Key Features

- **On-the-fly, in-place quantization**: Supports int8 & int4 quantization via BitsAndBytes.
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

3. (optional) If attempting to use Flash Attention 2, specific Nvidia GPUs are required. Check the [official-repo](https://github.com/Dao-AILab/flash-attention) for requirements and installation instructions.

## Installation

1. Clone this repository:

    ```
    git clone https://github.com/yourusername/hf-waitress.git
    cd hf-waitress
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage

To start the server, run:

```
python hf_waitress.py [arguments]
```
*launch-arguments are optional, even on the first run!*

### Command-line Arguments

- `--model_id`: The model ID in HF-Transformers format.
- `--access_gated`: Set to True if accessing gated models you're approved for.
- `--access_token`: Your Hugging Face Access Token.
- `--quantize`: Quantization method (e.g., 'bitsandbytes' or 'n' for none).
- `--quant_level`: Quantization level (e.g., 'int8', 'int4').
- `--push_to_hub`: Push quantized model to Hugging Face Hub.
- `--torch_device_map`: Specify inference device (e.g., 'cuda', 'cpu').
- `--torch_dtype`: Specify model tensor type.
- `--trust_remote_code`: Allow execution of custom code from the model's repository.
- `--use_flash_attention_2`: Attempt to use Flash Attention 2. Only for specific Nvidia GPUs, check the [official-repo](https://github.com/Dao-AILab/flash-attention) 
- `--pipeline_task`: Specify the pipeline task (default: 'text-generation').
- `--max_new_tokens`: Maximum number of tokens to generate.
- `--return_full_text`: Return the full text including the prompt.
- `--temperature`: Set LLM temperature (0.0 to 2.0).
- `--do_sample`: Perform sampling when selecting response tokens.
- `--top_k`, `--top_p`, `--min_p`: Token selection parameters.
- `--port`: Specify the server port (default: 9069).
- `--reset_to_defaults`: Reset all settings to default values.

## API Endpoints

1. `/completions` (POST): Generate completions for given messages.
2. `/completions_stream` (POST): Stream completions for given messages.
3. `/health` (GET): Check the health and get information about the loaded model.
4. `/hf_config_reader_api` (POST): Read values from the configuration.
5. `/hf_config_writer_api` (POST): Write values to the configuration.

## Configuration

The server uses a `hf_config.json` file to store and manage configurations. You can modify this file directly or use the provided API endpoints to update settings.

## Error Handling and Logging

Errors are logged to `hf_server_log.log`. The log file uses a rotating file handler, keeping the most recent logs and discarding older ones.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
