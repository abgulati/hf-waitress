from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, BitsAndBytesConfig
from huggingface_hub import login
import torch

import subprocess
import threading
import traceback
import argparse
import logging
import queue
import time
import json
import uuid
import sys
import os
import io

from functools import wraps
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, Response


app = Flask(__name__)

PIPE = None
BUSY = False
TIMEOUT = 300   # 5-min timeout


#########################------------Threading & Request-Concurrency Queueing-------------###############################

request_queue = queue.Queue(maxsize=11)

def queue_request(func):
    @wraps(func)
    def wrapper():
        
        if request_queue.full():
            return jsonify(success=False, error="Server is busy, retry after a while"), 503
        request_queue.put((func))
        
        start_time = time.time()
        while True:
            if not BUSY:
                try:
                    result = func()
                finally:
                    break
            else:
                if time.time() - start_time < TIMEOUT:
                    time.sleep(1)
                else:
                    request_queue.get()  # Remove this request from the queue
                    request_queue.task_done()
                    return jsonify(success=False, error="Request timed out"), 504

        request_queue.get()
        request_queue.task_done()
        return result
    
    return wrapper

#########################################################################################################################


#########################------------Setup & Handle Logging-------------###############################
try:
    # 1 - Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.ERROR)

    # 2 - Create a RotatingFileHandler
    # maxBytes: max file size of log file after which a new file is created; set to 1024 * 1024 * 5 for 5MB: 1024x1024 is 1MB, then a multiplyer for the number of MB
    # backupCount: number of backup files to keep specifying how many old log files to keep
    handler = RotatingFileHandler('hf_server_log.log', maxBytes=1024*1024*5, backupCount=2)
    handler.setLevel(logging.ERROR)

    # 3 - Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)

    # 4 - Add the handler to the logger
    logger.addHandler(handler)
    # Logger ready! Usage: logger.error(f"This is an error message with error {e}")
except Exception as e:
    print(f"\n\nCould not establish logger, encountered error: {e}")


def handle_api_error(message, exception=None):
    global BUSY
    BUSY = True
    error_message = f"\n\n{message} {str(exception) if exception else '; No exception info.'}\n\n"
    traceback_details = traceback.format_exc()
    full_message = f"\n\n{error_message}\n\nTraceback: {traceback_details}\n\n"

    if logger:
        logger.error(full_message)
        print(error_message)
    else:
        print(error_message)
    
    BUSY = False
    return jsonify(success=False, error=error_message), 500 #internal server error


def handle_local_error(message, exception=None):
    global BUSY
    BUSY = True
    error_message = f"\n\n{message} {str(exception) if exception else '; No exception info.'}\n\n"
    traceback_details = traceback.format_exc()
    full_message = f"\n\n{error_message}\n\nTraceback: {traceback_details}\n\n"

    if logger:
        logger.error(full_message)
        print(error_message)
    else:
        print(error_message)
    
    BUSY = False
    raise Exception(exception)


def handle_error_no_return(message, exception=None):
    error_message = f"\n\n{message} {str(exception) if exception else '; No exception info.'}\n\n"
    traceback_details = traceback.format_exc()
    full_message = f"\n\n{error_message}\n\nTraceback: {traceback_details}\n\n"

    if logger:
        logger.error(full_message)
        print(error_message)
    else:
        print(error_message)

############################----------------------------------------------###############################



############################------------configuration manager-------------###############################

if not os.path.exists('hf_config.json'):
    try:
        with open('hf_config.json', 'w') as file:
            json.dump({}, file)
    except Exception as e:
        handle_error_no_return("Could not init config.json. Multiple app restarts may be required to get the app to init correctly. Printing error and proceeding: ", e)


# Method to write to hf_config.json | input- dict of key:values to be written to hf_config.json
def write_config(config_updates, filename='hf_config.json'):

    # Open hf_config file to read-in all current params:
    try:
        with open(filename, 'r') as file:
            hf_config = json.load(file)
    except Exception as e:
        hf_config = {}     #init emply hf_config dict
        handle_error_no_return("Could not read hf_config.json when attempting to write, encountered error: ", e)

    #restart logic in write_config() might be unnecessary, circle back later
    restart_required = False
    triggers_for_hf_restart = ['torch_device_map', 'torch_dtype', 'model_id', 'trust_remote_code', 'attn_implementation', 'pipeline_task', 'max_new_tokens', 'return_full_text', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p', 'n_keep']
    for key in config_updates:
        if key in triggers_for_hf_restart and config_updates[key] != hf_config.get(key):
            try:
                #server_status = status()    #TO-DO
                server_status = {'is_online': True}
                if server_status['is_online']:
                    restart_required = True
            except Exception as e:
                handle_error_no_return("Could not query hf_server status() for write_config, encountered error: ", e)

    hf_config.update(config_updates)

    # Write updated hf_config.json:
    try:
        with open(filename, 'w') as file:
            json.dump(hf_config, file, indent=4)
    except Exception as e:
        handle_local_error("Could not update hf_config.json, encountered error: ", e)
     
    return {'success': True, 'restart_required':restart_required}
            

# Method to read from hf_config.json | input- list of keys to be read from hf_config.json; output- dict of key:value pairs; MANAGE DEFAULTS HERE!
def read_config(keys, default_value=None, filename='hf_config.json'):
    
    # Open hf_config file to read-in all current params:
    try:
        with open(filename, 'r') as file:
            hf_config = json.load(file)
    except Exception as e:
        handle_error_no_return("Could not read hf_config.json, encountered error: ", e)
        return {key: default_value for key in keys}     #because a read scenario wherein hf_config.json does not exist shouldn't occur!
    
    return_dict = {}
    update_config_dict = {}

    for key in keys:
        if key in hf_config:
            return_dict[key] = hf_config[key]
        else:
            default_value = {
                'access_gated':False,
                'access_token':"",
                'model_id':"microsoft/Phi-3-mini-4k-instruct",
                'quantize':"bitsandbytes",
                'quant_level':"int8",
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
            }.get(key, 'undefined')

            if default_value == 'undefined':
                raise KeyError(f"Key \'{key}\' not found in hf_config.json and no default value has been defined either.\n")
            
            return_dict[key] = default_value
            update_config_dict[key] = default_value
    
    if update_config_dict:
        # Write defaults
        try:
            write_config(update_config_dict)
        except Exception as e:
            handle_error_no_return("Could not write defaults to hf_config.json. Encountered error: ", e)

    return return_dict


# Method for API route to read from hf_config.json
# Deviates from typical RESTful principals to use a POST call to fetch values but practical & justifyable because we:
# 1. Do not want to make the URL huge with a ever-growing list of query-params 2. Do not wish to expose values via query-params
@app.route('/hf_config_reader_api', methods=['POST'])
@queue_request
def hf_config_reader_api():
    # keys = request.args.getlist('keys') # Assuming keys are passed as query parameters

    global BUSY
    BUSY = True
    
    try:
        keys = request.json.get('keys', []) # Could also do keys = request.json['keys'] but this way we can provide a default list should 'keys' be missing!
    except Exception as e:
        handle_api_error("Server-side error - could not read keys for hf_config_reader_api request. Encountered error:", e)

    try:
        values = read_config(keys)  # send list of keys, get dict of key:values
    except Exception as e:
        handle_api_error("Server-side error - could not read keys from hf_config.json. Encountered error: ", e)
    
    BUSY = False
    return jsonify(success=True, values=values)


# Method for API route to write to hf_config.json
@app.route('/hf_config_writer_api', methods=['POST'])
@queue_request
def hf_config_writer_api():

    global BUSY
    BUSY = True

    try:
        config_updates = request.json['config_updates']
        print(f"config_updates for hf_config_writer_api: {config_updates}")
    except Exception as e:
        handle_api_error("Server-side error - could not read values for hf_config_writer_api request. Encountered error: ", e)
    
    try:
        write_return = write_config(config_updates)
    except Exception as e:
        handle_api_error("Server-side error - could not write keys to hf_config.json. Encountered error: ", e)
    
    BUSY = False
    return jsonify({"success": write_return['success'], "restart_required": write_return['restart_required']})


############################----------------------------------------------###############################



def safe_int(value, default):
    if value is None:
        handle_error_no_return("Null value, cannot convert to integer type. Proceeding with default value.")
        return default
    try:
        return int(value)
    except(ValueError, TypeError) as e:
        handle_error_no_return(f"Could not convert {value} to an integer, proceeding with default value {default}. Encountered error: ", e)
        return default


def safe_float(value, default):
    if value is None:
        handle_error_no_return("Null value, cannot convert to float type. Proceeding with default value.")
        return default
    try:
        return float(value)
    except(ValueError, TypeError) as e:
        handle_error_no_return(f"Could not convert {value} to a float, proceeding with default value {default}. Encountered error: ", e)
        return default


def hf_login_for_gated_models():
    access_token = ""
    try:
        read_return = read_config(['access_token'])
        access_token = str(read_return['access_token'])
    except Exception as e:
        handle_api_error("403 - No access token found, please submit an access token via the /hf_login endpoint")

    try:
        login(token=access_token)
    except Exception as e:
        handle_api_error("Unable to login to the HuggingFace-Hub, please ensure the correct access token has been provided. Encountered error: ", e)


def parse_arguments():

    try:
        parser = argparse.ArgumentParser(description="Server for HuggingFace Transformers models")
    except Exception as e:
        handle_local_error("Could not create parser to parse_arguments(), proceeding with defaults. Encountered error: ", e)

    # Even if a parser object could not be created, a read_request will write & return defaults 
    try:
        read_return = read_config(['access_gated', 'access_token', 'model_id', 'quantize', 'quant_level', 'push_to_hub', 'torch_device_map', 'torch_dtype', 'trust_remote_code', 'use_flash_attention_2', 'pipeline_task', 'max_new_tokens', 'return_full_text', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p', 'n_keep'])
        access_gated = str(read_return['access_gated']).lower() == 'true'
        access_token = str(read_return['access_token'])
        model_id = str(read_return['model_id'])
        quantize = str(read_return['quantize'])
        quant_level = str(read_return['quant_level'])
        push_to_hub = str(read_return['push_to_hub']).lower() == 'true'
        torch_device_map = str(read_return['torch_device_map'])
        torch_dtype = str(read_return['torch_dtype'])
        trust_remote_code = str(read_return['trust_remote_code']).lower() == 'true'
        use_flash_attention_2 = str(read_return['use_flash_attention_2']).lower() == 'true'
        pipeline_task = str(read_return['pipeline_task'])
        max_new_tokens = int(read_return['max_new_tokens'])
        return_full_text = str(read_return['return_full_text']).lower() == 'true'
        temperature = float(read_return['temperature'])
        do_sample = str(read_return['do_sample']).lower() == 'true'
        top_k = int(read_return['top_k'])
        top_p = float(read_return['top_p'])
        min_p = float(read_return['min_p'])
        n_keep = int(read_return['n_keep'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to parse_arguments(), encountered error: ", e)

    if parser:

        parser.add_argument("--reset_to_defaults", action="store_true", default=False, help="Use default settings")
        parser.add_argument("--access_gated", action="store_true", default=access_gated, help="Specify True if you will be accessing gated models you've been approved to access")
        parser.add_argument("--access_token", type=str, default=access_token, help="Access Token obtained from HF-Settings -> Access Tokens")
        parser.add_argument("--model_id", type=str, default=model_id, help="model_id for for LLM in HF-Transformers format obtained from the model card. Remembers previously set value and falls-back to Phi3-mini-4k-instruct as the default.")
        parser.add_argument("--quantize", type=str, default=quantize, help="Quantization method to be utilized. Simply type 'n' to not use quantization. Remembers previously set value and falls-back to bitsandbytes as the default.")
        parser.add_argument("--quant_level", type=str, default=quant_level, help="Specify quantization level, for example int8, int4, etc. Remembers previously set value and falls-back to int8 as the default.")
        parser.add_argument("--push_to_hub", action="store_true", default=push_to_hub, help="Push quantized LLM to your HF-hub. Remembers previously set value and falls-back to False as the default.")
        parser.add_argument("--torch_device_map", type=str, default=torch_device_map, help="Specify inference device, example: cuda. Remembers previously set value and falls-back to auto as the default.")
        parser.add_argument("--torch_dtype", type=str, default=torch_dtype, help="Specify model tensor type, example: bfloat16. Remembers previously set value and falls-back to auto as the default.")
        parser.add_argument("-trust_remote_code", action="store_true", default=trust_remote_code, help="Allows the model to execute custom code that's part of the model's HF-repository. Remembers previously set value and falls-back to False by default as a security measure to prevent potentially malicious code from running automatically.")
        parser.add_argument("--use_flash_attention_2", action="store_true", default=use_flash_attention_2, help="Set to True to attempt using Flash Attention 2. Remembers previously set value and falls-back to False as a default. Failed attempt to use FA2 will proceed to load the model without FA2.")
        parser.add_argument("--pipeline_task", type=str, default=pipeline_task, help="Defaults to text-generation. For more details, open a Python shell, `import transformers`, and Run `help(transfomers.pipeline)`.")
        parser.add_argument("--max_new_tokens", type=int, default=max_new_tokens, help="Set a hard limit on the maximum number of tokens an LLM can generate when responding. Remembers previously set value and falls-back to 500 as a default.")
        parser.add_argument("--return_full_text", action="store_true", default=return_full_text, help="When set to True, the LLM response contains the entire messages list with the latest response appended at the end.")
        parser.add_argument("--temperature", type=float, default=temperature, help="Set LLM temperature on a scale of 0.0 to 2.0. Remembers previously set value and falls-back to 0.0 as a default.")
        parser.add_argument("--do_sample", action="store_true", default=do_sample, help="Perform sampling when selecting response tokens. Remembers previously set value and falls-back to Flase as a default. Must be set to True when temperature is above 0.0. For greedy decoding, leave this as False and set temp to 0.0")
        parser.add_argument("--top_k", type=int, default=top_k, help="Limit the next token selection to the K most probable tokens. Remembers previously set value and falls-back to 40 as a default.")
        parser.add_argument("--top_p", type=float, default=top_p, help="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. Remembers previously set value and falls-back to 0.95 as a default.")
        parser.add_argument("--min_p", type=float, default=min_p, help="The minimum probability for a token to be considered, relative to the probability of the most likely token. Remembers previously set value and falls-back to 0.05 as a default.")
        parser.add_argument("--n_keep", type=int, default=n_keep, help="Specify the number of tokens from the prompt to retain when the context size is exceeded and tokens need to be discarded. Remembers previously set value and falls-back to 0 as a default, meaning no tokens are kept. Use -1 to retain all tokens from the prompt.")

        args = parser.parse_args()
        print(f"\n\nparser.parse_args():\n\n{args}\n\n")

        if args.reset_to_defaults:

            try:
                # Empty hf_config.json
                with open('hf_config.json', 'w') as file:
                    json.dump({}, file, indent=4)
                
                # Set defaults
                read_config(['access_gated', 'access_token', 'model_id', 'quantize', 'quant_level', 'push_to_hub', 'torch_device_map', 'torch_dtype', 'trust_remote_code', 'use_flash_attention_2', 'pipeline_task', 'max_new_tokens', 'return_full_text', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p', 'n_keep'])

            except Exception as e:
                handle_local_error("Could not reset hf_config.json, encountered error: ", e)
        else:
            try:
                write_config({
                    'access_gated':args.access_gated,
                    'access_token':args.access_token,
                    'model_id':args.model_id,
                    'quantize':args.quantize,
                    'quant_level':args.quant_level,
                    'push_to_hub':args.push_to_hub, 
                    'torch_device_map':args.torch_device_map, 
                    'torch_dtype':args.torch_dtype, 
                    'trust_remote_code':args.trust_remote_code, 
                    'use_flash_attention_2':args.use_flash_attention_2, 
                    'pipeline_task':args.pipeline_task, 
                    'max_new_tokens':args.max_new_tokens, 
                    'return_full_text':args.return_full_text, 
                    'temperature':args.temperature,
                    'do_sample':args.do_sample, 
                    'top_k':args.top_k, 
                    'top_p':args.top_p, 
                    'min_p':args.min_p, 
                    'n_keep':args.n_keep
                })
            except Exception as e:
                handle_local_error("Could not write launch arguments to hf_config.json, encountered error: ", e)

            if args.access_gated:
                try:
                    hf_login_for_gated_models()
                except Exception as e:
                    handle_local_error("Login to HF-Hub unsuccessful, encountered error: ", e)

    #return parser.parse_args()
    return True


def initialize_model():

    global PIPE

    try:
        read_return = read_config(['model_id', 'quantize', 'quant_level', 'push_to_hub', 'torch_device_map', 'torch_dtype', 'trust_remote_code', 'use_flash_attention_2', 'pipeline_task'])
        model_id = str(read_return['model_id'])
        quantize = str(read_return['quantize'])
        quant_level = str(read_return['quant_level'])
        push_to_hub = str(read_return['push_to_hub']).lower() == 'true'
        torch_device_map = str(read_return['torch_device_map'])
        torch_dtype = str(read_return['torch_dtype'])
        trust_remote_code = str(read_return['trust_remote_code']).lower() == 'true'
        use_flash_attention_2 = str(read_return['use_flash_attention_2']).lower() == 'true'
        pipeline_task = str(read_return['pipeline_task'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to parse_arguments(), encountered error: ", e)

    model_params = {
        "device_map": torch_device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }

    if use_flash_attention_2:
        model_params["attn_implementation"] = "flash_attention_2"

    quantize = quantize.lower().strip()
    if quantize != "n":

        if quantize == "bitsandbytes":
            print("Quantizing with BitsAndBytes")
            quant_level = quant_level.lower().strip()

            try:
                if quant_level == "int8":
                    print("Proceeding with BitsAndBytes-Int8 Quant")
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    model_params["quantization_config"] = quantization_config
                elif quant_level == "int4":
                    print("Proceeding with BitsAndBytes-Int4 Quant")
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                    model_params["quantization_config"] = quantization_config
            except Exception as e:
                handle_local_error("Could not set BitsAndBytes config to initialize_model(), encountered error: ", e)

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_params)
    except Exception as e:
        handle_local_error("Could not create AutoModelForCausalLM, encountered error: ", e)

    try:
        print(f"Your model's memory footprint is: {model.get_memory_footprint()}")
    except Exception as e:
        handle_error_no_return("Could not determine the model's memory footprint, encountered error: ", e)

    try:
        if push_to_hub:
            if quant_level == "int8":
                model.push_to_hub(model_id + "-Int8")
            elif quant_level == "int4":
                model.push_to_hub(model_id + "-Int4")
    except Exception as e:
        handle_error_no_return("Could not push the model to your hub, encountered error: ", e)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        handle_local_error("Could not set AutoTokenizer, encountered error: ", e)

    try:
        PIPE = pipeline(
            pipeline_task,
            model=model,
            tokenizer=tokenizer,
        )
    except Exception as e:
        handle_local_error("Could not create model PIPELINE, encountered error: ", e)

    return True


@app.route('/completions', methods=['POST'])
@queue_request
def completions():

    global BUSY
    BUSY = True

    try:
        data = request.json
        messages = data.get('messages', [])
    except Exception as e:
        handle_api_error("Could not read POST-request messages for /completions, encountered error: ", e)

    try:
        read_return = read_config(['max_new_tokens', 'return_full_text', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p', 'n_keep'])
        max_new_tokens = int(read_return['max_new_tokens'])
        return_full_text = str(read_return['return_full_text']).lower() == 'true'
        temperature = float(read_return['temperature'])
        do_sample = str(read_return['do_sample']).lower() == 'true'
        top_k = int(read_return['top_k'])
        top_p = float(read_return['top_p'])
        min_p = float(read_return['min_p'])
        n_keep = int(read_return['n_keep'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to parse_arguments(), encountered error: ", e)

    try:
        generation_args = {
            "max_new_tokens": int(request.headers.get('X-Max-New-Tokens', str(max_new_tokens))),
            "return_full_text": request.headers.get('X-Return-Full-Text', str(return_full_text)).lower() == 'true',
            "temperature": float(request.headers.get('X-Temperature', str(temperature))),
            "do_sample": request.headers.get('X-Do-Sample', str(do_sample)).lower() == 'true',
            "top_k": int(request.headers.get('X-Top-K', str(top_k))),
            "top_p": float(request.headers.get('X-Top-P', str(top_p))),
            "min_p": float(request.headers.get('X-Min-P', str(min_p)))
        }
    except Exception as e:
        handle_error_no_return("Could not set generation-arguments for /completions, proceeding without them. Encountered error: ", e)

    try:
        if generation_args:
            output = PIPE(messages, **generation_args)
        else:
            output = PIPE(messages)
    except Exception as e:
        handle_api_error("Could not generate output, encountered error: ", e)

    BUSY = False
    return jsonify({"success": True, "response": output})



class CustomStream(io.StringIO):
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback

    def write(self, data):
        # If we have a callback, call it
        if self.callback:
            self.callback(data)

        return super().write(data)


@app.route('/completions_stream', methods=['POST'])
@queue_request
def completions_stream():

    global BUSY
    BUSY = True

    print("completions_stream route triggered")

    try:
        data = request.json
        messages = data.get('messages', [])
    except Exception as e:
        handle_api_error("Could not read POST-request messages for /completions_stream, encountered error: ", e)

    try:
        read_return = read_config(['max_new_tokens', 'return_full_text', 'temperature', 'do_sample', 'top_k', 'top_p', 'min_p', 'n_keep'])
        max_new_tokens = int(read_return['max_new_tokens'])
        return_full_text = str(read_return['return_full_text']).lower() == 'true'
        temperature = float(read_return['temperature'])
        do_sample = str(read_return['do_sample']).lower() == 'true'
        top_k = int(read_return['top_k'])
        top_p = float(read_return['top_p'])
        min_p = float(read_return['min_p'])
        n_keep = int(read_return['n_keep'])
    except Exception as e:
        handle_local_error("Could not read values from hf_config.json when trying to parse_arguments(), encountered error: ", e)

    try:
        generation_args = {
            "max_new_tokens": int(request.headers.get('X-Max-New-Tokens', str(max_new_tokens))),
            "return_full_text": request.headers.get('X-Return-Full-Text', str(return_full_text)).lower() == 'true',
            "temperature": float(request.headers.get('X-Temperature', str(temperature))),
            "do_sample": request.headers.get('X-Do-Sample', str(do_sample)).lower() == 'true',
            "top_k": int(request.headers.get('X-Top-K', str(top_k))),
            "top_p": float(request.headers.get('X-Top-P', str(top_p))),
            "min_p": float(request.headers.get('X-Min-P', str(min_p)))
        }
    except Exception as e:
        handle_error_no_return("Could not set generation-arguments for /completions_stream, proceeding without them. Encountered error: ", e)


    stop_thread = threading.Event()

    def generate():

        data_queue = queue.Queue()

        def callback(data):
            data_queue.put(data)

        custom_stream = CustomStream(callback=callback)

        original_stdout = sys.stdout
        sys.stdout = custom_stream

        def llm_task():

            global PIPE

            try:
                streamer = TextStreamer(PIPE.tokenizer, skip_special_tokens=True)
            except Exception as e:
                handle_api_error("Could not create TextStreamer for completions_stream. Try using the /completions endpoint instead. Encountered error: ", e)
            
            try:
                if generation_args:
                    generation_args["streamer"] = streamer
                    output = PIPE(messages, **generation_args)
                else:
                    output = PIPE(messages, streamer=streamer)
            except Exception as e:
                handle_api_error("Could not generate output, encountered error: ", e)
            finally:
                sys.stdout = original_stdout

                data_queue.put(None)
                stop_thread.set()
        
        thread = threading.Thread(target=llm_task)
        thread.start()

        i = 0
        while True:
            line = data_queue.get()
            if line is None:
                print("None read, breaking and stopping thread")
                thread.join()
                break
            if i == 0:
                line = line.strip('\n')
                i += 1
            yield f"data: {line}\n\n"
        
        yield "event: END\ndata: null\n\n"

        print("LLM stream done")
        global BUSY
        BUSY = False

    print("\n\nInferencing Begins!\n\n")
    return Response(generate(), content_type='text/event-stream')


@app.route('/health')
@queue_request
def health():

    global BUSY
    BUSY = True

    try:
        if PIPE is None:
            BUSY = False
            return jsonify(status="error", message="Model not loaded"), 503 # Service Unavailable
        
        model_info = {}

        # print(f"\n\nmodel details: {PIPE.model}\n\n")
        # print(f"\n\nmodel.config details: {PIPE.model.config}\n\n")
        # print(f"\n\ntokenizer details: {PIPE.tokenizer}\n\n")
        
        try:
            model_info["model_id"] = PIPE.model.config._name_or_path
        except Exception as e:
            handle_error_no_return(f"Could not determine model_id, encountered error: {e}")

        try:
            model_info["transformers_version"] = PIPE.model.config.transformers_version
        except Exception as e:
            handle_error_no_return(f"Could not determine transformers_version, encountered error: {e}")

        try:
            model_info["architecture"] = PIPE.model.config.architectures
        except Exception as e:
            handle_error_no_return(f"Could not determine model architecture, encountered error: {e}")

        try:
            model_info["model_type"] = PIPE.model.config.model_type
        except Exception as e:
            handle_error_no_return(f"Could not determine model_type, encountered error: {e}")

        try:
            model_info["torch_dtype"] = PIPE.model.config.torch_dtype
        except Exception as e:
            handle_error_no_return(f"Could not determine torch_dtype, encountered error: {e}")

        try:
            model_info["device"] = str(PIPE.device)
        except Exception as e:
            handle_error_no_return(f"Could not determine inference device, encountered error: {e}")

        try:
            if hasattr(PIPE.model.config, "quantization_config"):
                model_info["is_quantized"] = True
                model_info["quant_method"] = PIPE.model.config.quantization_config.quant_method
                model_info["quantization_config"] = PIPE.model.config.quantization_config
            else:
                model_info["is_quantized"] = False
        except Exception as e:
            handle_error_no_return(f"Could not determine quantization status, encountered error: {e}")
        
        try:
            model_info["model_vocab_size"] = PIPE.model.config.vocab_size
        except Exception as e:
            handle_error_no_return(f"Could not determine model_vocab_size, attempting to check length of the pipeline-tokenizer, encountered error: {e}")
            try:
                model_info["tokenizer_vocab_length"] = len(PIPE.tokenizer)
            except Exception as e:
                handle_error_no_return(f"Could not determine length of the pipeline-tokenizer! Encountered error: {e}")
        
        try:
            model_info["tokenizer_vocab_size"] = PIPE.tokenizer.vocab_size
        except Exception as e:
            handle_error_no_return(f"Could not determine tokenizer_vocab_size, encountered error: {e}")
        
        try:
            model_info["number_of_hidden_layers"] = PIPE.model.config.num_hidden_layers
        except Exception as e:
            handle_error_no_return(f"Could not determine number_of_hidden_layers, encountered error: {e}")
        
        try:
            model_info["number_of_attention_heads"] = PIPE.model.config.num_attention_heads
        except Exception as e:
            handle_error_no_return(f"Could not determine number_of_attention_heads, encountered error: {e}")

        try:
            model_info["hidden_dimensions"] = PIPE.model.config.head_dim
        except Exception as e:
            handle_error_no_return(f"Could not determine hidden_dimensions, encountered error: {e}")

        try:
            model_info["number_of_key_value_heads"] = PIPE.model.config.num_key_value_heads
        except Exception as e:
            handle_error_no_return(f"Could not determine number_of_key_value_heads, encountered error: {e}")
        
        try:
            model_info["hidden_activation"] = PIPE.model.config.hidden_act
        except Exception as e:
            handle_error_no_return(f"Could not determine hidden_act, encountered error: {e}")
        
        try:
            model_info["hidden_size"] = PIPE.model.config.hidden_size
        except Exception as e:
            handle_error_no_return(f"Could not determine hidden_size, encountered error: {e}")

        try:
            model_info["intermediate_size"] = PIPE.model.config.intermediate_size
        except Exception as e:
            handle_error_no_return(f"Could not determine intermediate_size, encountered error: {e}")

        try:
            model_info["max_position_embeddings"] = PIPE.model.config.max_position_embeddings
        except Exception as e:
            handle_error_no_return(f"Could not determine max_position_embeddings, encountered error: {e}")

        try:
            model_info["tokenizer"] = PIPE.tokenizer.name_or_path
        except Exception as e:
            handle_error_no_return(f"Could not determine the tokenizer used, encountered error: {e}")

        try:
            model_info["max_seq_length"] = PIPE.tokenizer.model_max_length
        except Exception as e:
            handle_error_no_return(f"Could not determine the sequence length of the model's tokenizer, encountered error: {e}")

        BUSY = False
        return jsonify(status="ok", model_info=model_info), 200

    except Exception as e:
        handle_api_error(f"Error checking hf-server health, encountered error: {e}")



def main():
    parse_arguments()
    initialize_model()
    app.run(host='0.0.0.0', port=9069)


if __name__ == '__main__':
    main()