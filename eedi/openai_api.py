"""Interface for interacting with OpenAI language model API."""

from typing import List
import time
import os
import json
from tqdm import tqdm
import concurrent.futures
import openai
from openai import OpenAI, AzureOpenAI, RateLimitError, APITimeoutError, APIError, APIConnectionError
import httpx
from dotenv import load_dotenv

load_dotenv()

delay_time = 0.5
decay_rate = 0.8
max_attempts = 10

def read_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")
    return data

def apply_arg_defaults(args: dict):
    assert args["max_completion_tokens"]
    args["temperature"] = args.get("temperature", 0.0)
    args["n"] = args.get("n", 1)

class OpenAIClient:
    def __init__(self, use_azure_client: bool):
        if use_azure_client:
            openai.api_type = "azure"
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-12-01-preview",
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            )
        else:
            openai.api_type = "openai"
            client = OpenAI(
                api_key = os.getenv("OPENAI_API"),
                http_client=httpx.Client()
            )
            self.client = client

    def get_responses(self, prompts: List[str], model: str, system_message: str | List[str], generation_kwargs: dict, batch_api: bool):
        if batch_api:
            return self.get_batch_api_responses(prompts, model, system_message, generation_kwargs)
        return self.get_batched_responses(prompts, model, 5, generation_kwargs, system_message, show_progress=True)

    def get_batch_api_responses(self, prompts: List[str], model: str, system_message: str | List[str], generation_kwargs: dict):
        """
        Use batch API to generate responses
        Based on https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/batch
        """
        apply_arg_defaults(generation_kwargs)

        # Load cache
        cache_filename = f"openai_cache_{model}.json"
        if os.path.exists(cache_filename):
            with open(cache_filename) as cache_file:
                cache: dict = json.load(cache_file)
        else:
            cache = {}
        top_key = f"#ARGS-{generation_kwargs}#SYS-{system_message}#"
        sm_cache = cache.setdefault(top_key, {})
        uncached_prompts = list({prompt for prompt in prompts if prompt not in sm_cache})
        print(f"Calling batch API - {len(prompts)} prompts, sending {len(uncached_prompts)} new requests")
        batch_id = None

        if uncached_prompts:
            # Create local file with prompts and generation arguments
            lines = [
                json.dumps({
                    "custom_id": f"{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": (system_message if type(system_message) == str else system_message[idx]) or "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        **generation_kwargs
                    }
                })
                for idx, prompt in enumerate(uncached_prompts)
            ]
            with open("temp.jsonl", "w") as f:
                f.write("\n".join(lines))

            # Upload data as remote file with "batch" purpose
            file = self.client.files.create(
                file=open("temp.jsonl", "rb"),
                purpose="batch",
                # From docs: "There's a max limit of 500 batch files per resource when no expiration is set.
                # By setting a value for expiration the number of batch files per resource is increased to 10,000 files per resource."
                extra_body={"expires_after": {"seconds": 1209600, "anchor": "created_at"}} # 14 days
            )
            print(file.model_dump_json(indent=2))
            file_id = file.id

            # Submit a batch job with the file
            batch_response = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h", # 24h is only valid value here
                extra_body={"output_expires_after": {"seconds": 1209600, "anchor": "created_at"}} # 14 days
            )
            print(batch_response.model_dump_json(indent=2))
            batch_id = batch_response.id

            # Wait for job to complete, poll every 60 seconds
            # with tqdm(desc="Running batch job", total=None) as pbar:
            #     num_polls = 0
            #     while True:
            #         batch_response = self.client.batches.retrieve(batch_id)
            #         status = batch_response.status
            #         num_polls += 1
            #         # Update the progress bar description with current status and poll count
            #         pbar.set_postfix({"status": status, "polls": num_polls})
            #         pbar.update(1)
            #         if status in ("completed", "failed", "canceled"):
            #             break
            #         time.sleep(60) # Sleep between polls

            # Report errors if failed
            # if batch_response.status == "failed":
            #     for error in batch_response.errors.data:
            #         print(f"Error code {error.code} Message {error.message}")
            #     raise Exception("Batch job failed")

        return sm_cache, uncached_prompts, batch_id


    def retrieve_batch_api_responses(self, batch_response, prompts, uncached_prompts, sm_cache, generation_kwargs):
        # Grab results from file
        output_file_id = batch_response.output_file_id
        if not output_file_id:
            raise Exception("No output file in batch response")
        
        try:
            file_response = self.client.files.content(output_file_id)
            raw_responses = file_response.text.strip().split("\n")
        except Exception as e:
            print("Error encountered retrieving file, will attempt to read local batch output file instead")
            batch_id = batch_response.id
            raw_responses = [json.dumps(x) for x in read_jsonl(f"batches/{batch_id}_output.jsonl")]
        print(f"{len(raw_responses)} / {len(uncached_prompts)} prompts succeeded")

        # Set results by identifier, fall back on default value (empty list or string) since some prompts can fail unexpectedly
        results = [("" if generation_kwargs["n"] == 1 else []) for _ in range(len(uncached_prompts))]
        for raw_response in raw_responses:
            json_response = json.loads(raw_response)
            choices = [choice["message"]["content"] for choice in json_response["response"]["body"]["choices"]]
            if generation_kwargs["n"] == 1:
                choices = choices[0]
            results[int(json_response["custom_id"])] = choices

        # Update cache
        for prompt, response in zip(uncached_prompts, results):
            if response: # Only save to cache if prompt succeeded
                sm_cache[prompt] = response
        return [sm_cache[prompt] for prompt in prompts]


    def get_batched_responses(self, prompts: List[str], model: str, batch_size: int, generation_args: dict,
                            system_message: str = None, histories: List[str] = None, show_progress: bool = False):
        apply_arg_defaults(generation_args)
        # Load model's response cache
        use_cache = histories is None
        cache_filename = f"openai_cache_{model}.json"
        if use_cache:
            if os.path.exists(cache_filename):
                with open(cache_filename) as cache_file:
                    cache: dict = json.load(cache_file)
            else:
                cache = {}
            top_key = f"#ARGS-{generation_args}#SYS-{system_message}#"
            sm_cache = cache.setdefault(top_key, {})
            uncached_prompts = list({prompt for prompt in prompts if prompt not in sm_cache})
        else:
            uncached_prompts = prompts
        print(f"{len(prompts)} prompts, sending {len(uncached_prompts)} new requests")

        # Batch parallel requests to API
        responses = []
        it = range(0, len(uncached_prompts), batch_size)
        if show_progress:
            it = tqdm(it)
        try:
            for batch_start_idx in it:
                batch = uncached_prompts[batch_start_idx : batch_start_idx + batch_size]
                histories_batch = histories[batch_start_idx : batch_start_idx + batch_size] if histories else None
                batch_responses = self._get_parallel_responses(batch, model, generation_args,
                                                        system_message=system_message, histories=histories_batch)
                if use_cache:
                    for prompt, response in zip(batch, batch_responses):
                        sm_cache[prompt] = response
                else:
                    responses.extend(batch_responses)
        finally:
            # Update model's response cache
            if use_cache:
                print(f"Saving response cache for {model}")
                with open(cache_filename, "w") as cache_file:
                    json.dump(cache, cache_file)

        # Return responses
        if use_cache:
            return [sm_cache[prompt] for prompt in prompts]
        return responses

    def _get_parallel_responses(self, prompts: List[str], model: str, generation_args: dict,
                            system_message: str = None, histories: List[dict] = None):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            # Submit requests to threads
            futures = [
                executor.submit(self._get_responses, [prompt], model, generation_args,
                                system_message=system_message, histories=[histories[prompt_idx]] if histories else None)
                for prompt_idx, prompt in enumerate(prompts)
            ]

            # Wait for all to complete
            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            # Accumulate results
            results = [future.result()[0] for future in futures]
            return results

    def _get_responses(self, prompts: List[str], model: str, generation_args: dict,
                    system_message: str = None, histories: List[dict] = None, attempt: int = 1):
        global delay_time

        # Wait for rate limit
        time.sleep(delay_time)

        # Send request
        try:
            results = []
            for prompt_idx, prompt in enumerate(prompts):
                history = histories[prompt_idx] if histories else []
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_message or "You are a helpful assistant."
                        },
                        *history,
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    **generation_args,
                    # timeout=45
                )
                if generation_args["n"] == 1:
                    results.append(response.choices[0].message.content)
                else:
                    results.append([choice.message.content for choice in response.choices])
            delay_time = max(delay_time * decay_rate, 0.1)
        except (RateLimitError, APITimeoutError, APIError, APIConnectionError) as exc:
            print(openai.api_key, exc)
            delay_time = min(delay_time * 2, 30)
            if attempt >= max_attempts:
                print("Max attempts reached, prompt:")
                print(prompt)
                raise exc
            return self._get_responses(prompts, model, generation_args, system_message=system_message,
                                histories=histories, attempt=attempt + 1)
        except Exception as exc:
            print(exc)
            raise exc

        return results
