from huggingface_hub import HfApi, hf_hub_download

pairs = {
    'llama2-7B-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13B-chat': 'meta-llama/Llama-2-13b-chat-hf',

    'Orca-2-7B' :'microsoft/Orca-2-7b',
    'Orca-2-13B': 'microsoft/Orca-2-13b',

    'StableBeluga-7B':'stabilityai/StableBeluga-7B',
    'StableBeluga-13B':'stabilityai/StableBeluga-13B',

    'SOLAR-10.7B-Instruct':'upstage/SOLAR-10.7B-Instruct-v1.0',
    'SOLAR-10.7B':'upstage/SOLAR-10.7B-v1.0',

    'StableLM-Zephyr-3B':'stabilityai/stablelm-zephyr-3b',
    'Zephyr-7B-beta':'HuggingFaceH4/zephyr-7b-beta',

    'Mistral-7B-v0.1':'mistralai/Mistral-7B-v0.1',
    'Mistral-7B-Instruct-v0.1':'mistralai/Mistral-7B-Instruct-v0.1',
    'Mistral-7B-Instruct-v0.2':'mistralai/Mistral-7B-Instruct-v0.2',

    'Tulu-2-DPO-7B':'allenai/tulu-2-dpo-7b',
    'Tulu-2-DPO-13B':'allenai/tulu-2-dpo-1',

}

def downloadall():

    # get all keys from pairs dict and convert to list
    keys = list(pairs.keys())
    print(keys)

    for key in keys:
        pickle_name = f'{key}.pickle'
        repo_name = f'divyapatel4/{key}-jax'

        print(f'Downloading {pickle_name} from {repo_name}')
        api = HfApi()
        # download it in /mnt/mydisk/models
        local_filepath = hf_hub_download(repo_id=repo_name, filename=pickle_name, revision='main', output_dir='/mnt/mydisk/models')

    

if __name__ == '__main__':
    downloadall()

