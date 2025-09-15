from huggingface_hub import hf_hub_download
from vocos import Vocos
import torch

def load_vocoder(is_local=False, local_path="", device='cuda', hf_cache_dir=None):
    # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    if is_local:
        print(f"Load vocos from local path {local_path}")
        config_path = f"{local_path}/config.yaml"
        model_path = f"{local_path}/pytorch_model.bin"
    else:
        print("Download Vocos from huggingface charactr/vocos-mel-24khz")
        repo_id = "charactr/vocos-mel-24khz"
        config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
        model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
    vocoder = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    from vocos.feature_extractors import EncodecFeatures

    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocoder.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)
    vocoder.load_state_dict(state_dict)
    vocoder = vocoder.eval().to(device)
    return vocoder