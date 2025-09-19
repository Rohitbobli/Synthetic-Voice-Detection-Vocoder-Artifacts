import torch
import yaml
from model import RawNet

# Load config
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['model']

def load_model(weights_path, config_path, device):
    d_args = load_config(config_path)
    model = RawNet(d_args, device).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(
        'librifake_pretrained_lambda0.5_epoch_25.pth',
        'model_config_RawNet.yaml',
        device
    )
    print("Model and weights loaded successfully.")
