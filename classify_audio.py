import torch
import yaml
import librosa
import numpy as np
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
    return model, d_args

def preprocess_audio(audio_path, sample_rate, nb_samp):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    if len(audio) < nb_samp:
        # Pad if too short
        audio = np.pad(audio, (0, nb_samp - len(audio)), mode='constant')
    else:
        # Truncate if too long
        audio = audio[:nb_samp]
    audio = torch.tensor(audio, dtype=torch.float32)
    return audio.unsqueeze(0)  # Add batch dimension

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 classify_audio.py <audio_file.wav>")
        sys.exit(1)
    audio_path = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, d_args = load_model(
        'librifake_pretrained_lambda0.5_epoch_25.pth',
        'model_config_RawNet.yaml',
        device
    )
    sample_rate = 24000
    nb_samp = d_args['nb_samp']
    audio_tensor = preprocess_audio(audio_path, sample_rate, nb_samp).to(device)
    with torch.no_grad():
        output = model(audio_tensor)
        print(f"Model output: {output}")
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            binary_output = output[0]
            print(f"Binary output (real/fake): {binary_output}")
            log_probs = binary_output.squeeze().cpu().numpy()
            print(f"Log-probabilities: {log_probs}")
            pred = torch.argmax(binary_output, dim=1).item()
            label = 'Real' if pred == 0 else 'Fake'
            print(f"Prediction: {label}")
        else:
            print("Unexpected output type or structure. Please check model definition.")
