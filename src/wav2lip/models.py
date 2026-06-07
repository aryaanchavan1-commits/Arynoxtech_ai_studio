import warnings
import torch
from torch import nn
from torch.nn import functional as F

warnings.filterwarnings("ignore", message=".*__path__._path.*")
warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes.*")


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(self.weight, requires_grad=False)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias, requires_grad=False)


class Wav2Lip(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(3, 16, (7, 7), 1, (3, 3)),
                          nn.BatchNorm2d(16, affine=False),
                          nn.ReLU(inplace=True)),

            nn.Sequential(Conv2d(16, 32, (3, 3), (1, 2), (1, 1)),
                          nn.BatchNorm2d(32, affine=False),
                          nn.ReLU(inplace=True)),

            nn.Sequential(Conv2d(32, 64, (3, 3), (1, 2), (1, 1)),
                          nn.BatchNorm2d(64, affine=False),
                          nn.ReLU(inplace=True)),

            nn.Sequential(Conv2d(64, 128, (3, 3), (1, 2), (1, 1)),
                          nn.BatchNorm2d(128, affine=False),
                          nn.ReLU(inplace=True)),
        ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, (3, 3), 1, (1, 1)),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),

            Conv2d(32, 32, (3, 3), 1, (1, 1)),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),

            Conv2d(32, 64, (3, 3), (3, 1), (1, 1)),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),

            Conv2d(64, 64, (3, 3), 1, (1, 1)),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),

            Conv2d(64, 128, (3, 3), (3, 1), (1, 1)),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(inplace=True),

            Conv2d(128, 128, (3, 3), 1, (1, 1)),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(inplace=True),

            Conv2d(128, 256, (3, 3), (3, 1), (1, 1)),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(inplace=True),

            Conv2d(256, 256, (3, 3), 1, (1, 1)),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(inplace=True),

            Conv2d(256, 512, (3, 3), (3, 1), (1, 1)),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(inplace=True),
        )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(384, 512, (3, 3), 1, (1, 1)),
                          nn.BatchNorm2d(512, affine=False),
                          nn.ReLU(inplace=True)),

            nn.Sequential(Conv2d(512, 512, (3, 3), 1, (1, 1)),
                          nn.BatchNorm2d(512, affine=False),
                          nn.ReLU(inplace=True)),

            nn.Sequential(Conv2d(512, 256, (3, 3), 1, (1, 1)),
                          nn.BatchNorm2d(256, affine=False),
                          nn.ReLU(inplace=True)),

            nn.Sequential(Conv2d(512, 128, (3, 3), 1, (1, 1)),
                          nn.BatchNorm2d(128, affine=False),
                          nn.ReLU(inplace=True)),
        ])

        self.output_block = nn.Sequential(
            Conv2d(256 + 128, 64, (3, 3), 1, (1, 1)),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),

            Conv2d(64, 32, (3, 3), 1, (1, 1)),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(inplace=True),

            Conv2d(32, 3, (1, 1), 1, (0, 0)),
        )

    def forward(self, face_sequences, mel_sequences):
        B = face_sequences.size(0)

        feats = []
        x = face_sequences
        for block in self.face_encoder_blocks:
            x = block(x)
            feats.append(x)

        x = x.view(B, -1, x.size(-2), x.size(-1))
        x = F.interpolate(x, size=(8, 8))

        mel = mel_sequences
        mel = mel.unsqueeze(1).unsqueeze(-1)
        mel = mel.expand(-1, -1, -1, 8)
        mel = mel.permute(0, 1, 3, 2)
        mel = mel.contiguous()

        a = self.audio_encoder(mel)

        x = torch.cat([x, a], dim=1)

        for i, block in enumerate(self.face_decoder_blocks):
            if i == 3:
                x = F.interpolate(x, size=(feats[-1].size(-2), feats[-1].size(-1)))
                x = torch.cat([x, feats[-1]], dim=1)
            else:
                x = block(x)

        x = self.output_block(x)
        return x


def load_wav2lip(model_path: str, device: str = "cpu") -> Wav2Lip:
    try:
        model = torch.jit.load(model_path, map_location=device)
        model = model.to(device)
        model.eval()
        return model
    except Exception:
        pass

    model = Wav2Lip()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
