import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_val_1 = 0.35
dropout_val_2 = 0.5

class CNNImagesBranch(nn.Module):
    def __init__(self, input_channels, dropout_val=dropout_val_1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=0),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_val),

            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_val),

            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_val),
        )

    def forward(self, x):
        # x shape: (batch, time, channels, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, -1)  # restore time dimension
        return x

class CNNFeaturesBranch(nn.Module):
    def __init__(self, input_channels, dropout_val=dropout_val_1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=0),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout_val),

            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout_val),
        )

    def forward(self, x):
        # x shape: (batch, time, channels, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.cnn(x)
        x = x.view(B, T, -1)  # restore time dimension
        return x


class FeaturesBranch(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_val_1)
        )
    def forward(self, x):
        return self.fc(x)


class GRUBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_val=dropout_val_1, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_val)

    def forward(self, x):
        # x shape: (batch, time, feature_dim)
        output, _ = self.gru(x)
        # Take last time step
        return output[:, -1, :]


class FullModel(nn.Module):
    def __init__(self, img_shape, energy_shape, spectrum_shape, feature_dim, output_dim, dropout_val_1=0.35, dropout_val_2=0.5):
        super().__init__()

        # CNN branches
        self.images_cnn = CNNImagesBranch(img_shape[0], dropout_val_1)
        self.energies_cnn = CNNFeaturesBranch(energy_shape[0], dropout_val_1)
        self.spectrums_cnn = CNNFeaturesBranch(spectrum_shape[0], dropout_val_1)

        # GRU branches
        self.images_gru = GRUBranch(self._cnn_output_dim(img_shape), dropout_val_1)
        self.energies_gru = GRUBranch(self._cnn_output_dim(energy_shape), dropout_val_1)
        self.spectrums_gru = GRUBranch(self._cnn_output_dim(spectrum_shape), dropout_val_1)

        # Feature branch
        self.features_branch = FeaturesBranch(feature_dim, dropout_val_1)

        # Final layers
        total_concat_dim = 2*64 + 2*64 + 2*64 + 64  # three GRUs (bidirectional 64), plus feature 64
        self.fc = nn.Sequential(
            nn.Linear(total_concat_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_val_2),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def _cnn_output_dim(self, input_shape):
        # Dummy forward pass to find flattened CNN feature size
        C, H, W = input_shape
        x = torch.zeros(1, C, H, W)
        out = self.images_cnn.cnn(x)
        return out.view(1, -1).size(1)

    def forward(self, images, features, spectrums, energies):
        # CNN + GRU for images
        img_cnn = self.images_cnn(images)
        img_gru = self.images_gru(img_cnn)

        # CNN + GRU for energies
        energy_cnn = self.energies_cnn(energies)
        energy_gru = self.energies_gru(energy_cnn)

        # CNN + GRU for spectrums
        spec_cnn = self.spectrums_cnn(spectrums)
        spec_gru = self.spectrums_gru(spec_cnn)

        # Features
        feat_out = self.features_branch(features)

        merged = torch.cat([img_gru, spec_gru, energy_gru, feat_out], dim=1)
        out = self.fc(merged)
        return out



