import torch
import torch.nn as nn
import sys
from torchvision.models import resnet34

sys.path.append("..")
sys.path.append("../..")


# Channel Attention Module (CWA)
class CABlock(nn.Module):
    def __init__(self, in_channels, resize_factor=4):
        super(CABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hid_channels = 1
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, image_features):
        avg_pool_weights = self.fc(self.avg_pool(image_features))
        max_pool_weights = self.fc(self.max_pool(image_features))
        weights = torch.sigmoid(avg_pool_weights + max_pool_weights)

        return image_features * weights


# Spatial Attention Module (SWA)
class SABlock(nn.Module):
    def __init__(self):
        super(SABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, image_features):
        transpose_features = image_features.view(*image_features.shape[:2], -1).transpose(1, 2)
        avg_pooled_features = self.avg_pool(transpose_features)
        max_pooled_features = self.max_pool(transpose_features)
        pooled_features = torch.cat((avg_pooled_features, max_pooled_features), 2)
        pooled_features = pooled_features.transpose(1, 2).view(-1, 2, *image_features.shape[2:])
        weights = torch.sigmoid(self.conv(pooled_features))

        return image_features * weights, weights

        # pooled_1d = nn.functional.adaptive_avg_pool2d(image_features * weights, (3, 3))
        # output = pooled_1d.view(pooled_1d.shape[0], -1)
        # return output


# Temporal Attention Module
class TemporalAttention(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 4)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc(x).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 0)
        return context


# full connect
class fullConnectBlock(nn.Module):

    def __init__(self, total_classes, channel):
        super(fullConnectBlock, self).__init__()

        if channel == 2:
            self.model = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(648, 20),
                nn.ReLU(),
                nn.Linear(20, total_classes))

        elif channel == 21:
            self.model = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(384, 20),  # 32 384
                nn.ReLU(),
                nn.Linear(20, total_classes), )

    def forward(self, x):
        x = self.model(x)
        return x


# backbone
def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Conv4(nn.Module):
    def __init__(self, in_channels):
        super(Conv4, self).__init__()
        if in_channels == 2:
            out_channels = 8
            hid_channels = 16
        elif in_channels == 21:
            out_channels = 32
            hid_channels = 64

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hid_channels),
            conv3x3(hid_channels, hid_channels),
            conv3x3(hid_channels, hid_channels),
            conv3x3(hid_channels, out_channels)
        )

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        return outputs


# Self Attention
def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """
    Create and initialize a `nn.Conv1d` layer with spectral normalization.
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    # return spectral_norm(conv)
    return conv

class SelfAttention(nn.Module):
    """
    # self-attention implementation from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
    Self attention layer for nd
    """
    def __init__(self, n_channels: int, div):
        super(SelfAttention, self).__init__()

        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels//div)
            self.key = conv1d(n_channels, n_channels//div)
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        import torch.nn.functional as F
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class ProtoNetAGAM(nn.Module):
    def __init__(self, total_class, channel, type=3):
        super(ProtoNetAGAM, self).__init__()
        from backbones import resnet12
        # self.encoder = Conv4(in_channels=channel)
        self.encoder = resnet12(channel)
        # self.encoder = resnet34(num_classes=4)
        self.total_class = total_class
        self.ca_block = CABlock(in_channels=channel)
        self.sa_block = SABlock()
        self.fc_block = fullConnectBlock(total_class, channel)
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.rnn = nn.GRU(
            input_size=3,
            hidden_size=3,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.saT = SelfAttention(68, 1)
        self.saF = SelfAttention(21, 1)
        self.ta = TemporalAttention(4)
        self.type=type



    def forward(self, inputs, output_weights=False):
        if self.type==3:
            # embeddings = self.encoder(inputs)
            #ca_embeddings = self.ca_block(inputs)
            sa_embeddings, _ = self.sa_block(inputs)
            # refined = ca_embeddings + sa_embeddings
            # shape [5, 21, 60, 68] f:60 t:68
            # inputs_reshape = inputs.reshape(inputs.shape[0], -1, inputs.shape[3])
            # t_sa = torch.cat(
            #     [self.saT(torch.unsqueeze(inputs_reshape[:, f, :], dim=2)) for f in range(inputs_reshape.shape[1])],
            #     dim=-1,
            # )
            # t_sa = t_sa.reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3])
            # f_sa = torch.cat(
            #     [self.saF(torch.unsqueeze(inputs[:, :, :, t], dim=3)) for t in range(inputs.shape[3])],
            #     dim=-1,
            # )
            # refined = sa_embeddings + f_sa
            # lamda = 0.7
            # sa_embeddings = refined + inputs
            # sa_embeddings = sa_embeddings.reshape(ca_embeddings.shape[0], -1, ca_embeddings.shape[2])
            # rnn_embeddings, _ = self.rnn(sa_embeddings)
            # rnn_embeddings = rnn_embeddings.reshape(ca_embeddings.shape[0], ca_embeddings.shape[1], ca_embeddings.shape[2], ca_embeddings.shape[3])
            #embeddings = self.encoder(ca_embeddings + sa_embeddings)
            embeddings = self.encoder(sa_embeddings)
        elif self.type==2:
            ca_embeddings = self.ca_block(inputs)
            embeddings = self.encoder(ca_embeddings)
        elif self.type==1:
            ca_embeddings = self.ca_block(inputs)
            sa_embeddings, _ = self.sa_block(inputs)
            embeddings = self.encoder(ca_embeddings + sa_embeddings)
        full_connect_input = embeddings.reshape(embeddings.shape[0], -1)
        full_connect_output = self.fc_block(full_connect_input)
        result = self.softmax(full_connect_output)
        return result
