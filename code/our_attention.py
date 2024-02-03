import torch
import torch.nn as nn
import numpy as np

class CustomAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(CustomAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        # rgb
        self.query_r = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_r = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_r = nn.Linear(feature_dim, feature_dim, bias=False)
        # skeleton
        self.query_s = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_s = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_s = nn.Linear(feature_dim, feature_dim, bias=False)

        self.fc_out = nn.Linear(feature_dim, feature_dim, bias=False)


    def forward(self, r, s, alpha=1.0):
        r_t, s_t = r, s
        r, s = r.unsqueeze(1), s.unsqueeze(1)
        B, N, _ = r.shape

        Q_r = self.query_r(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_r = self.key_r(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_r = self.value_r(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        Q_s = self.query_s(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_s = self.key_s(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_s = self.value_s(r).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)


        energy_r2s = torch.matmul(Q_r, K_s.permute(0, 1, 3, 2)) / (self.feature_dim ** 0.5)
        energy_s2r = torch.matmul(Q_s, K_r.permute(0, 1, 3, 2)) / (self.feature_dim ** 0.5)

        attention_r2s = torch.softmax(energy_r2s, dim=-1)
        attention_s2r = torch.softmax(energy_s2r, dim=-1)

        if alpha > 0:
            alpha = np.random.beta(alpha, alpha)
        else:
            alpha = 1.0

        out = (alpha * torch.matmul(attention_r2s, V_r) + (1-alpha)* torch.matmul(attention_s2r, V_s)).permute(0, 2, 1, 3).contiguous()
        out = out.view(B, N, self.feature_dim)
        out = self.fc_out(out)
        out = out.squeeze()
        return out + s_t + r_t

if __name__ == '__main__':

    feature_dim = 512
    num_heads = 8
    f1 = torch.randn(8, feature_dim)
    f2 = torch.randn(8, feature_dim)

    # Concatenate f1 and f2

    a = CustomAttention(feature_dim, num_heads)
    b = a(f1, f2)
    print(b, b.shape)
