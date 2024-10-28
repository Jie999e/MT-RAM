import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

class Retina(nn.Module):
    def __init__(self, k, s, g, c, out_c, kernel_size, stride):
        super().__init__()
        self.k = k
        self.s = s
        self.g = g
        self.c = c
        self.feat = nn.Sequential(
            nn.Conv2d(c * k, out_c[0], kernel_size, stride, (kernel_size - 1) // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_c[0], out_c[1], kernel_size, stride, (kernel_size - 1) // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_c[1], out_c[2], kernel_size, stride, (kernel_size - 1) // 2),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x, l, defect_map):
        phi = []
        size = self.g
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], k)
        phi = torch.cat(phi, 1)
        phi = self.feat(phi)
        phi = phi.view(phi.shape[0], -1)
        
        glimpse_precision = torch.sum(self.extract_patch(defect_map.unsqueeze(1), l, self.g), (3, 2, 1))
        
        return phi, glimpse_precision

    def extract_patch(self, x, l, size):
        B, C, H, W = x.shape
        start = self.denormalize(H, l)
        end = start + size
        #x = F.pad(x, (size // 2 + 1, size // 2 + 1, size // 2 + 1, size // 2 + 1))
        x = F.pad(x, (0, size, 0, size))
        patch = []
        for i in range(B):
            patch.append(x[i, :, start[i, 1]: end[i, 1], start[i, 0] : end[i, 0]])
        return torch.stack(patch)

    def denormalize(self, T, coords):
        return (0.5 * ((coords + 1.0) * T)).long()


class GlimpseNetwork(nn.Module):

    def __init__(self, h_g, h_l, g, k, s, c, out_c, kernel_size, stride):
        super().__init__()

        self.retina = Retina(k, s, g, c, out_c, kernel_size, stride)
        # glimpse layer
        D_in = g * g * out_c[2]
        self.fc1 = nn.Linear(D_in, h_g)
        # location layer
        D_in = 2
        self.fc2 = nn.Linear(D_in, h_l)
        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)

    def forward(self, x, l_t_prev, defect_map):
    
        # generate glimpse phi from image x
        phi, glimpse_precision = self.retina.forward(x, l_t_prev, defect_map)
        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))
        what = self.fc3(phi_out)
        where = self.fc4(l_out)
        # feed to fc layer
        g_t = F.relu(what + where)
        
        return g_t, glimpse_precision

class CoreNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, g_t, h1_t_prev, c1_t_prev, h2_t_prev, c2_t_prev):

        h1_t, c1_t = self.lstm1(g_t, (h1_t_prev, c1_t_prev))
        h2_t, c2_t = self.lstm2(h1_t, (h2_t_prev, c2_t_prev))

        return h1_t, c1_t, h2_t, c2_t

class ActionNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t

class DefectMapNetwork(nn.Module):

    def __init__(self, input_dim, x_shape, out_c, kernel_size, stride):
        super().__init__()
        
        hidden_dim_1 = input_dim * 2
        hidden_dim_2 = input_dim * 4

        self.x_shape = x_shape

        self.reshape = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_2, x_shape[0] * x_shape[1]),
            nn.ReLU(inplace=True),
        )

        self.feat = nn.Sequential(
            nn.Conv2d(1, out_c[0], kernel_size, stride, (kernel_size - 1) // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_c[0], out_c[1], kernel_size, stride, (kernel_size - 1) // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_c[1], 1, kernel_size, stride, (kernel_size - 1) // 2),
            nn.Sigmoid()
        )

    def forward(self, h_t):
        reshaped = torch.reshape(self.reshape(h_t), (h_t.shape[0], 1, self.x_shape[0], self.x_shape[1]))
        return self.feat(reshaped).squeeze()

class BaselineNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        return b_t


class LocationNetwork(nn.Module):

    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.std = std
        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)
 
    def forward(self, h_t):
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))
        l_t = torch.distributions.Normal(mu, self.std).rsample()
        l_t = l_t.detach()
        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = torch.sum(log_pi, dim=1)
        l_t = torch.clamp(l_t, -1, 1)
        return log_pi, l_t
