import torch.nn as nn

import modules


class RecurrentAttention(nn.Module):
    
    def __init__(
        self, g, k, s, c, h_g, h_l, std1, hidden_size, num_classes, out_c, out_c_2, kernel_size, stride, x_shape
    ):
        
        super().__init__()

        
        self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c, out_c, kernel_size, stride)
        self.rnn = modules.CoreNetwork(h_g + h_l, hidden_size)
        self.classifier = modules.ActionNetwork(hidden_size, num_classes)
        self.detector = modules.DefectMapNetwork(hidden_size, x_shape, out_c_2, kernel_size, stride)
        self.baseliner = modules.BaselineNetwork(hidden_size, 1)
        self.locator = modules.LocationNetwork(hidden_size, 2, std1)
        
    def forward(self, x, l_t_prev, h1_t_prev, c1_t_prev, h2_t_prev, c2_t_prev, defect_map, last = False):
        
        n_t, glimpse_precision = self.sensor(x, l_t_prev, defect_map)
        h1_t, c1_t, h2_t, c2_t = self.rnn(n_t, h1_t_prev, c1_t_prev, h2_t_prev, c2_t_prev)
        
        b_t = self.baseliner(h2_t).squeeze()
        log_pi_l, l_t = self.locator(h2_t)
        
        if last:
            log_probas = self.classifier(h1_t)
            defect_map = self.detector(h1_t)
            return h1_t, c1_t, h2_t, c2_t, l_t, b_t, log_probas, defect_map, log_pi_l, glimpse_precision

        return h1_t, c1_t, h2_t, c2_t, l_t, b_t, log_pi_l, glimpse_precision