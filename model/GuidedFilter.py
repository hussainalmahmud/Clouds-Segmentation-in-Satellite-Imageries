import torch
import torch.nn.functional as F

class GuidedFilter(torch.nn.Module):
    """ A layer implementing guided filter """
    
    def __init__(self):
        super(GuidedFilter, self).__init__()

    def forward(self, I, p, r, eps=1e-8):
        def diff_x(inputs, r):
            assert len(inputs.shape) == 4

            left    = inputs[:, :,         :r]
            middle  = inputs[:, :,  r:-r  ] - inputs[:, :, :-2*r]
            right   = inputs[:, :, -r:] - inputs[:, :, -2*r:-r]

            outputs = torch.cat([left, middle, right], dim=2)

            return outputs

        def diff_y(inputs, r):
            assert len(inputs.shape) == 4

            left    = inputs[:, :, :,       :r]
            middle  = inputs[:, :, :, r:-r] - inputs[:, :, :, :-2*r]
            right   = inputs[:, :, :, -r:] - inputs[:, :, :, -2*r:-r]

            outputs = torch.cat([left, middle, right], dim=3)

            return outputs


        def box_filter(x, r):
            assert len(x.shape) == 4

            return diff_y(torch.cumsum(diff_x(torch.cumsum(x, dim=2), r), dim=1), r)

        assert len(I.shape) == 4 and len(p.shape) == 4

        I_shape = I.shape
        p_shape = p.shape

        # N
        N = box_filter(torch.ones((1, I_shape[1], I_shape[2], 1), dtype=I.dtype).to(I.device), r)

        # mean_x
        mean_I = box_filter(I, r) / N
        # mean_y
        mean_p = box_filter(p, r) / N
        # cov_xy
        cov_Ip = box_filter(I * p, r) / N - mean_I * mean_p
        # var_x
        var_I = box_filter(I * I, r) / N - mean_I * mean_I

        # A
        A = cov_Ip / (var_I + eps)
        # b
        b = mean_p - A * mean_I

        mean_A = box_filter(A, r) / N
        mean_b = box_filter(b, r) / N

        q = mean_A * I + mean_b

        return q