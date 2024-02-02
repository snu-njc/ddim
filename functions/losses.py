import torch
import math

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          linear=False, reflow=False, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    
    if linear:
        x = (x0 * a + e * (1.0 - a)) / (1.0 - 2 * a + 2 * a**2).sqrt()
    else:
        x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    
    if reflow:
        desired = (e - x0) / math.sqrt(2.0)
    else:
        desired = e
        
    if keepdim:
        return (desired - output).square().sum(dim=(1, 2, 3))
    else:
        return (desired - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
