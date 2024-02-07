import torch
import math

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, linear, reflow, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        if linear:
            """
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                et = model(xt, t)
                x0_t = (xt - et * (1 - at)) / at
                x0_preds.append(x0_t.to('cpu'))
                c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at))
                )
                c2 = ((1 - at_next) ** 2 - c1 ** 2).sqrt()
                xt_next = at_next * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to('cpu'))
            """
        
            for idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                
                xt = xs[-1].to('cuda')
                if idx == 0:
                    xt *= (1 - 2 * (at - at**2)).sqrt()
                # et = model(xt * (at.sqrt() + (1-at).sqrt()), t)
                # rt = at.sqrt() / (at.sqrt() + (1-at).sqrt())
                # rt_next = at_next.sqrt() / (at_next.sqrt() + (1-at_next).sqrt())
                et = model(xt / (1 - 2 * (at - at**2)).sqrt(), t)
                rt = at
                rt_next = at_next
                x0_t = (xt - et * (1 - rt)) / rt
                x0_preds.append(x0_t.to('cpu'))
                c1 = (
                    kwargs.get("eta", 0) * ((1 - rt / rt_next) * (1 - rt_next) / (1 - rt))
                )
                c2 = ((1 - rt_next) ** 2 - c1 ** 2).sqrt()
                xt_next = rt_next * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to('cpu'))
        
        elif reflow:
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to(x.device)
                et_sub_x0 = model(xt, t) * math.sqrt(2.0)
                coef = (at_next.sqrt() + (1-at_next).sqrt()) / (at.sqrt() + (1-at).sqrt())
                xt_next = coef * xt - (at_next.sqrt() - coef * at.sqrt()) * et_sub_x0
                xs.append(xt_next.to('cpu'))
                
        else:
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to(x.device)
                et = model(xt, t)
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t.to('cpu'))
                c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    # NOTE : not fixed
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
