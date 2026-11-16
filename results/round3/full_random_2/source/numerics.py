import math
import torch


def safe_clip_grad_norm(parameters,limit):
    """Preserve ordinary torch clipping; avoid overflow of squared large finite gradients."""
    parameters=list(parameters);grads=[p.grad for p in parameters if p.grad is not None]
    if not grads:return torch.tensor(0.)
    norms=torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads])
    total=torch.linalg.vector_norm(norms)
    if torch.isfinite(total):return torch.nn.utils.clip_grad_norm_(parameters,limit)
    if not all(bool(torch.isfinite(g).all()) for g in grads):return torch.tensor(float('nan'))
    # CPU double only for the exceptional overflow case; do not silently zero gradients.
    total64=math.sqrt(sum(float(g.detach().cpu().double().square().sum()) for g in grads))
    scale=min(1.,limit/(total64+1e-6))
    with torch.no_grad():
        for g in grads:g.mul_(scale)
    return torch.tensor(total64,dtype=torch.float64)
