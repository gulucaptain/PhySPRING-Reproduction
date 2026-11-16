import torch
import torch.nn.functional as F


def observation_loss(pred,target,geometry_mask,tracking_mask):
    """Reference PhysTwin: one-way squared Chamfer + SmoothL1(beta=1).

    Frame means weighted equally, independent masks; empty frames excluded per loss.
    """
    # Sanitize invalid coordinates BEFORE any arithmetic to prevent NaN*0 leakage.
    gt_geo=torch.where(geometry_mask[...,None],target,torch.zeros_like(target))
    dd=(gt_geo[:,:,:,None,:]-pred[:,:,None,:,:]).square().sum(-1)
    nearest=dd.min(-1).values
    ng=geometry_mask.sum(-1);valid_g=ng>0
    per_geo=(nearest*geometry_mask).sum(-1)/ng.clamp_min(1)
    geo=(per_geo*valid_g).sum()/valid_g.sum().clamp_min(1)
    gt_track=torch.where(tracking_mask[...,None],target,pred.detach())
    tracking=F.smooth_l1_loss(pred,gt_track,reduction='none',beta=1.).mean(-1)
    nt=tracking_mask.sum(-1);valid_t=nt>0
    per_track=(tracking*tracking_mask).sum(-1)/nt.clamp_min(1)
    track=(per_track*valid_t).sum()/valid_t.sum().clamp_min(1)
    return geo+track,{'geometry':geo.detach(),'tracking':track.detach()}
