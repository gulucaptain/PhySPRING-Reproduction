"""PhysTwin motion-validity masks label transitions, not destination frames."""
import torch

def align_motion_mask(motion_valid,finite_target):
    result=torch.zeros_like(motion_valid,dtype=torch.bool)
    result[1:]=motion_valid[:-1].bool()&finite_target[1:]
    return result
