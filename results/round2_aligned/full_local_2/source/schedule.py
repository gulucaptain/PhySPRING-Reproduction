"""Stage-local sampling schedule with at least one deterministic supervised step."""
import math

def stochastic_step(epoch,epochs,legacy=False):
    if epochs<4 or not 0<=epoch<epochs:
        raise ValueError('Need at least four epochs and a valid epoch index')
    if legacy:return True
    stage=min(3,epoch*4//epochs)
    start=math.ceil(stage*epochs/4)
    end=math.ceil((stage+1)*epochs/4)
    tail=max(1,math.ceil((end-start)*.25))
    return stage>0 and epoch<end-tail
