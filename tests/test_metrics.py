import torch
from physpring.experiment import metrics

def test_empty_observation_frames_are_excluded():
    x=torch.zeros(1,3,4,3);y=x.clone();y[:,:,0,0]=1
    mask=torch.zeros(1,3,4,dtype=torch.bool);mask[:,0,:]=True
    m=metrics(x,y,mask)
    assert m['valid_frames']==1
    assert m['track_m']==.25
    assert m['cd_m']<1
    assert metrics(x,y,torch.zeros_like(mask))['track_m'] is None
