import torch
from .physics import System, laplacian, rollout


def make_scene(kind='cloth', size=6, frames=32, batch=4, seed=123, substeps=16, dtype=torch.float32):
    """Procedural diagnostic, NOT the PhysTwin dataset. No hidden parameters passed to model."""
    torch.manual_seed(seed)
    if kind=='rope':
        n=size*size
        rest=torch.stack([torch.linspace(-.4,.4,n),torch.zeros(n),torch.ones(n)*.6],-1).to(dtype)
        d=(rest[:,None]-rest[None]).norm(dim=-1)
        adjacency=(d>0)&(d<1.6*.8/(n-1))
        fixed=torch.zeros(n,dtype=torch.bool);fixed[0]=True;fixed[-1]=True
    else:
        a,b=torch.meshgrid(torch.linspace(-.35,.35,size),torch.linspace(-.3,.3,size),indexing='ij')
        rest=torch.stack([a.flatten(),b.flatten(),torch.ones(size*size)*.65],-1).to(dtype)
        d=(rest[:,None]-rest[None]).norm(dim=-1)
        adjacency=(d>0)&(d<1.5*.7/(size-1))
        n=len(rest);fixed=torch.zeros(n,dtype=torch.bool);fixed[0]=True;fixed[size-1]=True
    mass=torch.full((n,),1/n,dtype=dtype)
    # Nonuniform stiffness creates a nontrivial, spatially varying identification problem.
    weights=adjacency.to(dtype)*(8+16*torch.sigmoid(12*(rest[:,None,0]+rest[None,:,0])))
    damping=laplacian(adjacency.to(dtype)*.04)+torch.eye(n,dtype=dtype)*.03
    true=System(rest,mass,weights,damping,fixed,torch.tensor([200.,2.,.25],dtype=dtype))
    initial=System(rest,mass,adjacency.to(dtype)*12,laplacian(adjacency.to(dtype)*.02)+torch.eye(n,dtype=dtype)*.02,fixed,torch.tensor([120.,1.,.15],dtype=dtype))
    dt=1/30
    controls=rest[None,None].expand(batch,frames+1,n,3).clone()
    t=torch.arange(frames+1,dtype=dtype)*dt
    for b in range(batch):
        freq=.65+.2*b
        controls[b,:,:,0]+=(.045*torch.sin(2*torch.pi*freq*t))[:,None]
        controls[b,:,:,2]+=(.06*torch.sin(torch.pi*freq*t))[:,None]
    x0=rest[None].expand(batch,-1,-1).clone();v0=torch.zeros_like(x0)
    with torch.no_grad():
        obs=rollout(true,x0,v0,controls,dt,substeps,mode='axial',floor=0.)
    return dict(initial=initial,truth=true,controls=controls,x0=x0,v0=v0,observations=obs,dt=dt,substeps=substeps,kind=kind,seed=seed)


def save_scene(scene,path):
    # plain tensor dict, safe weights_only load supported
    obj={k:(v.__dict__ if isinstance(v,System) else v) for k,v in scene.items()}
    torch.save(obj,path)


def load_scene(path,device='cpu'):
    obj=torch.load(path,map_location=device,weights_only=True)
    for k in ('initial','truth'):
        if k in obj:obj[k]=System(**obj[k])
    return obj
