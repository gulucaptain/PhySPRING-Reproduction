import math
import torch
from torch import nn
from .physics import System, project, laplacian
from .physics_v3 import GalerkinSystem,project_full,mass_matrix


def mlp(inp, out, hidden):
    return nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden,out))


def fps(x, count, mandatory=()):
    """Deterministic geometry representatives; an explicit missing-detail assumption."""
    x=x.detach().cpu()
    chosen=list(map(int,mandatory))
    if not chosen: chosen=[0]
    dist=((x[:,None]-x[chosen][None])**2).sum(-1).min(-1).values
    while len(chosen)<count:
        idx=int(dist.argmax()); chosen.append(idx)
        dist=torch.minimum(dist,((x-x[idx])**2).sum(-1));dist[chosen]=-1
    return torch.tensor(chosen,device='cpu',dtype=torch.long)


class Message(nn.Module):
    def __init__(self, hidden):
        super().__init__(); self.edge=mlp(3*hidden+4,hidden,hidden);self.node=mlp(2*hidden,hidden,hidden)
        self.enorm=nn.LayerNorm(hidden);self.vnorm=nn.LayerNorm(hidden)
    def forward(self,h,e,idx,types):
        i,j=idx
        e=self.enorm(e+self.edge(torch.cat([h[i],h[j],e,types[i],types[j]],-1)))
        aggregate=torch.zeros_like(h).index_add(0,i,e)
        h=self.vnorm(h+self.node(torch.cat([h,aggregate],-1)))
        return h,e


class PhySpring(nn.Module):
    def __init__(self, hidden=128, ratios=(.6,.4,.3), strategy='learned', temperature=1., seed=0, merge_rule="global", head_rule="additive"):
        super().__init__();self.ratios=ratios;self.strategy=strategy;self.temperature=temperature;self.seed=seed
        self.node_enc=mlp(36,hidden,hidden);self.edge_enc=mlp(4,hidden,hidden)
        self.enc=nn.ModuleList([Message(hidden) for _ in range(4)])
        self.dec=nn.ModuleList([Message(hidden) for _ in range(4)])
        self.heads=nn.ModuleList([mlp(hidden,3,hidden) for _ in range(4)])
        self.contacts=nn.ModuleList([mlp(hidden,3,hidden) for _ in range(4)])
        for head in [*self.heads,*self.contacts]:
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        self.frozen={};self.merge_rule=merge_rule;self.head_rule=head_rule;self.active_assignment=None;self.last_assignments=[];self.projection_rule="full"

    def assignment(self,h,system,target,level,stochastic):
        if level in self.frozen: return self.frozen[level].to(h)
        mandatory=torch.where(system.fixed)[0].cpu().tolist()
        target=max(target,len(mandatory)+1)
        centers=fps(system.rest,target,mandatory).to(h.device)
        if self.strategy=='random':
            gen=torch.Generator().manual_seed(self.seed+level*1009)
            free=[i for i in torch.randperm(len(h),generator=gen).tolist() if i not in mandatory]
            centers=torch.tensor(mandatory+free[:target-len(mandatory)],device=h.device)
        if self.strategy=='learned':
            delta=h[:,None,:]-h[centers][None,:,:]
            scores=-torch.sqrt((delta*delta).sum(-1)+1e-8)
        elif self.strategy=='random':
            scores=-torch.sqrt(((system.rest[:,None]-system.rest[centers][None])**2).sum(-1)+1e-8)
        elif self.strategy=='geometry':
            scores=-torch.sqrt(((system.rest[:,None]-system.rest[centers][None])**2).sum(-1)+1e-8)
        else: raise ValueError(self.strategy)
        # Controllers stay singleton, each representative owns itself; guarantees nonempty columns.
        allowed=torch.ones_like(scores,dtype=torch.bool)
        if self.merge_rule=='local':
            # Geometric neighborhood is an ablation assumption, not disclosed in Eq. 6.
            distances=((system.rest[:,None]-system.rest[centers][None])**2).sum(-1)
            distances[:,:len(mandatory)]=float('inf')
            nearest=distances.topk(min(3,target-len(mandatory)),largest=False).indices
            allowed=torch.zeros_like(allowed).scatter(1,nearest,True)
        elif self.merge_rule!='global':raise ValueError(self.merge_rule)
        if mandatory: allowed[:, :len(mandatory)]=False
        allowed[centers]=False;allowed[centers,torch.arange(target,device=h.device)]=True
        scores=scores.masked_fill(~allowed,-1e6)
        if stochastic and self.strategy=='learned' and (self.active_assignment is None or level==self.active_assignment):
            uniform=torch.rand_like(scores).clamp(1e-6,1-1e-6)
            scores=scores-torch.log(-torch.log(uniform))
        soft=torch.softmax(scores/self.temperature,dim=-1)
        hard=torch.nn.functional.one_hot(soft.argmax(-1),target).to(h)
        return hard+(soft-soft.detach()) if self.strategy=='learned' else hard

    def forward(self,base,stochastic=False):
        x=base.rest
        freq=2**torch.arange(5,device=x.device,dtype=x.dtype)*math.pi
        pe=torch.cat([(x[:,:,None]*freq).sin().flatten(1),(x[:,:,None]*freq).cos().flatten(1)],-1)
        types=torch.stack([~base.fixed,base.fixed],-1).to(x)
        h=self.node_enc(torch.cat([x,base.mass[:,None],types,pe],-1))
        systems=[base]; features=[]; assignments=[]; maps=[torch.eye(len(x),device=x.device,dtype=x.dtype)]
        for level in range(4):
            s=systems[level];types=torch.stack([~s.fixed,s.fixed],-1).to(x)
            idx=torch.where(s.stiffness.detach()>1e-7)
            delta=s.rest[idx[0]]-s.rest[idx[1]]
            e=self.edge_enc(torch.cat([delta,torch.linalg.vector_norm(delta,dim=-1,keepdim=True)],-1))
            h,e=self.enc[level](h,e,idx,types)
            features.append((h,e,idx,types))
            if level<3:
                target=max(2,round(len(x)*self.ratios[level]));target=min(target,len(h)-1)
                p=self.assignment(h,s,target,level,stochastic)
                assignments.append(p);maps.append(maps[-1]@p)
                h=(p.T@h)/p.sum(0)[:,None].clamp_min(1)
                systems.append(project_full(s,p) if self.projection_rule=="full" else project(s,p))
        output=[None]*4; coarse=None
        for level in range(3,-1,-1):
            h,e,idx,types=features[level]
            if coarse is not None: h=h+assignments[level]@coarse
            h,e=self.dec[level](h,e,idx,types);coarse=h
            residual=self.heads[level](e)
            s=systems[level]
            # Relative bounded positive corrections to Galerkin weights; beta is undocumented.
            if self.head_rule=='additive':
                # Eq.9 with disclosed beta choice: mean base edge stiffness, fixed per level.
                beta=s.stiffness.detach()[idx].mean()
                correction=torch.zeros_like(s.stiffness).index_put(idx,beta*residual[:,0])
                stiffness=(s.stiffness+.5*(correction+correction.T)).clamp_min(0)
                avg=residual.mean(0)
                dash=(-s.damping+torch.diag_embed(s.damping.diagonal())).clamp_min(0)
                beta_d=dash.detach()[idx].mean().clamp_min(1e-6)
                dash=(dash+beta_d*avg[1]*(s.stiffness.detach()>0)).clamp_min(0)
                drag=s.damping.sum(-1).clamp_min(0)
                drag=(drag+drag.detach().mean().clamp_min(1e-6)*avg[2]).clamp_min(0)
                damp=laplacian(dash)+torch.diag_embed(drag)
            elif self.head_rule=='multiplicative':
                factor=torch.zeros_like(s.stiffness).index_put(idx,torch.exp(residual[:,0].clamp(-4,4)))
                factor=.5*(factor+factor.T)
                stiffness=s.stiffness*factor
                avg=residual.mean(0)
                damp=s.damping*torch.exp(avg[1].clamp(-4,4))
                damp=damp+torch.diag_embed(s.mass)*torch.nn.functional.softplus(avg[2]-5)
            else:raise ValueError(self.head_rule)
            coeff=s.contact*torch.exp(self.contacts[level](h.mean(0)).clamp(-3,3))
            output[level]=GalerkinSystem(s.rest,s.mass,stiffness,damp,s.fixed,coeff,mass_matrix(s)) if self.projection_rule=="full" else System(s.rest,s.mass,stiffness,damp,s.fixed,coeff)
        self.last_assignments=[p.detach().clone() for p in assignments]
        return output,maps,assignments

    def commit_last(self,level):
        # Freeze the actually supervised deterministic partition; never redraw it at evaluation.
        if not self.last_assignments:raise RuntimeError('No evaluated partition to commit')
        self.frozen[level]=self.last_assignments[level].clone()

    def commit(self,base,level):
        with torch.no_grad():
            self(base,False)
            self.commit_last(level)
