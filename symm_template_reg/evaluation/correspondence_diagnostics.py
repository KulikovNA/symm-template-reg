"""Interpretable diagnostics for dense canonical correspondences."""

from __future__ import annotations
import torch
from torch import Tensor
from symm_template_reg.models.geometry.point_ops import knn_indices


def rowwise_and_chamfer(predicted: Tensor, target: Tensor) -> dict[str, Tensor]:
    row = torch.linalg.vector_norm(predicted-target,dim=-1)
    distance=torch.cdist(predicted.unsqueeze(0).float(),target.unsqueeze(0).float())[0]
    return {'rowwise_distance':row,'symmetric_chamfer_distance':.5*(distance.amin(1).mean()+distance.amin(0).mean())}


def attention_distribution_metrics(logits: Tensor) -> dict[str, Tensor]:
    probability=torch.softmax(logits.float(),-1);eps=torch.finfo(probability.dtype).eps
    entropy=-(probability*probability.clamp_min(eps).log()).sum(-1)
    sorted_probability=probability.sort(-1,descending=True).values
    argmax=probability.argmax(-1);counts=torch.bincount(argmax,minlength=probability.shape[-1]);unique=(counts>0).sum()
    return {
        'entropy':entropy,'normalized_entropy':entropy/torch.log(torch.tensor(float(probability.shape[-1]),device=entropy.device)).clamp_min(eps),
        'top1_mass':sorted_probability[:,:1].sum(-1),'top5_mass':sorted_probability[:,:5].sum(-1),'top16_mass':sorted_probability[:,:16].sum(-1),
        'argmax':argmax,'anchor_counts':counts,'unique_argmax_anchors':unique,
        'collision_ratio':1.0-unique.float()/max(len(argmax),1),'most_popular_anchor_fraction':counts.max().float()/max(len(argmax),1),
        'effective_candidate_count':torch.exp(entropy),
    }


def covariance_geometry(points: Tensor, axis: Tensor|None=None)->dict[str,Tensor]:
    centered=points-points.mean(0);cov=centered.T@centered/max(len(points),1);eigen=torch.linalg.eigvalsh(cov)
    result={'bbox_min':points.amin(0),'bbox_max':points.amax(0),'covariance_eigenvalues':eigen,'rank':torch.linalg.matrix_rank(centered)}
    if axis is not None:
        unit=torch.nn.functional.normalize(axis.float(),dim=0);axial=centered@unit;radial=centered-axial[:,None]*unit
        result.update(axial_extent=axial.max()-axial.min(),radial_extent=torch.linalg.vector_norm(radial,dim=-1).max())
    return result


def local_rigidity_errors(predicted:Tensor,observed:Tensor,k:int=8)->Tensor:
    if len(observed) < 2:
        return predicted.new_zeros((len(observed), 1))
    k=min(int(k),len(observed)-1)
    mask=torch.ones((1,len(observed)),dtype=torch.bool,device=observed.device);neighbors=knn_indices(observed[None],observed[None],mask,k+1)[0,:,1:]
    observed_edges=torch.linalg.vector_norm(observed[:,None]-observed[neighbors],dim=-1);predicted_edges=torch.linalg.vector_norm(predicted[:,None]-predicted[neighbors],dim=-1)
    return (predicted_edges-observed_edges).abs()


def pairwise_distance_correlation(predicted:Tensor,observed:Tensor,max_points:int=512)->float:
    ids=torch.linspace(0,len(predicted)-1,min(len(predicted),max_points),device=predicted.device).long();left=torch.pdist(predicted[ids].float());right=torch.pdist(observed[ids].float());left-=left.mean();right-=right.mean();den=torch.linalg.vector_norm(left)*torch.linalg.vector_norm(right)
    return float(torch.dot(left,right)/den.clamp_min(1e-12))


__all__=['attention_distribution_metrics','covariance_geometry','local_rigidity_errors','pairwise_distance_correlation','rowwise_and_chamfer']
