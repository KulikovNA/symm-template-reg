import unittest
import torch
from symm_template_reg.models.losses import JointSurfaceCorrespondencePoseLossV3
from tests.test_fragment_symmetry_targets import metadata


class SharedSymmetryV3Test(unittest.TestCase):
    def test_all_components_select_one_joint_element(self):
        q=torch.tensor([[[.2,.2,0.],[.6,.2,0.],[.2,.6,0.]]])
        pose=torch.eye(4).unsqueeze(0);mask=torch.ones((1,3),dtype=torch.bool)
        vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces=torch.tensor([[0,1,2]])
        bary=torch.tensor([[[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]]])
        auxiliary={
            "coarse_patch_logits":torch.zeros((1,3,3),requires_grad=True),
            "fine_local_logits":torch.zeros((1,3,1),requires_grad=True),
            "selected_patch_ids":torch.zeros((1,3),dtype=torch.long),
            "selected_triangle_ids":torch.zeros((1,3),dtype=torch.long),
            "predicted_barycentric":bary,
            "all_candidate_triangle_ids":torch.zeros((1,3,1),dtype=torch.long),
            "candidate_triangle_ids":torch.zeros((1,3,1),dtype=torch.long),
            "patch_points_O":vertices[None],
        }
        result=JointSurfaceCorrespondencePoseLossV3()(q,pose,pose,q,q,mask,q,mask,[metadata()],[{"type":"C","order":1}],auxiliary,[vertices],[faces])
        self.assertEqual(int(result["selected_shared_symmetry_element"][0]),0)
        self.assertEqual(tuple(result["v3_loss_by_symmetry_element"].shape),(1,1))
        self.assertTrue(torch.isfinite(result["loss_total"]))

    def test_exact_candidate_is_required_only_for_selected_shared_element(self):
        vertices=torch.tensor([
            [1.,0.,0.],[2.,0.,0.],[1.,1.,0.],
            [-1.,0.,0.],[-2.,0.,0.],[-1.,1.,0.],
        ])
        faces=torch.tensor([[0,1,2],[3,4,5]])
        bary=torch.tensor([[[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]]])
        target=(bary[...,None]*vertices[faces[0]][None,None]).sum(2)
        predicted=(bary[...,None]*vertices[faces[1]][None,None]).sum(2)
        pose=torch.eye(4).unsqueeze(0)
        mask=torch.ones((1,3),dtype=torch.bool)
        auxiliary={
            "coarse_patch_logits":torch.zeros((1,3,2),requires_grad=True),
            "fine_local_logits":torch.zeros((1,3,1),requires_grad=True),
            "selected_patch_ids":torch.ones((1,3),dtype=torch.long),
            "selected_triangle_ids":torch.ones((1,3),dtype=torch.long),
            "predicted_barycentric":bary,
            "all_candidate_triangle_ids":torch.tensor([[[0],[1]]]),
            "face_owner_patch_ids":torch.tensor([[0,1]]),
            # Only the S=1 exact triangle is present.  S=0 is intentionally
            # absent and must not invalidate the shared-S contract.
            "candidate_triangle_ids":torch.ones((1,3,1),dtype=torch.long),
            "candidate_triangle_mask":torch.ones((1,3,1),dtype=torch.bool),
            "patch_points_O":vertices[[0,3]][None],
            "teacher_forcing_selected_symmetry_element":torch.tensor([1]),
        }
        loss=JointSurfaceCorrespondencePoseLossV3(
            lambda_patch_ce=0.,lambda_local_fine=1.,lambda_barycentric=0.,
            lambda_corr_mean=0.,lambda_corr_tail=0.,lambda_rot=0.,
            lambda_trans=0.,lambda_align_mean=0.,lambda_align_tail=0.,
            lambda_local_rigidity=0.,lambda_covariance=0.,
            lambda_min_eigenvalue=0.,lambda_patch_diversity=0.,
            patch_target_mode="multi_valid_patch_set",
            triangle_target_mode="multi_valid_patch_set",
            require_exact_triangle_candidate=True,
            use_teacher_forcing_shared_symmetry_element=True,
        )
        result=loss(
            predicted,pose,pose,predicted,target,mask,predicted,mask,
            [metadata()],[{"type":"C","order":2}],auxiliary,[vertices],[faces],
        )
        self.assertEqual(int(result["selected_shared_symmetry_element"][0]),1)
        self.assertEqual(float(result["triangle_target_index_mismatch_fraction"]),0.)
        self.assertTrue(torch.isfinite(result["loss_total"]))
