"""Fine local correspondence ladder; never run this base directly."""

_base_ = ["../correspondence_head_v4/_base.py"]

experiment = dict(name="fine_correspondence_v1_DO_NOT_RUN")

model = dict(
    correspondence_head=dict(
        teacher_forcing_initial_probability=1.0,
        teacher_forcing_final_probability=1.0,
        teacher_forcing_select_shared_symmetry_element=True,
        teacher_forcing_during_evaluation=True,
        deduplicate_local_candidates=True,
        inject_all_valid_triangles=True,
        teacher_force_exact_triangle=True,
        max_local_candidate_total=32,
        sort_owned_faces_by_distance=True,
        fine_feature_adapter=dict(
            type="FineLocalCorrespondenceFeatureAdapter",
            embed_dim=256,
            knn_scales=(8, 16, 32),
        ),
        fine_candidate_triangle_head=dict(
            type="FineCandidateTriangleHead", embed_dim=256,
            observed_geometry_dim=30, candidate_geometry_dim=22,
        ),
        fine_coordinate_auxiliary_head=dict(
            type="FineCanonicalCoordinateAuxiliaryHead", embed_dim=256,
        ),
    )
)

train = dict(
    optimizer=dict(type="AdamW", lr=3e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
)

loss = dict(
    joint_surface_correspondence_pose_v3=dict(
        triangle_target_mode="multi_valid_patch_set",
    )
)

stage_gate_dependencies = dict(
    require_parameterization_capacity=False,
    require_patch_gate=False,
    require_no_correspondence_collapse=False,
    require_no_excessive_attention_diffusion=False,
)
