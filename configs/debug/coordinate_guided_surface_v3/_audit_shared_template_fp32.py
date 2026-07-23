_base_ = ["four_fragments_four_views_scratch.py"]
experiment = dict(name="audit_shared_template_fp32")
model = dict(shared_template_encoding=True, static_geometry_cache=False)

