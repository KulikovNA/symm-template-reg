_base_ = ["../conditioning/legacy_absolute_queries.py"]

experiment = dict(name="conditioned_legacy_static_query_baseline")
stage = dict(name="legacy_static_query_baseline", checkpoint_filename="best_legacy_static.pth")
