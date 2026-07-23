seed = 0

dataloader = dict(
    batch_size=2,
    num_workers=0,
)

collate = dict(type="FragmentTemplateCollator", mode="packed")
