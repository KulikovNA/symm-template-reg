import unittest

from symm_template_reg.engine.overfit_trainer import validate_initialization_request


class ScratchFlagTest(unittest.TestCase):
    def test_scratch_excludes_all_checkpoint_loading(self):
        config = {"initialization_mode": "scratch", "pretrained_checkpoint": None}
        validate_initialization_request(config, from_scratch=True, resume=None, init_checkpoint=None, init_modules=None)
        for values in (
            {"resume": "last.pth", "init_checkpoint": None, "init_modules": None},
            {"resume": None, "init_checkpoint": "best.pth", "init_modules": None},
            {"resume": None, "init_checkpoint": "best.pth", "init_modules": ["observed_encoder"]},
        ):
            with self.assertRaisesRegex(ValueError, "excludes checkpoint"):
                validate_initialization_request(config, from_scratch=True, **values)

    def test_scratch_config_accepts_full_state_resume(self):
        config = {"initialization_mode": "scratch", "pretrained_checkpoint": None}
        validate_initialization_request(
            config,
            from_scratch=False,
            resume="best.pth",
            init_checkpoint=None,
            init_modules=None,
        )

    def test_scratch_config_still_rejects_implicit_or_transfer_initialization(self):
        config = {"initialization_mode": "scratch", "pretrained_checkpoint": None}
        for values in (
            {"resume": None, "init_checkpoint": None, "init_modules": None},
            {"resume": None, "init_checkpoint": "best.pth", "init_modules": None},
        ):
            with self.assertRaisesRegex(ValueError, "--from-scratch or --resume"):
                validate_initialization_request(config, from_scratch=False, **values)


if __name__ == "__main__": unittest.main()
