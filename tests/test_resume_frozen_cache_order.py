import inspect
import unittest

from symm_template_reg.engine.overfit_trainer import run_overfit_training


class ResumeFrozenCacheOrderTest(unittest.TestCase):
    def test_resumed_model_is_loaded_before_cache_capture(self):
        source = inspect.getsource(run_overfit_training)
        restore = source.index('model.load_state_dict(resumed["model"], strict=True)')
        capture = source.index("capture_fine_adapter_inputs(")
        self.assertLess(
            restore,
            capture,
            "resume model state must be restored before frozen features are captured",
        )


if __name__ == "__main__":
    unittest.main()
