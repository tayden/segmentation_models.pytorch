import inspect

from tests.models import base


class TestEoMTModel(base.BaseModelTester):
    test_model_type = "eomt"
    test_encoder_name = "tu-vit_tiny_patch16_224"
    files_for_diff = [r"decoders/eomt/", r"base/"]

    default_height = 224
    default_width = 224

    compile_dynamic = False

    @property
    def decoder_channels(self):
        signature = inspect.signature(self.model_class)
        return signature.parameters["decoder_num_queries"].default
