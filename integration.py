"""
Additional integration between libraries and AI2 Tango
"""
import copy

from typing import Optional, Type

from tango.common import Registrable
from tango.integrations.torch.model import Model
from tango.integrations.transformers import Config

from transformers import GPTQConfig as GPTQConfigOriginal
from transformers.models.auto import modeling_auto
from transformers.utils.quantization_config import QuantizationConfigMixin

import peft
from peft import get_peft_model, PeftModel, AdaLoraConfig, LoraConfig, PeftConfig as PeftConfigOriginal


# PEFT

# lora config
class PeftConfig(PeftConfigOriginal, Registrable):
    pass


PeftConfig.register('lora-config')(LoraConfig)
PeftConfig.register('adalora-config')(AdaLoraConfig)


# get_peft_model
class PeftWrapper(PeftModel):

    @classmethod
    def get_peft_model_constr(cls, base_model: Model, peft_config: PeftConfig) -> PeftModel:
        if base_model.is_quantized and base_model.quantization_method == 'gptq':
            base_model = peft.prepare_model_for_kbit_training(base_model)
        peft_model = get_peft_model(base_model, peft_config)

        # At some point tango needs to deepcopy the model, but auto-gptq's cuda-based QuantLinear classes have an
        #  attribute whose value is a module, which deepcopy cannot copy.
        # Here we patch the cuda versions of QuantLinear with a custom __deepcopy__ function to avoid this problem.

        def my_deepcopy(self, memo):
            new_instance = self.__class__.__new__(self.__class__)
            for k, v in self.__dict__.items():
                if k == 'autogptq_cuda':
                    setattr(new_instance, k, v)
                else:
                    setattr(new_instance, k, copy.deepcopy(v, memo))
            return new_instance

        from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear as QuantLinearCUDA
        from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as QuantLinearCUDAOld
        QuantLinearCUDA.__deepcopy__ = my_deepcopy
        QuantLinearCUDAOld.__deepcopy__ = my_deepcopy

        return peft_model


Model.register("peft::get_peft_model", constructor="get_peft_model_constr")(PeftWrapper)


# QUANTIZATION


class QuantizationConfig(QuantizationConfigMixin, Registrable):
    def __init__(self, *args, **kwargs):
        super(QuantizationConfigMixin).__init__(*args, **kwargs)


QuantizationConfig.register('gptq-config')(GPTQConfigOriginal)
# TODO register other quantization configs


# override default `from_pretrained` wrappers


def auto_model_wrapper_factory(cls: type) -> Type[Model]:
    class AutoModelWrapper(cls, Model):  # type: ignore
        @classmethod
        def from_pretrained(
                cls, pretrained_model_name_or_path: str, config: Optional[Config] = None,
                quantization_config: Optional[QuantizationConfig] = None, **kwargs
        ) -> Model:
            return super().from_pretrained(pretrained_model_name_or_path, config=config,
                                           quantization_config=quantization_config, **kwargs)

        @classmethod
        def from_config(cls, config: Config, **kwargs) -> Model:
            return super().from_config(config, **kwargs)

    return AutoModelWrapper


for name, cls in modeling_auto.__dict__.items():
    if isinstance(cls, type) and name.startswith("AutoModel"):
        wrapped_cls = auto_model_wrapper_factory(cls)
        Model.register(
            "transformers::" + name + "::from_pretrained", constructor="from_pretrained", exist_ok=True
        )(wrapped_cls)
        Model.register(
            "transformers::" + name + "::from_config", constructor="from_config", exist_ok=True
        )(wrapped_cls)
