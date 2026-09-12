from .common import Decoder


class Llama(Decoder):
    def __init__(self, config, weights):
        super().__init__(config, weights, config.get("rope_traditional", False))
