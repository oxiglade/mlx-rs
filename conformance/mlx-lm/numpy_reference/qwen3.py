from .common import Decoder, rms_norm


class Qwen3(Decoder):
    def normalize_qk(self, queries, keys, prefix):
        queries = rms_norm(queries, self.weights[prefix + ".q_norm.weight"], self.eps)
        keys = rms_norm(keys, self.weights[prefix + ".k_norm.weight"], self.eps)
        return queries, keys
