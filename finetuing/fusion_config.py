'''
Author: your name
Date: 2021-10-12 19:08:06
LastEditTime: 2021-10-13 15:54:45
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /model/lsc_code/preTrain_in_MIMIC-III/code/Fusion_bert/fusio_config.py
'''
import json
import copy

class FuseBertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 code_hidden_size=300,
                 code_num_hidden_layers=2,
                 code_num_attention_heads=4,
                 code_intermediate_size=300,
                 code_hidden_act="relu",
                 code_hidden_dropout_prob=0.4,
                 code_attention_probs_dropout_prob=0.1,
                 code_max_position_embeddings=1,
                 code_type_vocab_size=2,
                 graph=False,
                 graph_hidden_size=75,
                 graph_heads=4,
                 attention_probs_dropout_prob=0.1,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 hidden_size=768,
                 initializer_range=0.02,
                 intermediate_size=3072,
                 max_position_embeddings=512,
                 num_attention_heads=12,
                 num_hidden_layers=12,
                 type_vocab_size=2,
                 vocab_size=30522
                 ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.code_vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.graph = graph
            self.graph_hidden_size = graph_hidden_size
            self.graph_heads = graph_heads
            self.code_hidden_size=code_hidden_size
            self.code_num_hidden_layers=code_num_hidden_layers
            self.code_num_attention_heads=code_num_attention_heads
            self.code_intermediate_size=code_intermediate_size
            self.code_hidden_act=code_hidden_act
            self.code_hidden_dropout_prob=code_hidden_dropout_prob
            self.code_attention_probs_dropout_prob=code_attention_probs_dropout_prob
            self.code_max_position_embeddings=code_max_position_embeddings
            self.code_type_vocab_size=code_type_vocab_size
            self.vocab_size = vocab_size
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                                "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = FuseBertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
