import json
from argparse import Namespace


def load_hyperparam(args):
    # with open("~/llmfortraffic/IncrementalLearning/PyCIL/convs/models/bert/base_config_packets.json", mode="r", encoding="utf-8") as f:
    with open("convs/models/bert/base_config_packets.json", mode="r", encoding="utf-8") as f:
        param = json.load(f)
    #param:{'emb_size': 768, 'feedforward_size': 3072, 'hidden_size': 768, 
    # 'hidden_act': 'gelu', 'heads_num': 12, 'layers_num': 12, 'max_seq_length': 512, 'dropout': 0.1}
    
    args.update(param)

    return args
