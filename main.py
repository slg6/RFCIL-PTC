#程序入口，解析命令行参数，加载JSON配置文件
import json
import argparse
from trainer import train
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if sys.gettrace() is None:      
    os.environ["Pycil_Debug"] = "0" #没有debug
else:
    os.environ["Pycil_Debug"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

from convs.uer.opts import finetune_opts

def main():
    args = setup_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    model_list = {
        1: ['adapter_one_feature', False],
    }
    for _, (config, cnn) in model_list.items():
        args['config'] = f'./exps/{config}.json'
        args['cnn'] = cnn
        param = load_json(args['config'])
        args.update(param)  # Add parameters from json
        train(args)



def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default='./exps/adapter_one_feature.json',
                        help='Json file of settings.')
    parser.add_argument('--cnn', type=bool, default=False,
                        help='cnn or transformer')
    finetune_opts(parser)



    return parser


if __name__ == '__main__':
    main()
