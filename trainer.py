#训练流程控制，循环每个task
import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    pretrained_list = [True]
    if args["cnn"]:
        pretrained_list = [False]



    test_list = {
        1: ['ustc', 10, 5],     #数据集名称，初始类别数，任务数
        2: ['cstent', 60, 5],
        3: ['itcnet', 19, 5],
        4: ['Edge-IIoTset', 13, 5],
        5: ['cipherspectrum_aes128', 21, 5],
        6: ['cipherspectrum_mix', 21, 5],
        7: ['cipherspectrum_cha', 21, 5],
        8: ['cipherspectrum_aes256', 21, 5],

        9: ['ustc', 10, 2],     
        10: ['cipherspectrum_aes128', 21, 2],
        11: ['cstent', 60, 2],
        12: ['itcnet', 19, 2],
        13: ['Edge-IIoTset', 13, 2],
        14: ['cipherspectrum_mix', 21, 2],
        15: ['cipherspectrum_cha', 21, 2],
        16: ['cipherspectrum_aes256', 21, 2],

        17: ['ustc', 10, 10],     
        18: ['cipherspectrum_aes128', 21, 10],
        19: ['cstent', 60, 10],
        20: ['itcnet', 19, 10],
        21: ['Edge-IIoTset', 13, 10],
        22: ['cipherspectrum_mix', 21, 10],
        23: ['cipherspectrum_cha', 21, 10],
        24: ['cipherspectrum_aes256', 21, 10],
    }
    adapter_list = [11]
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        for pretrained in pretrained_list:
            args["pretrained"] = pretrained
            for _, (dataset_name, init_cls, tasknum) in test_list.items():
                args["dataset_name"] = dataset_name
                args["init_cls"] = init_cls
                args["tasknum"] = tasknum
                print(f"train {dataset_name}-{str(init_cls)}-{str(tasknum)}")
                if "adapter_one_feature" in args['config']:
                    for adapter_num in adapter_list:
                        args["adapter_num"] = adapter_num
                        _train(args)
                else:
                    _train(args)


def _train(args):

    #slg
    if not args["cnn"]:
        args["dataset"] = 'encrypted_traffic'   #固定        
        args["convnet_type"] = "traffic_transformer"#固定
    else:
        args["dataset"] = 'itrafficcifar'
        # args["convnet_type"] =    #保持不变
    args["prefix"] = "traffic"              #固定
    args['device'] = ["0"]
    args['fixed_memory'] = True   #固定   保存旧类别样本数量不变，随着类别增加，内存总量增加
    args['memory_per_class'] = 50   #固定
    # args['aug'] = 1         #固定 tagfex中需要aug=2
    args["increment"] = 0   #固定
    #slg
    if not args["cnn"]:
        if "adapter_num" in args and args["adapter_num"] != 12:
            model_name_log = args["model_name"] + "_" + str(args["adapter_num"])
        else:
            model_name_log = args["model_name"]
    else: 
        model_name_log = args["model_name"] + "_cnn"

    logs_name = "logs/{}/{}/".format(model_name_log,args["dataset_name"],)

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}_{}_{}_{}".format(
        model_name_log,
        args["dataset_name"],
        args["init_cls"],
        args["tasknum"],
        args["pretrained"],
        args["seed"],       #2
    )
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    _set_random()
    _set_device(args)
    
    # print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["aug"] if "aug" in args else 1,     #aug 数据增强
        args,
        args["tasknum"] if "tasknum" in args else None,   #slg，任务数量
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    fc_curve, simple_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    fc_matrix, simple_matrix = [], []
    cnn_f1, nme_f1, fc_f1, simple_f1 = [], [], [], []       #存储f1值用于计算f1平均值

    for task in range(data_manager.nb_tasks):
        logging.info("Task: {}".format(task))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy, fc_accy, simple_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: ACC: {:.2f}, PR: {:.2f}, RC: {:.2f}, F1: {:.2f}"
                         .format(cnn_accy["top1"], cnn_accy["precision"]*100, cnn_accy["recall"]*100, cnn_accy["f1"]*100))
            logging.info("NME: ACC: {:.2f}, PR: {:.2f}, RC: {:.2f}, F1: {:.2f}"
                         .format(nme_accy["top1"], nme_accy["precision"]*100, nme_accy["recall"]*100, nme_accy["f1"]*100))
            cnn_f1.append(cnn_accy["f1"]*100)
            nme_f1.append(nme_accy["f1"]*100)
            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)  ##00-09    00-09, 10-19
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]  
            cnn_matrix.append(cnn_values)   #42.4   41.6, 47.6

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_keys_sorted = sorted(nme_keys)
            nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
            nme_matrix.append(nme_values)
            cnn_curve["top1"].append(cnn_accy["top1"])
            nme_curve["top1"].append(nme_accy["top1"])
            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))
        else:
            logging.info("No NME accuracy.")
            # logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("CNN: ACC: {:.2f}, PR: {:.2f}, RC: {:.2f}, F1: {:.2f}"
                         .format(cnn_accy["top1"], cnn_accy["precision"]*100, cnn_accy["recall"]*100, cnn_accy["f1"]*100))
            cnn_f1.append(cnn_accy["f1"]*100)

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)
            cnn_curve["top1"].append(cnn_accy["top1"])
            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))

        if fc_accy is not None:
            logging.info("FC: ACC: {:.2f}, PR: {:.2f}, RC: {:.2f}, F1: {:.2f}"
                         .format(fc_accy["top1"], fc_accy["precision"]*100, fc_accy["recall"]*100, fc_accy["f1"]*100))
            fc_f1.append(fc_accy["f1"]*100)
            fc_keys = [key for key in fc_accy["grouped"].keys() if '-' in key]
            fc_keys_sorted = sorted(fc_keys)
            fc_values = [fc_accy["grouped"][key] for key in fc_keys_sorted]
            fc_matrix.append(fc_values)
            fc_curve["top1"].append(fc_accy["top1"])
            print('Average Accuracy (fc):', sum(fc_curve["top1"])/len(fc_curve["top1"]))
        if simple_accy is not None:
            logging.info("Simple: ACC: {:.2f}, PR: {:.2f}, RC: {:.2f}, F1: {:.2f}"
                         .format(simple_accy["top1"], simple_accy["precision"]*100, simple_accy["recall"]*100, simple_accy["f1"]*100))
            simple_f1.append(simple_accy["f1"]*100) 
            simple_keys = [key for key in simple_accy["grouped"].keys() if '-' in key]
            simple_keys_sorted = sorted(simple_keys)
            simple_values = [simple_accy["grouped"][key] for key in simple_keys_sorted]
            simple_matrix.append(simple_values)
            simple_curve["top1"].append(simple_accy["top1"])
            print('Average Accuracy (simple):', sum(simple_curve["top1"])/len(simple_curve["top1"]))
    
    if len(cnn_f1) > 0:
        avg_cnn_f1 = sum(cnn_f1) / len(cnn_f1)
        logging.info('Average F1 Score (CNN): {:.2f}'.format(avg_cnn_f1))
    if len(nme_f1) > 0:
        avg_nme_f1 = sum(nme_f1) / len(nme_f1)
        logging.info('Average F1 Score (NME): {:.2f}'.format(avg_nme_f1))
    if len(fc_f1) > 0:
        avg_fc_f1 = sum(fc_f1) / len(fc_f1)
        logging.info('Average F1 Score (FC): {:.2f}'.format(avg_fc_f1))
    if len(simple_f1) > 0:
        avg_simple_f1 = sum(simple_f1) / len(simple_f1)
        logging.info('Average F1 Score (Simple): {:.2f}'.format(avg_simple_f1))

    if len(cnn_matrix)>0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (CNN):')
        print(np_acctable)
        print('Forgetting (CNN):', forgetting)
        logging.info('Forgetting (CNN): {}'.format(forgetting))
    if len(nme_matrix)>0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (NME):')
        print(np_acctable)
        print('Forgetting (NME):', forgetting)
        logging.info('Forgetting (NME): {}'.format(forgetting))

    model.recoder_epcho()
    try:
        if len(cnn_matrix) > 0:
            for idxx, line in enumerate(cnn_matrix):
                logging.info(line)
        if len(nme_matrix) > 0:
            for idxx, line in enumerate(nme_matrix):
                logging.info(line)
    except:
        logging.info("matrix logging error")
    logging.info("\n\n\n\n\n")
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
