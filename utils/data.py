#定义数据集类 ，指定数据下载、预处理变换和类别顺序
#定义数据集类，指定数据下载、预处理变换
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from . import autoaugment
from . import ops
from figure.process import process_pcap_to_cnn_format
class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class iEncryptedTraffic(iData): #slg
    #use_path 决定数据加载方式： False 表示数据是numpy数组直接加载， True 表示数据是图片路径需要动态加载。
    use_path = False 
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None
    def download_data(self, args):
        from convs.traffic_transformer import get_input_ids
        get_input_ids(args)

        data_path = \
        "~/llmfortraffic/IncrementalLearning/PyCIL/data/" + \
        args['dataset_name'] + "/"
        # 加载数据
        train_src = np.load(data_path+"train_src.npy")
        args['train_instances_num'] = len(train_src)
        
        train_seg = np.load(data_path+"train_seg.npy")
        self.train_targets = np.load(data_path+"train_label.npy")
        self.train_data = np.stack([train_src, train_seg], axis=-1)
        
        test_src = np.load(data_path+"test_src.npy")
        test_seg = np.load(data_path+"test_seg.npy")
        self.test_targets = np.load(data_path+"test_label.npy")
        self.test_data = np.stack([test_src, test_seg], axis=-1)

        valid_src = np.load(data_path+"valid_src.npy")
        valid_seg = np.load(data_path+"valid_seg.npy")
        self.valid_targets = np.load(data_path+"valid_label.npy")
        self.valid_data = np.stack([valid_src, valid_seg], axis=-1)

class iTrafficCIFAR(iData):
    # 使用统一的配置
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),  # 显式指定更清晰
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),  # 流量数据用通用值
            std=(0.5, 0.5, 0.5)
        ),
    ]
    use_path = False
    def download_data(self, args):
        train_data, train_labels, test_data, test_labels  = process_pcap_to_cnn_format(args["dataset_name"])
        self.train_data = train_data
        self.train_targets = train_labels
        self.test_data = test_data
        self.test_targets = test_labels
        # self.valid_data = valid_data
        # self.valid_targets = valid_labels


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        # datasets.ImageFolder
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100_AA(iCIFAR100):  #AA = AutoAugment       使用强化学习自动搜索最优增强策略
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),        # 针对 CIFAR 优化的策略
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),       # 随机遮挡
    ]


class iCIFAR10_AA(iCIFAR10):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
