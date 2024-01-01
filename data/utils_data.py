from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch.utils.data import Dataset
import torch
from .data_splits import get_splits
import numpy as np
import random


class PartialLabelDataset(Dataset):
    def __init__(self, dataset, partialY, train_transforms, num_aug=6):
        super(PartialLabelDataset, self).__init__()

        self.dataset = dataset
        self.partialY = partialY
        self.aug_transforms = train_transforms
        self.num_aug = num_aug
        print('num_aug:', self.num_aug)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}

        anchor = self.dataset[index]
        output['graph'] = anchor
        output['aug'] = [self.aug_transforms(anchor) for _ in range(self.num_aug)]
        output['index'] = index
        output['target'] = self.partialY[index]

        return output


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def add_pose(data):
    data.x = torch.cat([data.x, data.pos], dim=-1)
    return data


def generate_uniform_graph_candidate_labels(train_labels, K, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def get_dataset(DS, path, args):
    setup_seed(0)

    if DS in ['Letter-high', 'COIL-DEL']:
        dataset = TUDataset(path, name=DS, use_node_attr=True, transform=T.AddSelfLoops())
        feat_mean = dataset.data.x.mean(dim=0)
        feat_std = dataset.data.x.std(dim=0)
        dataset.data.x = (dataset.data.x - feat_mean) / feat_std
    elif DS == 'ENZYMES':
        dataset = TUDataset(path, name=DS, use_node_attr=True, transform=T.AddSelfLoops())
    elif DS == 'CIFAR10':
        dataset = GNNBenchmarkDataset(path, name=DS, split='train', transform=add_pose)
    else:
        raise NotImplementedError

    dataset.data.edge_attr = None

    if DS == 'CIFAR10':
        train_dataset = GNNBenchmarkDataset(path, name=DS, split='train', transform=add_pose)
        valid_dataset = GNNBenchmarkDataset(path, name=DS, split='val', transform=add_pose)
        test_dataset = GNNBenchmarkDataset(path, name=DS, split='test', transform=add_pose)
        train_dataset.data.edge_attr = None
        valid_dataset.data.edge_attr = None
        test_dataset.data.edge_attr = None
    else:
        train_dataset, valid_dataset, test_dataset = get_splits(dataset, args)

    print(train_dataset, valid_dataset, test_dataset)
    print(train_dataset[0].x.shape)

    y_ori = []
    for i in range(len(train_dataset)):
        y_ori.append(train_dataset[i].y)
    y_ori = torch.tensor(y_ori).long()

    print(y_ori)
    partialY = generate_uniform_graph_candidate_labels(y_ori, train_dataset.num_classes, args.partial_rate)

    return dataset, train_dataset, valid_dataset, test_dataset, partialY
