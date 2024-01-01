import torch


def get_splits(dataset, args):
    y = []
    for i in range(len(dataset)):
        y.append(dataset[i].y)
    y = torch.tensor(y)

    indices = []
    for i in range(dataset.num_classes):
        index = (y == i).nonzero().view(-1)
        if args.randperm:
            index = index[torch.randperm(index.size(0))]
        indices.append(index)

    num_graph = len(dataset)
    train_class_size = int(num_graph * 0.8 / dataset.num_classes)
    valid_class_size = int(num_graph * 0.05 / dataset.num_classes)
    print(train_class_size, valid_class_size)

    train_index = torch.cat([i[:train_class_size] for i in indices], dim=0)
    valid_index = torch.cat([i[train_class_size:train_class_size + valid_class_size] for i in indices], dim=0)
    test_index = torch.cat([i[train_class_size + valid_class_size:] for i in indices], dim=0)

    print(valid_index)

    train_dataset = dataset[train_index]
    valid_dataset = dataset[valid_index]
    test_dataset = dataset[test_index]

    return train_dataset, valid_dataset, test_dataset
