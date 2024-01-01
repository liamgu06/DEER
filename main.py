import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
import time
from arguments import arg_parse
from data.utils_data import get_dataset, PartialLabelDataset
from data.graph_aug import AugTransform
from models.model import DEER
from utils.loss import SupConLoss, PartialLoss
from utils.utils import compute_mmd_batch, get_logger, setup_seed


@torch.no_grad()
def test(loader, model):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        feat, feat_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        pred = model.classifier(feat).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


@torch.no_grad()
def eval_train(loader, model):
    model.eval()

    total_correct = 0
    for data_dict in loader:
        data = data_dict['graph']
        data = data.to(device)
        feat, feat_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        pred = model.classifier(feat).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


def run(seed):
    logger.info('seed:{}'.format(seed))

    epochs = args.epochs
    eval_interval = args.eval_interval
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    dataset, train_dataset, valid_dataset, test_dataset, partialY = get_dataset(DS, path, args)

    train_transforms = AugTransform(args.aug)
    partial_dataset = PartialLabelDataset(train_dataset, partialY, train_transforms, num_aug=args.num_aug)

    print('Calculating uniform targets...')
    tempY = partialY.sum(dim=1).unsqueeze(1).repeat(1, partialY.shape[1])
    confidence = partialY.float() / tempY
    confidence = confidence.to(device)

    loss_fn = PartialLoss(confidence, partialY.to(device))
    loss_cont_fn = SupConLoss(device)

    print(len(dataset))
    dataset_num_features = train_dataset[0].x.shape[1]
    print(dataset_num_features)

    dataloader = DataLoader(partial_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    dataloader_valid = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    setup_seed(seed)
    model = DEER(dataset_num_features, args.hidden_dim, args.num_gc_layers, dataset.num_classes, args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    best_val_acc = 0.0
    final_test_acc = 0.0
    for epoch in range(1, epochs + 1):
        time_start = time.time()

        label_correct = 0

        loss_all = 0
        loss_contr_all = 0
        loss_sup_all = 0
        model.train()
        for data_dict in dataloader:

            data = data_dict['graph']
            data_augs = data_dict['aug']
            partialy = data_dict['target']
            batch_index = data_dict['index'].to(device)

            optimizer.zero_grad()

            node_num, _ = data.x.size()

            data = data.to(device)
            data_augs = [data_aug.to(device) for data_aug in data_augs]
            x, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                def aug_node_filter(data_aug, data):
                    edge_idx = data_aug.edge_index
                    _, edge_num = edge_idx.shape
                    idx_not_missing = torch.unique(edge_idx)

                    node_num_aug = idx_not_missing.shape[0]
                    if 'x' in data.keys:
                        data_aug.x = data_aug.x[idx_not_missing]

                    data_aug.batch = data.batch[idx_not_missing]
                    # available_index = torch.unique(data_aug.batch)

                    idx_list = torch.zeros((idx_not_missing.max() + 1,), device=device, dtype=torch.long)
                    idx_list[idx_not_missing] = torch.arange(0, node_num_aug, device=device, dtype=torch.long)

                    self_circle = (edge_idx[0] == edge_idx[1])
                    edge_idx = edge_idx[:, ~self_circle]
                    edge_idx = idx_list[edge_idx.reshape(-1)].reshape(*edge_idx.shape)

                    data_aug.edge_index = edge_idx  # .transpose_(0, 1)

                    return data_aug

                data_augs = [aug_node_filter(data_aug, data) for data_aug in data_augs]

            x_augs = []
            x_aug_projs = []
            for data_aug in data_augs:
                x_aug, x_aug_proj = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
                x_augs.append(x_aug)
                x_aug_projs.append(x_aug_proj)
                if epoch <= args.start_upd_epoch:
                    break

            pred = model.classifier(x)

            aug_sizes = torch.tensor([x_aug_proj.size(0) for x_aug_proj in x_aug_projs])
            ori_size = x_proj.size(0)
            flag_missing = torch.sum(aug_sizes == ori_size) < len(aug_sizes)
            if flag_missing:
                print('Lose graph after augmentation! Skip contrast...')
                print(aug_sizes, ori_size)
                loss_contr = torch.tensor(0.0)
            else:
                if epoch <= args.start_upd_epoch:
                    loss_contr = model.loss_cal(x_proj, x_aug_projs[0])
                else:
                    # Distribution Divergence-based Graph Contrast
                    with torch.no_grad():
                        batch_size_ = len(batch_index)
                        x_augs_tensor = F.normalize(torch.stack(x_augs, dim=1), dim=-1)
                        mmd_sim = compute_mmd_batch(x_augs_tensor)

                        confidence = (loss_fn.confidence[batch_index, :] * loss_fn.partialY[batch_index, :]).clone()
                        pseudo_target_cont = torch.argmax(confidence, dim=-1, keepdim=True)
                        confidence_mask = torch.eq(pseudo_target_cont[:batch_size_], pseudo_target_cont.T).float()

                        mmd_threshold = torch.quantile(mmd_sim[confidence_mask.bool()].reshape(-1), args.mmd_filter_ratio)
                        mmd_mask = (mmd_sim <= mmd_threshold).float()

                        mask_pos = mmd_mask * confidence_mask
                        mask_pos.fill_diagonal_(1.0)
                        mask_pos = torch.cat((mask_pos, mask_pos), dim=1)

                    features_cont = torch.cat((x_proj, x_aug_projs[0]), dim=0)
                    loss_contr = loss_cont_fn(features=features_cont, mask=mask_pos, batch_size=batch_size_)

            partialy = partialy.to(device)

            if epoch > args.start_upd_epoch:
                # Exponential Moving Average (EMA)-based Label Correction
                with torch.no_grad():
                    pseudo_labels = torch.softmax(pred.clone().detach(), dim=-1) * partialy
                    loss_fn.confidence_update_soft(pseudo_label=pseudo_labels, batch_index=batch_index)

            loss_sup = loss_fn(pred, batch_index)

            loss = args.alpha * loss_contr + loss_sup

            pseudo_label_pred = torch.argmax(loss_fn.confidence[batch_index, :], dim=-1)

            label_correct += int((pseudo_label_pred == data.y).sum())

            loss_all += loss.item() * data.num_graphs
            loss_contr_all += loss_contr.item() * data.num_graphs
            loss_sup_all += loss_sup.item() * data.num_graphs
            loss.backward()
            optimizer.step()

        time_end = time.time()
        print('Epoch {}, Loss {:.4f}, '.format(epoch, loss_all / len(dataloader))
              + f'loss_contr: {loss_contr_all / len(dataloader):.4f}, '
              + f'loss_sup: {loss_sup_all / len(dataloader):.4f}, '
              + f'label_acc: {label_correct / len(dataloader.dataset):.4f},'
              + f' time_cost: {time_end-time_start:.2f} s')

        if epoch % eval_interval == 0:
            model.eval()
            train_acc = eval_train(dataloader, model)
            val_acc = test(dataloader_valid, model)
            test_acc = test(dataloader_test, model)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
            print(f'Epoch: {epoch:03d}, Loss: {loss_all / len(dataloader):.2f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

            if epoch % log_interval == 0:
                logger.info(f'Epoch: {epoch:03d}, Loss: {loss_all / len(dataloader):.2f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                            f'Test: {test_acc:.4f}, ' + f'label_acc: {label_correct / len(dataloader.dataset):.4f},'
                            + f' time_cost: {time_end-time_start:.2f} s')

    logger.info('best_val_acc: {:.2f}, final_test_acc: {:.2f}'.format(best_val_acc * 100, final_test_acc * 100))

    return final_test_acc


if __name__ == '__main__':
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = get_logger(args)

    acc_list = []
    for seed in range(args.st_seed, args.st_seed+args.number_of_run):
        test_acc = run(seed)
        acc_list.append(test_acc)

    print(acc_list)
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)

    logger.info("acc_mean: {:.2f}, acc_std: {:.2f}".format(acc_mean * 100, acc_std * 100))
