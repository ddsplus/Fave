import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import time
import os
import pickle
import matplotlib.pyplot as plt
from contextlib import contextmanager
from thop import profile, clever_format


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics[f'HR@{k}'] = hr_temp
        metrics[f'NDCG@{k}'] = ndcg_temp
    return metrics


def LSHT_inference(model_joint, args, data_loader):
    device = args.device
    model_joint = model_joint.to(device)
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]
            scores_rec, rep_fave, _, _, _, _ = model_joint(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_fave = model_joint.fave_rep_pre(rep_fave)
            metrics = hrs_and_ndcgs_k(scores_rec_fave, test_batch[1], [5, 10, 20])
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)


@contextmanager
def timer(name, logger=None):
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    if logger:
        logger.info(f"{name} time: {elapsed:.4f} s")
    else:
        print(f"{name} time: {elapsed:.4f} s")

def optimizers(model, args, params=None):
    params_to_optimize = params if params is not None else model.parameters()
    if args.optimizer.lower() == 'adam':
        return optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError

def model_train(tra_data_loader, val_data_loader, test_data_loader, model_joint, args, logger, override_lr=None, override_params=None):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    Loss_Alpha = args.Loss_Alpha
    Loss_Beta = args.Loss_Beta

    model_joint = model_joint.to(device)

    current_lr = override_lr if override_lr is not None else args.lr
    optimizer = optimizers(model_joint, args, params=override_params)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)

    best_metrics_dict = {f'Best_HR@{k}': 0.0 for k in metric_ks}
    best_metrics_dict.update({f'Best_NDCG@{k}': 0.0 for k in metric_ks})
    best_epoch = {f'Best_epoch_HR@{k}': -1 for k in metric_ks}
    best_epoch.update({f'Best_epoch_NDCG@{k}': -1 for k in metric_ks})

    bad_count = 0
    best_model = None

    print(f"Start Training Stage: {args.stage_mode}, LR: {current_lr}, Epochs: {epochs}")
    logger.info(f"Start Training Stage: {args.stage_mode}, LR: {current_lr}, Epochs: {epochs}")

    for epoch_temp in range(epochs):
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()

        epoch_losses = []
        forward_mse_time = 0.0
        flag_update = 0

        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()

            loss_all, rep_fave, _, _, loss_main, _ = model_joint(
                train_batch[0],
                train_batch[1],
                forward_mse_time,
                train_flag=True
            )
            loss_fave_value = model_joint.loss_fave_tat(rep_fave, train_batch[1])
            loss_all += Loss_Alpha * loss_fave_value

            loss_all.backward()

            torch.nn.utils.clip_grad_norm_(model_joint.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_losses.append(loss_all.item())

            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Total Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                logger.info('[%d/%d] Total Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))

        average_loss = sum(epoch_losses) / len(epoch_losses)
        print("Average loss in epoch {}: {:.4f}".format(epoch_temp, average_loss))
        logger.info("Average loss in epoch {}: {:.4f}".format(epoch_temp, average_loss))

        lr_scheduler.step()

        # Evaluation
        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            model_joint.eval()
            with torch.no_grad():
                metrics_dict = {f'HR@{k}': [] for k in metric_ks}
                metrics_dict.update({f'NDCG@{k}': [] for k in metric_ks})

                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]
                    scores_rec, rep_fave, _, _, _, _ = model_joint(val_batch[0], val_batch[1], train_flag=False)
                    scores_rec_fave = model_joint.fave_rep_pre(rep_fave)

                    metrics = hrs_and_ndcgs_k(scores_rec_fave, val_batch[1], metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)

            current_epoch_metrics = {}
            for key_temp, values_temp in metrics_dict.items():
                values_mean = round(np.mean(values_temp) * 100, 4)
                current_epoch_metrics[key_temp] = values_mean

                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update = 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp

            print(f"Epoch {epoch_temp} Metrics: {current_epoch_metrics}")
            logger.info(f"Epoch {epoch_temp} Metrics: {current_epoch_metrics}")

            if flag_update == 1:
                print("New best model found!")
                best_model = copy.deepcopy(model_joint)

                ckpt_dir = os.path.dirname(args.ckpt_path)
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir, exist_ok=True)

                if args.stage_mode == 'pretrain':
                    torch.save(model_joint.state_dict(), args.ckpt_path)
                    print(f"Saved Best Pretrain Model to {args.ckpt_path}")
                elif args.stage_mode == 'finetune':
                    finetune_path = args.ckpt_path.replace('.pt', '_stage2.pt')
                    torch.save(model_joint.state_dict(), finetune_path)
                    print(f"Saved Best Finetune Model to {finetune_path}")
            else:
                bad_count += 1

            if bad_count >= args.patience:
                print("Early stopping triggered.")
                break

    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    if best_model is None:
        best_model = copy.deepcopy(model_joint)

    # Force finetune stage before final test
    if args.stage_mode == 'finetune':
        if isinstance(best_model, nn.DataParallel):
            best_model.module.stage = 'finetune'
        else:
            best_model.stage = 'finetune'
        print("[Final Test] Force setting best_model stage to 'finetune'")

    test_start_time = time.time()
    # Final Test
    top_100_item = []
    with torch.no_grad():
        test_metrics_dict = {f'HR@{k}': [] for k in metric_ks}
        test_metrics_dict.update({f'NDCG@{k}': [] for k in metric_ks})
        test_metrics_dict_mean = {}

        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]
            scores_rec, rep_fave, _, _, _, _ = best_model(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_fave = best_model.fave_rep_pre(rep_fave)

            metrics = hrs_and_ndcgs_k(scores_rec_fave, test_batch[1], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)

    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean

    test_end_time = time.time()
    test_duration = test_end_time - test_start_time

    print('Test Metrics:', test_metrics_dict_mean)
    logger.info('Test Metrics: {}'.format(test_metrics_dict_mean))
    print(f'Test Inference Time: {test_duration:.4f} s')
    logger.info(f'Test Inference Time: {test_duration:.4f} s')

    return best_model, test_metrics_dict_mean
