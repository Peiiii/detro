import torch
import numpy as np
import os
import time
import logging

def load_weights(model, weights_path=None, except_keys=[]):
    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path)
            remove_keys = []
            for k in state_dict.keys():
                for key in except_keys:
                    if k.startswith(key):
                        remove_keys.append(k)
            for k in remove_keys:
                del state_dict[k]
                print('ignore key:', k)
            model.load_state_dict(state_dict, strict=False)
        except:
            logging.warning('Cannot load pretrained model %s' % (weights_path))
            pass
            # raise
    return model


def train(cfg=None):
    t = time.time()
    time_stamp = time.strftime('%m%d-%H', time.localtime(t))
    model = cfg.get_model()
    # model = load_weights(model, cfg.WEIGHTS_INIT, except_keys=['reg_layer'])
    model = load_weights(model, cfg.WEIGHTS_INIT)
    device = cfg.DEVICE
    model.to(device)
    train_data = cfg.dataset
    optimizer = torch.optim.Adam(model.parameters(), cfg.LEARN_RATE_INIT)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.MAX_EPOCHS, cfg.LEARN_RATE_END)
    min_loss = 1e+10
    for epoch in range(cfg.MAX_EPOCHS):
        model.train()
        loss_history = {}
        loss_avg_dict = {}
        loss_summary = []

        for batch_image, batch_labels in train_data:
            preds = model(batch_image)
            loss_dict = cfg.criterion(preds, batch_labels)
            loss = loss_dict['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k, v in loss_dict.items():
                if not k in loss_history.keys():
                    loss_history[k] = []
                loss_history[k].append(v.item())
        lr_scheduler.step()
        for k, v in loss_history.items():
            loss_avg_dict[k] = np.array(v).mean()
            loss_summary.append('%s:%.4f' % (k, loss_avg_dict[k]))
        loss_epoch = loss_avg_dict['loss']
        loss_summary = '\t'.join(loss_summary)
        print('Epoch: %s\tLearningRate:%.6f\t%s' % (
            epoch, optimizer.state_dict()['param_groups'][0]['lr'], loss_summary))
        if epoch and cfg.WEIGHTS_SAVE_INTERVAL and epoch % cfg.WEIGHTS_SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), cfg.WEIGHTS_DIR + f'/model_{time_stamp}_{epoch}_loss_{loss_epoch:.4f}.pkl')
        if loss_epoch < min_loss:
            min_loss = loss_epoch
            torch.save(model.state_dict(), cfg.WEIGHTS_DIR + f'/model_best_{time_stamp}.pkl')
            torch.save(model.state_dict(), cfg.WEIGHTS_DIR + f'/model_best.pkl')
        torch.save(model.state_dict(), cfg.WEIGHTS_DIR + f'/model.pkl')



