import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from dataloder.dataset1718 import NPY_datasets
from utils import LogWritter
# from loss_fn import ConfidentLoss
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import cv2

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss




class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, edge_indices=0, edge_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = 1.0 - alpha
        self.edge_indices = edge_indices
        self.edge_weight = edge_weight

    def dice_loss(self, pred, target):
        smooth = 1.
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1.0 - (2. * intersection + smooth) / (union + smooth)
    def get_edge_from_mask(self, mask):
        edge_batch = []
        mask_np = mask.cpu().numpy()
        for b in range(mask_np.shape[0]):
            mask_img = (mask_np[b, 0] * 255).astype(np.uint8)
            edge = cv2.Canny(mask_img, 100, 200)
            edge = torch.from_numpy(edge / 255.).float().unsqueeze(0)
            edge_batch.append(edge)
        return torch.stack(edge_batch).to(mask.device)

    def forward(self, preds, mask):
        total_loss = 0.0
        edge = self.get_edge_from_mask(mask)  # shape: [B,1,H,W]

        for i, pred in enumerate(preds):
            if i == self.edge_indices:
                focal = self.focal(pred, edge)
                dice = self.dice_loss(pred, edge)
                loss = self.edge_weight * (focal + self.dice_weight * dice)
            else:
                focal = self.focal(pred, mask)
                dice = self.dice_loss(pred, mask)
                loss = focal + self.dice_weight * dice

            total_loss += loss

        return total_loss





# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.7, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-bce)
#         loss = self.alpha * (1 - pt) ** self.gamma * bce
#         return loss.mean() if self.reduction == 'mean' else loss.sum()
    

# class HybridLoss(nn.Module):
#     """
#     Focal + Dice (无边缘项)
#     """
#     def __init__(self, alpha=0.7, gamma=2.0):
#         super().__init__()
#         self.focal = FocalLoss(alpha=alpha, gamma=gamma)
#         self.dice_weight = 1.0 - alpha

#     def dice_loss(self, pred, target):
#         smooth = 1.
#         pred = torch.sigmoid(pred)
#         intersection = (pred * target).sum()
#         union = pred.sum() + target.sum()
#         return 1 - (2 * intersection + smooth) / (union + smooth)

#     def forward(self, pred, mask):
#         focal = self.focal(pred, mask)
#         dice = self.dice_loss(pred, mask)
#         return focal + self.dice_weight * dice


# class MultiScaleLoss(nn.Module):
#     """
#     preds: [P5, P4, P3, P2, P1]
#     """
#     def __init__(self):
#         super().__init__()
#         self.loss_fn = HybridLoss(alpha=0.7, gamma=2.0)
#         self.weights = [0.2, 0.4, 0.6, 0.8, 1.0]

#     def forward(self, preds, gt):
#         total = 0.0
#         for i, p in enumerate(preds):
#             if p.shape[2:] != gt.shape[2:]:
#                 p = F.interpolate(p, size=gt.shape[2:], mode='bilinear', align_corners=False)
#             total += self.weights[i] * self.loss_fn(p, gt)
#         return total


class Solver():
    def __init__(self, module, opt, exp_id, train):
        self.opt = opt
        self.logger = LogWritter(opt)

        self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt)
        # if not train:
        #     self.net.load_state_dict(torch.load(opt.pretrain))
        
        if opt.pretrain:
            state_dict = torch.load(opt.pretrain, map_location=self.dev)
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            
            for k, v in state_dict.items():
                if k in model_dict and v.shape != model_dict[k].shape:
                    print(f"参数 {k} 尺寸不匹配: 预训练 {v.shape} vs 模型 {model_dict[k].shape}")
                    # 跳过尺寸不匹配的参数
                    del pretrained_dict[k]
             
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict) 
            print(f"成功加载 {len(pretrained_dict)}/{len(state_dict)} 个参数")
            missing = set(state_dict.keys()) - set(pretrained_dict.keys())
            if missing:
                print(f"以下预训练参数未使用: {missing}")
        
        self.net = self.net.to(self.dev)

        # 使用带边缘监督的 loss
        # edge_indices=(1, 5)
        self.loss_fn = HybridLoss(alpha=0.7, edge_indices=0, edge_weight=1.0)
        # self.loss_fn = MultiScaleLoss()

        msg = "# params:{}\n".format(sum(map(lambda x: x.numel(), self.net.parameters())))
        print(msg)
        self.logger.update_txt(msg)

        # 参数分组优化器（encoder / 非 encoder）
        base, head = [], []
        for name, param in self.net.named_parameters():
            if "encoder" in name:
                base.append(param)
            else:
                head.append(param)
        assert base != [], 'encoder is empty'
        #self.optim = torch.optim.Adam([{'params': base}, {'params': head}], opt.lr, betas=(0.9, 0.999), eps=1e-8)

        self.optim = torch.optim.Adam([
            {'params': base, 'lr': opt.lr * 0.1},  # 基础网络使用较低学习率
            {'params': head, 'lr': opt.lr}          # 新添加的头部使用正常学习率
        ], lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
        # self.optim = torch.optim.Adam(
        #     self.net.parameters(),
        #     lr=opt.lr,
        #     betas=(0.9, 0.999),
        #     eps=1e-8
        # )

        
        self.exp_id = f'{opt.dataset}_{exp_id}'

        trainset = NPY_datasets(path_Data=opt.dataset_root, train=True)
        validset = NPY_datasets(path_Data=opt.dataset_root, train=False)

        self.train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True,
                                                        pin_memory=True, num_workers=0)

        self.eval_loader = torch.utils.data.DataLoader(dataset=validset, batch_size=1, shuffle=False, pin_memory=True,
                                                       num_workers=0)

        self.best_dice, self.best_step = 0, 0


    def fit(self):
        opt = self.opt

        for step in range(self.opt.max_epoch):
            epoch_start_time = time.time()

            # 动态调整学习率
            power = (step + 1) // opt.decay_step
            self.optim.param_groups[0]['lr'] = opt.lr * 0.1 * (0.5 ** power)  # base
            self.optim.param_groups[1]['lr'] = opt.lr * (0.5 ** power)       # head
            print('LR base: {}, LR head: {}'.format(self.optim.param_groups[0]['lr'],
                                                    self.optim.param_groups[1]['lr']))
            # 动态调整学习率（统一 lr）
            # power = (step + 1) // opt.decay_step
            # lr = opt.lr * (0.5 ** power)

            # for g in self.optim.param_groups:
            #     g['lr'] = lr

            # print(f'LR: {lr}')


            for i, inputs in enumerate(tqdm(self.train_loader)):
                self.optim.zero_grad()

                IMG = inputs[0].to(self.dev, dtype=torch.float)
                MASK = inputs[1].to(self.dev, dtype=torch.float32)

                pred = self.net(IMG)

                # 自动处理边缘 supervision（pred[2], pred[5]）
                loss = self.loss_fn(pred, MASK)

                if i % 200 == 0:
                    print('iter: {}, loss: {:.4f}'.format(i, loss.item()))

                loss.backward()

                if opt.gclip > 0:
                    torch.nn.utils.clip_grad_value_(self.net.parameters(), opt.gclip)

                self.optim.step()

            print("[{}/{}]".format(step + 1, self.opt.max_epoch))
            self.summary_and_save(step)
            epoch_end_time = time.time()
            print('epoch time: {:.2f}s'.format(epoch_end_time - epoch_start_time))





    def summary_and_save(self, step):
        print('evaluate...')
        dice = self.evaluate()
        if dice > self.best_dice:
            self.best_dice = dice
            self.best_step = step
            # 保存模型
            self.save()

        print('epoch: {}, best_dice: {}, best_step: {}'.format(step, self.best_dice, self.best_step))

    @torch.no_grad()
    def evaluate(self):
        self.net.eval()
        # ---- 统计 FLOPs 和参数 ----
        from thop import profile
        dummy_input = torch.randn(1, 3, 256, 256).to(self.dev)
        flops, params = profile(self.net, inputs=(dummy_input,))
        print(f"FLOPs: {flops / 1e9:.2f} G")
        print(f"Params: {params / 1e6:.2f} M")
# -------------------------

        gts = []
        preds = []

        for i, inputs in enumerate(tqdm(self.eval_loader)):
            IMG = inputs[0].to(self.dev, dtype=torch.float)
            MASK = inputs[1].to(self.dev, dtype=torch.float32)

            b, c, h, w = MASK.shape

            pred = self.net(IMG)

            pred_sal = pred[-1]
            pred_sal = torch.sigmoid(pred_sal)
            gts.append(MASK.squeeze(1).cpu().detach().numpy())
            preds.append(pred_sal.squeeze(1).cpu().detach().numpy())

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= 0.5, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)

        # 处理无正样本的情况
        if confusion.shape == (1, 1):
            # 只有一种类别（全负样本）
            TN = confusion[0, 0]
            FP = FN = TP = 0
        else:
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = (TN + TP) / np.sum(confusion) if np.sum(confusion) != 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        DSC = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
        miou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

        print('accuracy: {:.4f}, sensitivity: {:.4f}, specificity: {:.4f}, DSC: {:.4f}, miou: {:.4f}'.format(
            accuracy, sensitivity, specificity, DSC, miou))

        self.net.train()
        return DSC
        


    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage, strict=False)
        self.net.load_state_dict(state_dict)
        return

    def save(self):
        path = os.path.join(self.opt.ckpt_root, self.exp_id)
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, "best.pt")
        torch.save(self.net.state_dict(), save_path)




# import os
# import time
# import numpy as np
# import torch
# import torch.nn.functional as F
# from sklearn.metrics import confusion_matrix
# from dataloder.dataset1718 import NPY_datasets
# from utils import LogWritter
# # from loss_fn import ConfidentLoss
# from tqdm import tqdm
# import torch.nn as nn
# from sklearn.metrics import confusion_matrix

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-bce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         return focal_loss

# class HybridLoss(nn.Module):
#     def __init__(self, alpha=0.7, gamma=2.0):
#         super().__init__()
#         self.focal = FocalLoss(alpha=alpha, gamma=gamma)
#         self.dice_weight = 1.0 - alpha

#     def dice_loss(self, pred, target):
#         smooth = 1.
#         pred = torch.sigmoid(pred)
#         intersection = (pred * target).sum()
#         union = pred.sum() + target.sum()
#         return 1.0 - (2. * intersection + smooth) / (union + smooth)

#     def forward(self, pred, target):
#         focal = self.focal(pred, target)
#         dice = self.dice_loss(pred, target)
#         return focal + self.dice_weight * dice

# class Solver():
#     def __init__(self, module, opt, exp_id, train):
#         self.opt = opt
#         self.logger = LogWritter(opt)
#         self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
#         self.net = module.Net(opt)
        
#         if not train:
#             self.net.load_state_dict(torch.load(opt.pretrain))
#         self.net = self.net.to(self.dev)
#         self.loss_fn = HybridLoss(alpha=0.7)

#         msg = "# params:{}\n".format(sum(map(lambda x: x.numel(), self.net.parameters())))
#         print(msg)
#         self.logger.update_txt(msg)

#         base, head = [], []
#         for name, param in self.net.named_parameters():
#             if "encoder" in name:
#                 base.append(param)
#             else:
#                 head.append(param)
#         self.optim = torch.optim.Adam([{'params': base}, {'params': head}], opt.lr, betas=(0.9, 0.999), eps=1e-8)

#         self.exp_id = f'{opt.dataset}_{exp_id}'

#         # 训练集加载器
#         trainset = NPY_datasets(path_Data=opt.dataset_root, train=True)
#         self.train_loader = torch.utils.data.DataLoader(
#             dataset=trainset, 
#             batch_size=opt.batch_size, 
#             shuffle=True,
#             pin_memory=True, 
#             num_workers=0
#         )

#         # 创建多个测试集加载器
#         self.eval_loaders = {}
#         val_dir = os.path.join(opt.dataset_root, 'val')
        
#         # 获取所有测试集名称
#         test_sets = [d for d in os.listdir(val_dir) 
#                     if os.path.isdir(os.path.join(val_dir, d)) 
#                     and d != 'images' and d != 'masks']  # 排除可能的通用目录
        
#         print(f"发现测试集: {test_sets}")
        
#         for test_set in test_sets:
#             testset = NPY_datasets(
#                 path_Data=opt.dataset_root, 
#                 train=False, 
#                 test_set_name=test_set
#             )
#             self.eval_loaders[test_set] = torch.utils.data.DataLoader(
#                 dataset=testset, 
#                 batch_size=1, 
#                 shuffle=False,
#                 pin_memory=True, 
#                 num_workers=0
#             )

#         self.best_dice, self.best_step = 0, 0
#         self.best_metrics = {}  # 存储每个测试集的最佳指标

#     def fit(self):
#         opt = self.opt

#         for step in range(self.opt.max_epoch):
#             epoch_start_time = time.time()
#             #  assign different learning rate
#             power = (step + 1) // opt.decay_step
#             self.optim.param_groups[0]['lr'] = opt.lr * 0.1 * (0.5 ** power)  # for base
#             self.optim.param_groups[1]['lr'] = opt.lr * (0.5 ** power)  # for head
#             print('LR base: {}, LR head: {}'.format(self.optim.param_groups[0]['lr'],
#                                                     self.optim.param_groups[1]['lr']))

#             for i, inputs in enumerate(tqdm(self.train_loader)):
#                 self.optim.zero_grad()

#                 IMG = inputs[0].to(self.dev, dtype=torch.float)
#                 MASK = inputs[1].to(self.dev, dtype=torch.float32)

#                 pred = self.net(IMG)
#                 loss1 = self.loss_fn(pred[0], MASK)
#                 loss2 = self.loss_fn(pred[1], MASK)
#                 loss3 = self.loss_fn(pred[2], MASK)
#                 loss4 = self.loss_fn(pred[3], MASK)
#                 loss5 = self.loss_fn(pred[4], MASK)
#                 loss6 = self.loss_fn(pred[5], MASK)

#                 loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

#                 if i % 200 == 0:
#                     print('iter: {}, loss: {}'.format(i, loss.item()))

#                 loss.backward()

#                 if opt.gclip > 0:
#                     torch.nn.utils.clip_grad_value_(self.net.parameters(), opt.gclip)

#                 self.optim.step()
#             # eval
#             print("[{}/{}]".format(step + 1, self.opt.max_epoch))
#             self.summary_and_save(step)
#             epoch_end_time = time.time()
#             print('epoch time: {}'.format(epoch_end_time - epoch_start_time))

#     def summary_and_save(self, step):
#         print('开始评估所有测试集...')
#         overall_dice = 0
#         test_set_count = len(self.eval_loaders)
        
#         results = {}
#         for test_name, loader in self.eval_loaders.items():
#             print(f"\n评估测试集: {test_name}")
#             metrics = self.evaluate(loader)
#             results[test_name] = metrics
#             overall_dice += metrics['DSC']
            
#             # 记录最佳指标
#             if test_name not in self.best_metrics or metrics['DSC'] > self.best_metrics[test_name]['DSC']:
#                 self.best_metrics[test_name] = metrics
#                 self.best_metrics[test_name]['epoch'] = step
        
#         # 计算平均DSC
#         avg_dice = overall_dice / test_set_count if test_set_count > 0 else 0
        
#         # 打印当前epoch结果
#         print(f"\nEpoch {step} 结果:")
#         for test_name, metrics in results.items():
#             print(f"{test_name}: DSC={metrics['DSC']:.4f}, "
#                   f"Best DSC={self.best_metrics[test_name]['DSC']:.4f} (epoch={self.best_metrics[test_name]['epoch']})")
        
#         # 使用平均DSC决定是否保存模型
#         if avg_dice > self.best_dice:
#             self.best_dice = avg_dice
#             self.best_step = step
#             self.save()
#             print(f"保存模型! 平均DSC: {avg_dice:.4f}")
        
#         print(f'Epoch: {step}, 最佳平均DSC: {self.best_dice:.4f}, 最佳epoch: {self.best_step}')

#     @torch.no_grad()
#     def evaluate(self, eval_loader):
#         """评估指定数据加载器并返回指标字典"""
#         self.net.eval()
        
#         gts = []
#         preds = []
#         img_names = []

#         for i, inputs in enumerate(tqdm(eval_loader)):
#             IMG = inputs[0].to(self.dev, dtype=torch.float)
#             MASK = inputs[1].to(self.dev, dtype=torch.float32)
            
#             # 如果返回了图像名称
#             if len(inputs) > 2:
#                 img_names.extend(inputs[2])

#             b, c, h, w = MASK.shape

#             pred = self.net(IMG)
#             pred_sal = pred[-2]  # 使用倒数第二个输出
#             pred_sal = torch.sigmoid(pred_sal)
            
#             gts.append(MASK.squeeze(1).cpu().detach().numpy())
#             preds.append(pred_sal.squeeze(1).cpu().detach().numpy())

#         preds = np.concatenate(preds, axis=0).reshape(-1)
#         gts = np.concatenate(gts, axis=0).reshape(-1)

#         y_pre = np.where(preds >= 0.5, 1, 0)
#         y_true = np.where(gts >= 0.5, 1, 0)

#         confusion = confusion_matrix(y_true, y_pre)
        
#         # 处理可能的单类别情况
#         if confusion.shape == (1, 1):
#             TN = confusion[0, 0]
#             FP, FN, TP = 0, 0, 0
#         else:
#             TN, FP, FN, TP = confusion.ravel()

#         total = TN + FP + FN + TP
#         accuracy = (TN + TP) / total if total > 0 else 0
#         sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
#         specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
#         DSC = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
#         iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

#         print(f'准确率: {accuracy:.4f}, 敏感度: {sensitivity:.4f}, 特异性: {specificity:.4f}, DSC: {DSC:.4f}, IoU: {iou:.4f}')

#         self.net.train()
        
#         return {
#             'accuracy': accuracy,
#             'sensitivity': sensitivity,
#             'specificity': specificity,
#             'DSC': DSC,
#             'iou': iou,
#             'img_names': img_names,
#             'preds': preds,
#             'gts': gts
#         }



#     def load(self, path):
#         state_dict = torch.load(path, map_location=lambda storage, loc: storage)
#         self.net.load_state_dict(state_dict)
#         return

#     def save(self):
#         path = os.path.join(self.opt.ckpt_root, self.exp_id)
#         os.makedirs(path, exist_ok=True)
#         save_path = os.path.join(path, "best.pt")
#         torch.save(self.net.state_dict(), save_path)






# import os
# import time
# import numpy as np
# import torch
# import torch.nn.functional as F
# from sklearn.metrics import confusion_matrix
# from dataloder.dataset1718 import NPY_datasets
# from utils import LogWritter
# # from loss_fn import ConfidentLoss
# from tqdm import tqdm
# import torch.nn as nn
# from sklearn.metrics import confusion_matrix
# import logging  # 添加日志模块
# import sys
# import pandas as pd  # 确保开头导入了
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-bce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         return focal_loss

# class HybridLoss(nn.Module):
#     def __init__(self, alpha=0.7, gamma=2.0):
#         super().__init__()
#         self.focal = FocalLoss(alpha=alpha, gamma=gamma)
#         self.dice_weight = 1.0 - alpha

#     def dice_loss(self, pred, target):
#         smooth = 1.
#         pred = torch.sigmoid(pred)
#         intersection = (pred * target).sum()
#         union = pred.sum() + target.sum()
#         return 1.0 - (2. * intersection + smooth) / (union + smooth)

#     def forward(self, pred, target):
#         focal = self.focal(pred, target)
#         dice = self.dice_loss(pred, target)
#         return focal + self.dice_weight * dice

# class Solver():
#     def __init__(self, module, opt, exp_id, train):
#         self.opt = opt
#         self.logger = LogWritter(opt)
#         self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
#         self.net = module.Net(opt)
        
#         if not train:
#             self.net.load_state_dict(torch.load(opt.pretrain))
#         self.net = self.net.to(self.dev)
#         self.loss_fn = HybridLoss(alpha=0.7)

#         # 设置日志
#         self.exp_id = f'{opt.dataset}_{exp_id}'
#         log_dir = os.path.join(opt.ckpt_root, self.exp_id)
#         os.makedirs(log_dir, exist_ok=True)
#         log_file = os.path.join(log_dir, 'training.log')
        
#         # 配置日志记录器
#         self.logging = logging.getLogger('training')
#         self.logging.setLevel(logging.INFO)
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
#         # 文件处理器 - 保存所有日志
#         file_handler = logging.FileHandler(log_file, mode='w')
#         file_handler.setFormatter(formatter)
#         self.logging.addHandler(file_handler)
        
#         # 控制台处理器 - 只显示INFO及以上级别
#         console_handler = logging.StreamHandler(sys.stdout)
#         console_handler.setFormatter(formatter)
#         self.logging.addHandler(console_handler)
        
#         # 记录初始信息
#         self.logging.info(f"实验ID: {self.exp_id}")
#         self.logging.info(f"设备: {self.dev}")
#         self.logging.info(f"训练模式: {train}")
        
#         msg = "# params:{}\n".format(sum(map(lambda x: x.numel(), self.net.parameters())))
#         self.logging.info(msg)
#         self.logger.update_txt(msg)

#         base, head = [], []
#         for name, param in self.net.named_parameters():
#             if "encoder" in name:
#                 base.append(param)
#             else:
#                 head.append(param)
#         self.optim = torch.optim.Adam([{'params': base}, {'params': head}], opt.lr, betas=(0.9, 0.999), eps=1e-8)

#         # 训练集加载器
#         trainset = NPY_datasets(path_Data=opt.dataset_root, train=True)
#         self.train_loader = torch.utils.data.DataLoader(
#             dataset=trainset, 
#             batch_size=opt.batch_size, 
#             shuffle=True,
#             pin_memory=True, 
#             num_workers=0
#         )
#         self.logging.info(f"训练集大小: {len(trainset)}")

#         # 创建多个测试集加载器
#         self.eval_loaders = {}
#         val_dir = os.path.join(opt.dataset_root, 'val')
        
#         # 获取所有测试集名称
#         test_sets = [d for d in os.listdir(val_dir) 
#                     if os.path.isdir(os.path.join(val_dir, d)) 
#                     and d != 'images' and d != 'masks']  # 排除可能的通用目录
        
#         self.logging.info(f"发现测试集: {test_sets}")
        
#         for test_set in test_sets:
#             testset = NPY_datasets(
#                 path_Data=opt.dataset_root, 
#                 train=False, 
#                 test_set_name=test_set
#             )
#             self.eval_loaders[test_set] = torch.utils.data.DataLoader(
#                 dataset=testset, 
#                 batch_size=1, 
#                 shuffle=False,
#                 pin_memory=True, 
#                 num_workers=0
#             )
#             self.logging.info(f"测试集 '{test_set}' 大小: {len(testset)}")

#         self.best_dice, self.best_step = 0, 0
#         self.best_metrics = {}  # 存储每个测试集的最佳指标

#     def fit(self):
#         opt = self.opt
#         self.logging.info(f"开始训练，最大周期数: {opt.max_epoch}")

#         for step in range(self.opt.max_epoch):
#             epoch_start_time = time.time()
#             #  assign different learning rate
#             power = (step + 1) // opt.decay_step
#             self.optim.param_groups[0]['lr'] = opt.lr * 0.1 * (0.5 ** power)  # for base
#             self.optim.param_groups[1]['lr'] = opt.lr * (0.5 ** power)  # for head
#             self.logging.info('LR base: {}, LR head: {}'.format(self.optim.param_groups[0]['lr'],
#                                                     self.optim.param_groups[1]['lr']))

#             for i, inputs in enumerate(tqdm(self.train_loader, desc=f'Epoch {step+1}/{opt.max_epoch}')):
#                 self.optim.zero_grad()

#                 IMG = inputs[0].to(self.dev, dtype=torch.float)
#                 MASK = inputs[1].to(self.dev, dtype=torch.float32)

#                 pred = self.net(IMG)
#                 loss1 = self.loss_fn(pred[0], MASK)
#                 loss2 = self.loss_fn(pred[1], MASK)
#                 loss3 = self.loss_fn(pred[2], MASK)
#                 loss4 = self.loss_fn(pred[3], MASK)
#                 loss5 = self.loss_fn(pred[4], MASK)
#                 loss6 = self.loss_fn(pred[5], MASK)

#                 loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

#                 if i % 200 == 0:
#                     self.logging.info('iter: {}, loss: {}'.format(i, loss.item()))

#                 loss.backward()

#                 if opt.gclip > 0:
#                     torch.nn.utils.clip_grad_value_(self.net.parameters(), opt.gclip)

#                 self.optim.step()
#             # eval
#             self.logging.info("[{}/{}]".format(step + 1, self.opt.max_epoch))
#             self.summary_and_save(step)
#             epoch_end_time = time.time()
#             epoch_time = epoch_end_time - epoch_start_time
#             self.logging.info('epoch time: {:.2f}秒 ({:.2f}分钟)'.format(epoch_time, epoch_time/60))

#     def summary_and_save(self, step):
#         self.logging.info('开始评估所有测试集...')
#         overall_dice = 0
#         test_set_count = len(self.eval_loaders)
        
#         results = {}
#         for test_name, loader in self.eval_loaders.items():
#             self.logging.info(f"\n评估测试集: {test_name}")
#             metrics = self.evaluate(loader)
#             results[test_name] = metrics
#             overall_dice += metrics['DSC']
            
#             # 记录最佳指标
#             if test_name not in self.best_metrics or metrics['DSC'] > self.best_metrics[test_name]['DSC']:
#                 self.best_metrics[test_name] = metrics
#                 self.best_metrics[test_name]['epoch'] = step
#                 self.logging.info(f"测试集 '{test_name}' 新的最佳DSC: {metrics['DSC']:.4f} (epoch {step})")
        
#         # 计算平均DSC
#         avg_dice = overall_dice / test_set_count if test_set_count > 0 else 0
        
#         # 打印当前epoch结果
#         self.logging.info(f"\nEpoch {step} 结果:")
#         for test_name, metrics in results.items():
#             self.logging.info(f"{test_name}: DSC={metrics['DSC']:.4f}, "
#                   f"Best DSC={self.best_metrics[test_name]['DSC']:.4f} (epoch={self.best_metrics[test_name]['epoch']})")
        
#         # 使用平均DSC决定是否保存模型
#         if avg_dice > self.best_dice:
#             self.best_dice = avg_dice
#             self.best_step = step
#             self.save()
#             self.logging.info(f"保存模型! 平均DSC: {avg_dice:.4f}")
#         else:
#             self.logging.info(f"当前平均DSC: {avg_dice:.4f}, 未超过最佳值 {self.best_dice:.4f}")
        
#         self.logging.info(f'Epoch: {step}, 最佳平均DSC: {self.best_dice:.4f}, 最佳epoch: {self.best_step}')

#     @torch.no_grad()
    

#     @torch.no_grad()
#     def evaluate(self, eval_loader, test_name=None, step=None):
#         """评估指定数据加载器并返回指标字典，并保存为CSV（可选）"""
#         self.net.eval()

#         gts = []
#         preds = []
#         img_names = []

#         for inputs in tqdm(eval_loader, desc="评估中"):
#             IMG = inputs[0].to(self.dev, dtype=torch.float)
#             MASK = inputs[1].to(self.dev, dtype=torch.float32)

#             if len(inputs) > 2:
#                 img_names.extend(inputs[2])

#             pred = self.net(IMG)
#             pred_sal = pred[-2]  # 使用倒数第二层
#             pred_sal = torch.sigmoid(pred_sal)

#             gts.append(MASK.cpu().numpy())
#             preds.append(pred_sal.cpu().numpy())

#         # 拼接为大数组
#         gts = np.concatenate(gts, axis=0)
#         preds = np.concatenate(preds, axis=0)

#         assert gts.shape == preds.shape, f"预测和标签维度不一致: {preds.shape} vs {gts.shape}"

#         # Flatten
#         y_true = (gts >= 0.5).astype(np.uint8).reshape(-1)
#         y_pred = (preds >= 0.5).astype(np.uint8).reshape(-1)

#         # 混淆矩阵 (显式 labels)
#         confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
#         if confusion.shape == (2, 2):
#             TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
#         else:
#             TN = FP = FN = TP = 0
#             if confusion.shape == (1, 1):
#                 TN = confusion[0, 0]
#             elif confusion.shape == (1, 2):
#                 TN, FP = confusion[0]
#             elif confusion.shape == (2, 1):
#                 FN, TP = confusion[:, 0]

#         total = TN + FP + FN + TP
#         accuracy = (TP + TN) / total if total > 0 else 0
#         sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
#         specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
#         DSC = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
#         miou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

#         self.logging.info(f'准确率: {accuracy:.4f}, 灵敏度: {sensitivity:.4f}, 特异性: {specificity:.4f}, DSC: {DSC:.4f}, mIoU: {miou:.4f}')

#         self.net.train()

#         # ✅ 可选保存为CSV
#         if test_name and step is not None:
#             csv_dir = os.path.join(self.opt.ckpt_root, self.exp_id, "metrics")
#             os.makedirs(csv_dir, exist_ok=True)
#             csv_path = os.path.join(csv_dir, f"{test_name}.csv")

#             row = {
#                 "epoch": step,
#                 "accuracy": accuracy,
#                 "sensitivity": sensitivity,
#                 "specificity": specificity,
#                 "DSC": DSC,
#                 "miou": miou
#             }

#             # 追加写入（带表头）
#             df = pd.DataFrame([row])
#             if not os.path.exists(csv_path):
#                 df.to_csv(csv_path, index=False)
#             else:
#                 df.to_csv(csv_path, mode='a', header=False, index=False)

#         return {
#             'accuracy': accuracy,
#             'sensitivity': sensitivity,
#             'specificity': specificity,
#             'DSC': DSC,
#             'miou': miou,
#             'img_names': img_names,
#             'preds': preds,
#             'gts': gts
#         }


#     def load(self, path):
#         state_dict = torch.load(path, map_location=lambda storage, loc: storage)
#         self.net.load_state_dict(state_dict)
#         return

#     def save(self):
#         path = os.path.join(self.opt.ckpt_root, self.exp_id)
#         os.makedirs(path, exist_ok=True)
#         save_path = os.path.join(path, "best.pt")
#         torch.save(self.net.state_dict(), save_path)