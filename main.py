import argparse
import logging
import math
import os
import random
import time
import pandas as pd
import numpy as np
import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel, BertConfig, BertForSequenceClassification
from sklearn.model_selection import KFold, train_test_split

from torch.autograd import Variable
# import wandb

from tqdm import tqdm
from model import bert

from data import DATASET_GETTERS
from utils import (AverageMeter, accuracy, create_loss_fn,
                   save_checkpoint, reduce_tensor, model_load_state_dict)

logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='Yahoo1000-lr3e-4',type=str,help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--save-path', default='./f-checkpoint', type=str, help='save path')
parser.add_argument('--dataset', default='Yahoo', type=str,
                    choices=['base'], help='dataset name')
parser.add_argument('--num_labeled', type=int, default=1000, help='number of labeled data')
parser.add_argument('--num_unlabeled', type=int, default=20000, help='number of unlabeled data')
parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
parser.add_argument('--total-steps', default=2000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=100, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
parser.add_argument('--num_classes', default=5, type=int, help='number of classes')
parser.add_argument('--resize', default=32, type=int, help='resize image')
parser.add_argument('--batch-size', default=32, type=int, help='train batch size')
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default= 0.0001, type=float, help='train learning late')
parser.add_argument('--student_lr', default= 0.0001, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight-decay', default= 0, type=float, help='train weight decay')
parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=0., type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
parser.add_argument('--evaluate', default=False, action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true',
                    help='only finetune model on labeled dataset')
parser.add_argument('--finetune-epochs', default=125, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=16, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=1e-5, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=2333, type=int, help='seed for initializing training')
parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default= -1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default= -1,
                    help="For distributed training: clocal_rank")
parser.add_argument('--max_len', default=64, type=int, help='text_len')
parser.add_argument('--model', default='bert',type=str,help='model name')
parser.add_argument('--mode', default='train',type=str,help='mode name')
parser.add_argument("--gpu_ids", type=list, default= [0], help="gpu-ids")
parser.add_argument('--drop', default=0.7, type=float, help='SGD Momentum')

parser.add_argument("--gpu", type=str, default= '5',help="gpu")



def setup_seed(seed):
   torch.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True



def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    #
    # return LambdaLR(optimizer, lr_lambda = lambda epoch:0.99**epoch)
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_loop(args, labeled_loader, unlabeled_loader, dev_loader,test_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):
    criteria = nn.KLDivLoss()
    torch.cuda.empty_cache()
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"   Total steps = {args.total_steps}")

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    moving_dot_product = torch.empty(1).to(args.device)
    limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    nn.init.uniform_(moving_dot_product, -limit, limit)

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            text_l, targets = labeled_iter.next()

        except:
            if args.world_size > 1:
                labeled_epoch = labeled_epoch+1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            text_l, targets = labeled_iter.next()

        try:
            text_uw, text_us, _ = unlabeled_iter.next()

        except:
            if args.world_size > 1:
                unlabeled_epoch = unlabeled_epoch+1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            text_uw, text_us, _ = unlabeled_iter.next()

        data_time.update(time.time() - end)

        text_l = text_l.to(args.device)
        text_uw = text_uw.to(args.device)
        text_us = text_us.to(args.device)
        targets = targets.to(args.device)
        with amp.autocast(enabled=args.amp):

            batch_size = text_l.shape[0]
            t_text = torch.cat((text_l, text_uw, text_us))
            t_logits = teacher_model(t_text)

            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)

            # t_logits_lf = t_logitsf[:batch_size]
            # t_logits_uwf, t_logits_usf = t_logitsf[batch_size:].chunk(2)

            #del t_logits

            t_loss_l = criterion(t_logits_l, targets)


            soft_pseudo_label = torch.softmax(t_logits_uw.detach()/args.temperature, dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            ######################################################################################
            # kl3
            # t_loss_simi3 = criteria(torch.log_softmax(t_logits_usf,dim = -1) , torch.softmax(t_logits_lf , dim = -1))
            # t_loss_l = t_loss_l + t_loss_simi3
            ######################################################################################





            ######################################################################################
            #kl1
            mask = max_probs.ge(args.threshold).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )

            weight_u = args.lambda_u * min(1., (step+1) / args.uda_steps)
            # t_loss_uda = t_loss_l + weight_u * t_loss_u
            ts_loss_uda = t_loss_l + weight_u * t_loss_u#t_loss_uda
            ######################################################################################
            # t_loss_simi1 = criteria(torch.log_softmax(t_logits_us,dim = -1) , torch.softmax(t_logits_uw, dim = -1))
            # ts_loss_uda = t_loss_l + t_loss_simi1
            ######################################################################################







            s_text = torch.cat((text_l, text_us))
            s_logits = student_model(s_text)
            s_logits_l = s_logits[:batch_size]
            # s_logits_lf = s_logitsf[:batch_size]
            # print("@@@@@@@@old",s_logits_l)
            s_logits_us = s_logits[batch_size:]
            # s_logits_usf = s_logitsf[batch_size:]
            #del s_logits
            s_loss_l_old = F.cross_entropy(s_logits_l, targets)
            st_loss_l_old = s_loss_l_old.detach()

            ######################################################################################
            #kl3
            # s_loss_simi3 = criteria(torch.log_softmax(s_logits_usf,dim = -1) , torch.softmax(s_logits_lf.detach() , dim = -1))
            # s_loss_pseu = criterion(s_logits_us, hard_pseudo_label)
            # s_loss = s_loss_simi3 + s_loss_pseu
            ######################################################################################

            ######################################################################################
            #org
            s_loss = criterion(s_logits_us, hard_pseudo_label)
            ######################################################################################
            # kl2
            # a= torch.log_softmax(s_logits_us , dim = -1)
            # b = torch.softmax(t_logits_us, dim = -1)
            # # b = soft_pseudo_label
            # s_loss_simi2 = criteria(a, b)
            # s_loss = criterion(s_logits_us, hard_pseudo_label)
            # s_loss = s_loss_simi2 + s_loss
            ######################################################################################

            # print("      ")
            # print("````````s_loss```````", s_loss.item())
        s_scaler.scale(s_loss).backward(retain_graph=True)
        # if args.grad_clip > 0:
        #     s_scaler.unscale_(s_optimizer)
        #     nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        if args.ema > 0:
            avg_student_model.update_parameters(student_model)

        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l1 = student_model(text_l)
                # print("@@@@@@@@new",s_logits_l)
            s_loss_l_new = F.cross_entropy(s_logits_l1.detach(), targets)
            # # dot_product = s_loss_l_new - st_loss_l_old
            # # test
            dot_product = st_loss_l_old - s_loss_l_new
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product
            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
            # t_loss_mpl.backward()
            t_loss = ts_loss_uda + t_loss_mpl
            #
            # print("``````teacher`````",t_loss_mpl.item(), t_loss_l.item(), t_loss.item())

        t_scaler.scale(t_loss).backward(retain_graph=True)
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()
        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            # t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            # t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)
            mask = reduce_tensor(mask, args.world_size)

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())

        t_losses_l.update(t_loss_l.item())
        # t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())


        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            # f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f" S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), step)
            # wandb.log({"lr": get_lr(s_optimizer)})

        args.num_eval = step//args.eval_step
        if (step+1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                # args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                # args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                # args.writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
                # args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                # # wandb.log({"train/1.s_loss": s_losses.avg,
                # #            "train/2.t_loss": t_losses.avg,
                # #            "train/3.t_labeled": t_losses_l.avg,
                # #            "train/4.t_unlabeled": t_losses_u.avg,
                # #            "train/5.t_mpl": t_losses_mpl.avg,
                # #            "train/6.mask": mean_mask.avg})

                test_model = avg_student_model if avg_student_model is not None else student_model

                # test_loss, top1, top5 = evaluate(args, dev_loader, test_model, criterion)

                test_loss, top1 = evaluate(args, dev_loader, test_model, criterion)
                # report, confusion,mif1,maf1 = inference_fn(args,test_model,dev_loader)
                # print(mif1,maf1)
                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/acc@1", top1, args.num_eval)
                # args.writer.add_scalar("test/acc@5", top5, args.num_eval)
                # wandb.log({"test/loss": test_loss,
                #            "test/acc@1": top1,
                #            "test/acc@5": top5})

                is_best = top1 > args.best_top1
                if is_best:
                    args.best_top1 = top1
                    # args.best_top5 = top5

                logger.info(f"top-1 acc: {top1:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_top1': args.best_top1,
                    # 'best_top5': args.best_top5,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)

    if args.local_rank in [-1, 0]:
        args.writer.add_scalar("f-result/test_acc@1", args.best_top1)
        # wandb.log({"result/test_acc@1": args.best_top1})
    # finetune
    # del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
    # del s_scaler, s_scheduler, s_optimizer
    # ckpt_name = f'{args.save_path}/{args.name}_best.pth.tar'
    # loc = f'cuda:{args.gpu}'
    # checkpoint = torch.load(ckpt_name, map_location=loc)
    # logger.info(f"=> loading checkpoint '{ckpt_name}'")
    # if checkpoint['avg_state_dict'] is not None:
    #     model_load_state_dict(student_model, checkpoint['avg_state_dict'])
    # else:
    #     model_load_state_dict(student_model, checkpoint['student_state_dict'])
    # finetune(args, labeled_loader, dev_loader, student_model, criterion)
    return


def evaluate(args, dev_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    test_iter = tqdm(dev_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        # lbs = []
        for step, (text, targets) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = text.shape[0]
            text = text.to(args.device)
            targets = targets.to(args.device)
            target = targets.detach().cpu().numpy().tolist()
            with amp.autocast(enabled=args.amp):
                outputs = model(text)

                outputs = F.softmax(outputs, dim=1)
                op = outputs.detach().cpu().numpy().tolist()
                for output in op:

                    lb_value = max(output)
                    #
                    # lbs.append(output.index(lb_value))

                loss = criterion(outputs, targets)

            # acc1, acc5 = accuracy(outputs, targets, (1, 5))
            acc1 = accuracy(outputs, targets)

            losses.update(loss.item(), batch_size)



            top1.update(acc1[0].item(), batch_size)
            # top5.update(acc5[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(dev_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"top1: {top1.avg:.2f}")

            test_iter.close()
        # csv_df = pd.DataFrame(lbs,columns = ["label"])
        #
        # csv_df.to_csv("out_csv.csv",columns = ["label"])  # 追加后生成的文件为out_csv

        # return losses.avg, top1.avg, top5.avg, lbs
        return losses.avg, top1.avg


def inference_fn(args,model, dataloader):
    model.eval()
    preds = []
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for data in dataloader:
        inputs, targets = data[0].to(args.device), data[1].to(args.device)
        with torch.no_grad():
            outputs = model(inputs)
            labels = targets.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()


            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
        # preds.append(outputs.sigmoid().detach().cpu().numpy())
    mif1 = f1_score(labels_all, predict_all, average='micro')  # 调用并输出计算的值
    maf1 = f1_score(labels_all, predict_all, average='macro')
    report = metrics.classification_report(labels_all, predict_all, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return report, confusion,mif1,maf1

def finetune(args, train_loader, dev_loader, model, criterion):
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        train_loader.dataset,
        sampler=train_sampler(train_loader.dataset),
        batch_size=args.finetune_batch_size,
        num_workers=args.workers,
        pin_memory=True)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.finetune_lr,
                          momentum=args.finetune_momentum,
                          weight_decay=args.finetune_weight_decay)
    scaler = amp.GradScaler(enabled=args.amp)

    logger.info("***** Running Finetuning *****")
    logger.info(f"   Finetuning steps = {len(labeled_loader)*args.finetune_epochs}")

    for epoch in range(args.finetune_epochs):
        if args.world_size > 1:
            labeled_loader.sampler.set_epoch(epoch+624)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        labeled_iter = tqdm(labeled_loader, disable=args.local_rank not in [-1, 0])
        for step, (text, targets) in enumerate(labeled_iter):
            data_time.update(time.time() - end)
            batch_size = text.shape[0]
            text = text.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                model.zero_grad()
                outputs = model(text).to(args.device)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()

            if args.world_size > 1:
                loss = reduce_tensor(loss.detach(), args.world_size)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch+1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)
            test_loss, top1, top5 = evaluate(args, dev_loader, model, criterion)
            # test_loss, top1 = evaluate(args, dev_loader, model, criterion)
            args.writer.add_scalar("finetune/test_loss", test_loss, epoch)
            args.writer.add_scalar("finetune/acc@1", top1, epoch)
            args.writer.add_scalar("finetune/acc@5", top5, epoch)
            # wandb.log({"finetune/train_loss": losses.avg,
            #            "finetune/test_loss": test_loss,
            #            "finetune/acc@1": top1,
            #            "finetune/acc@5": top5})

            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1
                args.best_top5 = top5

            logger.info(f"top-1 acc: {top1:.2f}")
            logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

            save_checkpoint(args, {
                'step': step + 1,
                'best_top1': args.best_top1,
                'best_top5': args.best_top5,
                'student_state_dict': model.state_dict(),
                'avg_state_dict': None,
                'student_optimizer': optimizer.state_dict(),
            }, is_best, finetune=True)
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("f-result/finetune_acc@1", args.best_top1)
            # wandb.log({"result/fintune_acc@1": args.best_top1})
    return

def create_model(args):
    model = bert.Model(args)
    # model = BertForSequenceClassification.from_pretrained("bert-base-cased",num_labels = 4)

    def weights_init(m):
        classname = m.__class__.__name__

    #     if classname.find('Conv1d') != -1:
    #         m.weight.data.normal_(0.0, 0.02)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0)
    #     elif classname.find('Conv2d') != -1:
    #         m.weight.data.normal_(0.0, 0.02)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0)
    #     elif classname.find('BatchNorm') != -1:
    #         m.weight.data.normal_(1.0, 0.02)
    #         m.bias.data.fill_(0)
    #     elif classname.find('Linear') != -1:
    #         m.weight.data.normal_(0, 0.02)
    #         m.bias.data.fill_(0)
    # model.apply(weights_init)
    return model
def main():
    args = parser.parse_args()
    args.best_top1 = 0.
    args.best_top5 = 0.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.dataset == "Yelp":
        args.batch_size = 4
        args.max_len = 256
        if args.num_labeled == 100:
            args.total_steps = 2000
        elif args.num_labeled == 1000:
            args.total_steps = 15000
    if args.dataset == "Yahoo":
        args.batch_size = 4
        args.max_len = 256
        args.num_classes = 10
        args.num_unlabeled = 40000
        if args.num_labeled == 100:
            args.total_steps = 5000

        elif args.num_labeled == 1000:
            args.total_steps = 15000
            args.teacher_lr = 0.00007
            args.student_lr = 0.00007
        elif args.num_labeled == 10000:
            args.total_steps = 30000
            args.teacher_lr = 0.0001
            args.student_lr = 0.0001



    if args.local_rank != -1:
        args.gpu = args.local_rank
        torch.distributed.init_process_group(backend='nccl',world_size = 4,rank = args.local_rank )
        args.world_size = torch.distributed.get_world_size()
    else:

        args.world_size = 1

    # args.device = torch.device("cuda" % args.gpu_ids[0] if torch.cuda.is_available() else "cpu")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        args.writer = SummaryWriter(f"f-result/{args.name}")
        # wandb.init(name=args.name, project='MPL', config=args)

    if args.seed is not None:
        setup_seed(args.seed)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.evaluate == False:
        labeled_dataset, unlabeled_dataset, dev_dataset,test_dataset = DATASET_GETTERS[args.dataset](args)




        if args.local_rank == 0:
            torch.distributed.barrier()

        train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
        labeled_loader = DataLoader(
            labeled_dataset,
            sampler=train_sampler(labeled_dataset),
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True)

        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            sampler=train_sampler(unlabeled_dataset),
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True)

        dev_loader = DataLoader(dev_dataset,
                                 sampler=SequentialSampler(dev_dataset),
                                 batch_size=args.batch_size,
                                 num_workers=args.workers)
    else:
        test_dataset = DATASET_GETTERS[args.dataset](args)
    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.workers)


    # if args.dataset == "cifar10":
    #     depth, widen_factor = 28, 2
    # elif args.dataset == 'cifar100':
    #     depth, widen_factor = 28, 8
    #
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()

    teacher_model =create_model(args)
    student_model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    # logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    # logger.info(f"Params: {sum(p.numel() for p in teacher_model. ters())/1e6:.2f}M")

    teacher_model = nn.DataParallel(teacher_model, device_ids=args.gpu_ids)

    teacher_model.to(args.device)
    student_model = nn.DataParallel(student_model,device_ids=args.gpu_ids)
    student_model.to(args.device)
    avg_student_model = None
    criterion = create_loss_fn(args)

    no_decay = ['bn']
    teacher_parameters = [
        {'params': [p for n, p in teacher_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in teacher_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    student_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in student_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

   #optimizer = optim.Adam([{'params':model.encoder.parameters()},{'params':model.fc.parameters(),'lr':0.001}], lr=learning_rate, weight_decay=weight_decay)
    # t_optimizer = optim.SGD(teacher_parameters,
    #                         lr=args.teacher_lr,
    #                         momentum=args.momentum,
    #                         # weight_decay=args.weight_decay,
    #                         nesterov=args.nesterov)
    # s_optimizer = optim.SGD(student_parameters,
    #                         lr=args.student_lr,
    #                         momentum=args.momentum,
    #                         # weight_decay=args.weight_decay,
    #                         nesterov=args.nesterov)
    #不同学习率
    t_optimizer = optim.SGD([{'params':teacher_model.module.encoder.parameters()},{'params':teacher_model.module.fc.parameters(),'lr':0.001}],
                            lr=args.teacher_lr,
                            momentum=args.momentum,
                            # weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    s_optimizer = optim.SGD([{'params':student_model.module.encoder.parameters()},{'params':student_model.module.fc.parameters(),'lr':0.001}],
                            lr=args.student_lr,
                            momentum=args.momentum,
                            # weight_decay=args.weight_decay,
                            nesterov=args.nesterov)

    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # logger.info(f"=> loading checkpoint '{args.gpu}'")
            # loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=args.device)
            # args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            # args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))
            if not (args.evaluate or args.finetune):
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                s_optimizer.load_state_dict(checkpoint['student_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                s_scheduler.load_state_dict(checkpoint['student_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                if avg_student_model is not None:
                    model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])

            else:
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(student_model, checkpoint['avg_state_dict'])
                else:
                    model_load_state_dict(student_model, checkpoint['student_state_dict'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    if args.local_rank != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    if args.finetune:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
        del s_scaler, s_scheduler, s_optimizer
        finetune(args, labeled_loader, dev_loader, student_model, criterion)
        return

    if args.evaluate:
        del t_scaler, t_scheduler, t_optimizer, teacher_model
        del s_scaler, s_scheduler, s_optimizer
        # evaluate(args, dev_loader, teacher_model, criterion)

        report, confusion,mif1,maf1 =inference_fn(args, student_model, test_loader)
        print(report, confusion,mif1,maf1)
        return



    teacher_model.zero_grad()
    student_model.zero_grad()
    train_loop(args, labeled_loader, unlabeled_loader, dev_loader, test_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler)
    report, confusion, mif1, maf1 = inference_fn(args, student_model, test_loader)
    print("@@@@@@@@@@@@@@@@1",report, confusion, mif1, maf1)

    args.resume = f'{args.save_path}/{args.name}_best.pth.tar'
    checkpoint = torch.load(args.resume, map_location=args.device)
    model_load_state_dict(student_model, checkpoint['student_state_dict'])
    report, confusion, mif1, maf1 = inference_fn(args, student_model, test_loader)
    print("@@@@@@@@@@@@@@@@@@@@@2",report, confusion, mif1, maf1)

    return


if __name__ == '__main__':
    main()
