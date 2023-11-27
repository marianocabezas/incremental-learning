# Python imports
import os
import io
import gzip
import shutil
import quadprog
import numpy as np
from copy import deepcopy
from functools import partial

# Pytorch imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Project imports
from base import BaseModel
from datasets import MultiDataset
from memory import MemoryContainer


def update_y(y, mask):
    y = torch.cat(
        [torch.where(mask.to(y.device) == y_i)[0] for y_i in y]
    ).to(y.device)
    return y


class IncrementalModel(BaseModel):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True
    ):
        super().__init__()
        self.init = basemodel.init
        self.best = best
        self.first = True
        self.model = basemodel
        self.device = basemodel.device
        self.memory_manager = memory_manager
        self.lr = lr
        if self.lr is not None:
            self.model.lr = self.lr
            self.reset_optimiser()
        self.task = task
        # Counters
        self.n_classes = n_classes
        self.n_tasks = n_tasks
        self.observed_tasks = []
        self.current_task = -1
        self.task_mask = None
        self.last_step = False
        self.task_masks = []
        self.grams = []
        self.logits = []
        self.cum_grad = []
        for param in self.model.parameters():
            if param.requires_grad:
                self.cum_grad.append(
                    torch.zeros(param.data.numel(), dtype=torch.float32)
                )

        self.train_functions = self.model.train_functions
        self.val_functions = self.model.val_functions

        self.update_logs()

        self.optimizer_alg = self.model.optimizer_alg

    def _update_cum_grad(self, norm):
        p = 0
        for param in self.model.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    flat_grad = param.grad.cpu().detach().data.flatten()
                    self.cum_grad[p] += flat_grad / norm
                p += 1

    def _kl_div_loss(self, mask):
        losses = []
        x = []
        y_logits = []
        for k in mask:
            x_k, y_k = self.memory_manager.get_class(k)
            indx = np.random.permutation(list(range(len(x_k))))[0]
            x.append(x_k[indx].clone())
            y_logits.append(y_k[indx].clone())
        x = torch.stack(x, dim=0).to(self.device)
        y_logits = torch.stack(y_logits, dim=0).to(self.device)

        # Add distillation loss.
        # In the task incremental case, offsets will select the required tasks,
        # while on the class incremental case, offsets will be 0 and the number
        # of classes when passed to the loss. Therefore, we don't need to check for that.
        prediction = F.log_softmax(
            self.model(x)[:, mask], 1
        )
        y = F.softmax(y_logits[:, mask], 1)
        losses.append(
            F.kl_div(
                prediction, y, reduction='batchmean'
            ) * len(mask)
        )
        return losses

    def distillation_loss(self):
        if not self.first and self.memory_manager is not None:
            if self.task:
                losses = []
                for mask in self.task_masks[:-1]:
                    losses += self._kl_div_loss(
                        mask.cpu().detach().numpy()
                    )
            else:
                losses = self._kl_div_loss(
                    torch.cat(self.task_masks[:-1]).cpu().detach().numpy()
                )
        else:
            losses = []
        return sum(losses)

    def prebatch_update(self, batch, batches, x, y):
        if self.task:
            y = self.task_mask[y]
        if self.memory_manager is not None:
            training = self.model.training
            self.model.eval()
            with torch.no_grad():
                self.memory_manager.update_memory(
                    x.cpu(), y.cpu(), self.current_task, self.model
                )
            if training:
                self.model.train()
        self._update_cum_grad(batches)

    def fill_grams(self, batch_size=None):
        if self.memory_manager is not None:
            self.grams = []
            for task_data in self.memory_manager.get_tasks():
                task_loader = DataLoader(
                    task_data, batch_size=batch_size,
                    drop_last=True
                )
                task_grams = []
                for x, _ in task_loader:
                    new_grams = self.model.gram_matrix(x.to(self.device)).cpu()
                    for g in new_grams:
                        task_grams.append(
                            g[torch.triu(torch.ones_like(g)) > 0]
                        )
                self.grams.append(torch.stack(task_grams))

    def fill_logits(self, batch_size=None):
        if self.memory_manager is not None:
            self.logits = []
            for task_data in self.memory_manager.get_tasks():
                task_loader = DataLoader(task_data, batch_size=batch_size)
                task_logits = []
                for x, _ in task_loader:
                    logits = self.model.gram_matrix(x.to(self.device)).cpu()
                    task_logits.append(logits)
                self.logits.append(torch.cat(task_logits))

    def task_inference(self, data, nonbatched=True):
        if nonbatched:
            task = self.current_task
        else:
            task = np.array([self.current_task] * len(data))
        if self.memory_manager is not None:
            if len(self.grams) == 0:
                self.fill_grams()
            with torch.no_grad():
                x_cuda = torch.from_numpy(data).to(self.device)
                if nonbatched:
                    x_cuda = x_cuda.unsqueeze(0)
                torch.cuda.empty_cache()
                grams_tensor = torch.stack(self.grams)
                grams_mean = torch.mean(grams_tensor, dim=1, keepdim=True)
                grams_cov = [torch.cov(g) for g in grams_tensor]
                grams_var = [torch.diag(cov) for cov in grams_cov]
                for var_i in grams_var:
                    var_i[var_i > 0] = 1 / var_i[var_i > 0]
                grams_icov = torch.stack([
                    torch.diag(var_i) for var_i in grams_var
                ])

                new_grams = self.model.gram_matrix(x_cuda)
                grams_mask = torch.triu(torch.ones_like(new_grams)) > 0
                g_tensor = new_grams[grams_mask].view((1, len(new_grams), -1))
                g_norm = g_tensor - grams_mean
                distances = torch.stack([
                    torch.diag((g_i @ icov_i.t()) @ g_i.t())
                    for g_i, icov_i in zip(g_norm, grams_icov)
                ])
                task = torch.argmax(distances, dim=0).numpy()

        return task

    def reset_parameters(self, flat_grads, th=0):
        p = 0
        for param in self.model.parameters():
            if param.requires_grad:
                grad_mask = (flat_grads[p] > th).view(param.data.shape)
                new_param = torch.zeros(
                    torch.sum(grad_mask), dtype=torch.float32
                )
                nn.init.xavier_uniform(new_param)
                param.data[grad_mask].fill_(new_param)
                p += 1

    def reset_optimiser(self, model_params=None):
        super().reset_optimiser(model_params)
        self.model.reset_optimiser(model_params)
        self.optimizer_alg = self.model.optimizer_alg

    def get_state(self):
        net_state = {
            'first': self.first,
            'n_classes': self.n_classes,
            'task': self.current_task,
            'lr': self.lr,
            'task_incremental': self.task,
            'manager': self.memory_manager,
            'cum_grad': self.cum_grad,
            'grams': self.grams,
            'logits': self.logits,
            'state': self.state_dict(),
            'train-log': self.train_log,
            'val-log': self.val_log,
        }
        return net_state

    def save_model(self, net_name):
        net_state = self.get_state()
        torch.save(net_state, net_name)
        if net_name.endswith('.gz'):
            zip_name = net_name
        else:
            zip_name = net_name + '.gz'
        with open(net_name, 'rb') as f_in, gzip.open(zip_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            os.remove(net_name)

    def load_model(self, net_name):
        if net_name.endswith('.gz'):
            zip_name = net_name
        else:
            zip_name = net_name + '.gz'
        with gzip.open(zip_name, 'rb') as f:
            # Use an intermediate buffer
            x = io.BytesIO(f.read())
            net_state = torch.load(x, map_location=torch.device('cpu'))
        self.first = net_state['first']
        self.n_classes = net_state['n_classes']
        self.current_task = net_state['task']
        self.memory_manager = net_state['manager']
        self.cum_grad = net_state['cum_grad']
        self.grams = net_state['grams']
        self.logits = net_state['logits']
        self.lr = net_state['lr']
        if self.lr is not None:
            self.model.lr = self.lr
            self.reset_optimiser()
        self.task = net_state['task_incremental']
        self.train_log = net_state['train-log']
        self.val_log = net_state['val-log']
        self.load_state_dict(net_state['state'])
        return net_state

    def forward(self, *inputs):
        return self.model(*inputs)

    def observe(self, x, y):
        pred_labels, x_cuda, y_cuda = super().observe(x, y)
        if self.task:
            pred_labels = pred_labels[:, self.task_mask]
            y_cuda = update_y(y_cuda, self.task_mask)
        else:
            ignore_mask = torch.from_numpy(
                np.array([
                    idx for idx in range(self.n_classes)
                    if idx not in self.task_mask
                ])
            )

            pred_labels = torch.cat([
                pred_labels[:, self.task_mask],
                pred_labels[:, ignore_mask].detach()
            ], dim=-1)

        return pred_labels, x_cuda, y_cuda

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        task=None,
        task_mask=None,
        last_step=False,
        verbose=True
    ):
        self.task_mask = task_mask
        self.last_step = last_step
        if task is not None:
            self.current_task = task
        if self.current_task not in self.observed_tasks:
            self.observed_tasks.append(self.current_task)
            self.task_masks.append(task_mask)
        super().fit(train_loader, val_loader, epochs, patience, verbose)


class IncrementalModelMemory(IncrementalModel):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, task
        )

    def prebatch_update(self, batch, batches, x, y):
        self._update_cum_grad(batches)

    def batch_update(self, batch, batches, x, y):
        if self.task:
            y = self.task_mask[y]
        if self.memory_manager is not None:
            training = self.model.training
            self.model.eval()
            with torch.no_grad():
                self.memory_manager.update_memory(
                    x.cpu(), y.cpu(), self.current_task, self.model
                )
            if training:
                self.model.train()

    def mini_batch_loop(self, data, train=True, verbose=True):
        if self.memory_manager is not None and self.current_task > 0 and train:
            if self.task_mask is None:
                self.task_mask = torch.cat(self.task_masks)
            memory_sets = list(
                self.memory_manager.get_tasks(self.current_task)
            )
            new_dataset = MultiDataset([data.dataset] + memory_sets)
            data = DataLoader(
                new_dataset, data.batch_size, True,
                num_workers=data.num_workers, drop_last=True
            )
        return super().mini_batch_loop(data, train, verbose)


class Independent(IncrementalModel):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, task
        )
        self.model = nn.ModuleList([deepcopy(basemodel) for _ in range(n_tasks)])
        if self.lr is not None:
            for model in self.model:
                model.lr = self.lr
                model.reset_optimiser()
        self.first = True
        self.device = basemodel.device

    def forward(self, *inputs):
        return self.model[self.current_task](*inputs)

    def reset_optimiser(self, model_params=None):
        pass

    def _update_cum_grad(self, norm):
        pass

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        task=None,
        task_mask=None,
        last_step=False,
        verbose=True
    ):
        if task is None:
            self.optimizer_alg = self.model[self.current_task + 1].optimizer_alg
        else:
            self.optimizer_alg = self.model[task].optimizer_alg
        super().fit(
            train_loader, val_loader, epochs, patience, task, task_mask,
            last_step, verbose
        )
        if (self.current_task + 1) < len(self.model) and last_step:
            self.model[self.current_task + 1].load_state_dict(
                self.model[self.current_task].state_dict()
            )


class EWC(IncrementalModel):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True,
        ewc_weight=1e6, ewc_binary=True, ewc_alpha=None
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, task
        )
        self.ewc_weight = ewc_weight
        self.ewc_binary = ewc_binary
        self.ewc_alpha = ewc_alpha

        self.train_functions = self.model.train_functions + [
            {
                'name': 'ewc',
                'weight': ewc_weight,
                'f': lambda p, t: self.ewc_loss()
            }
        ]
        self.val_functions = self.model.val_functions

        self.update_logs()

        # Gradient tensors
        self.ewc_parameters = {
            n: {
                # 'means': p.data.detach(),
                'means': [],
                # 'fisher': torch.zeros_like(p.data)
                'fisher': []
            }
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.optimizer_alg = self.model.optimizer_alg

    def ewc_loss(self):
        if self.ewc_alpha is None:
            losses = [
                torch.sum(
                    fisher.to(self.device) * (
                            p - means.to(self.device)
                    ) ** 2
                )
                for n, p in self.model.named_parameters()
                for fisher, means in zip(
                    self.ewc_parameters[n]['fisher'],
                    self.ewc_parameters[n]['means']
                )
                if p.requires_grad
            ]
        else:
            if not self.first:
                losses = [
                    torch.sum(
                        self.ewc_parameters[n]['fisher'].to(self.device) * (
                                p - self.ewc_parameters[n]['means'].to(
                                    self.device
                                )
                        ) ** 2
                    )
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                ]
            else:
                losses = []

        return sum(losses)

    def fisher(self, dataloader):
        self.model.eval()
        new_fisher = {
            n: torch.zeros_like(p.data)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        for batch_i, (x, y) in enumerate(dataloader):
            # In case we are training the gradient to zero.
            self.model.zero_grad()

            # First, we do a forward pass through the network.
            if isinstance(x, list) or isinstance(x, tuple):
                x_cuda = tuple(x_i.to(self.device) for x_i in x)
                pred_labels = self(*x_cuda)
            else:
                pred_labels = self(x.to(self.device))

            if self.ewc_binary:
                loss = F.binary_cross_entropy(
                    pred_labels, y.type_as(pred_labels).to(self.device)
                )
            else:
                loss = F.cross_entropy(
                    pred_labels, y.to(self.device)
                )
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grad = p.grad.data.detach() ** 2 / len(dataloader)
                    new_fisher[n] += grad

        if self.ewc_alpha is None:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.ewc_parameters[n]['fisher'].append(
                        new_fisher[n].cpu()
                    )
                    self.ewc_parameters[n]['means'].append(
                        p.data.detach().cpu()
                    )
        else:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.ewc_parameters[n]['means'] = p.data.detach().cpu()
                    if self.first:
                        self.ewc_parameters[n]['fisher'] = new_fisher[n].cpu()
                    else:
                        prev_fisher = self.ewc_parameters[n]['fisher']
                        fisher_t0 = (1 - self.ewc_alpha) * prev_fisher
                        fisher_t1 = self.ewc_alpha * new_fisher[n]
                        self.ewc_parameters[n]['fisher'] = (
                            fisher_t0.to(fisher_t1.device) + fisher_t1
                        )

        self.first = False

    def get_state(self):
        net_state = super().get_state()
        net_state['ewc_param'] = self.ewc_parameters
        return net_state

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.ewc_parameters = net_state['ewc_param']

    def forward(self, *inputs):
        return self.model(*inputs)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        task=None,
        task_mask=None,
        last_step=False,
        verbose=True
    ):
        if self.first:
            for loss_f in self.train_functions:
                if loss_f['name'] == 'ewc':
                    loss_f['weight'] = 0
        super().fit(
            train_loader, val_loader, epochs, patience, task, task_mask,
            last_step, verbose
        )
        if last_step:
            if self.memory_manager is None:
                self.fisher(train_loader)
            else:
                if self.ewc_alpha is None:
                    for n, param in self.ewc_parameters.items():
                        param['fisher'] = []
                        param['means'] = []
                for task in self.observed_tasks:
                    task_memory = self.memory_manager.get_task(self, task)
                    task_loader = DataLoader(
                        task_memory, train_loader.batch_size, drop_last=True
                    )
                    self.fisher(task_loader)

            for loss_f in self.train_functions:
                if loss_f['name'] == 'ewc':
                    loss_f['weight'] = self.ewc_weight

    def epoch_update(self, epochs, loader):
        if self.ewc_alpha is not None:
            self.fisher(loader)
            if self.epoch == 0:
                for loss_f in self.train_functions:
                    if loss_f['name'] == 'ewc':
                        loss_f['weight'] = self.ewc_weight


class GEM(IncrementalModel):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True,
        memory_strength=0.5,
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, task
        )
        self.margin = memory_strength
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        self.memory_data = [[] for _ in range(n_tasks)]
        self.memory_labs = [[] for _ in range(n_tasks)]

    def store_grad(self, tid):
        """
            This stores parameter gradients of past tasks.
            pp: parameters
            grads: gradients
            grad_dims: list with number of parameters per layers
            tid: task id
        """
        # store the gradients
        self.grads[:, tid].fill_(0.0)
        cnt = 0
        for param in self.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                en = sum(self.grad_dims[:cnt + 1])
                self.grads[beg:en, tid].copy_(param.grad.cpu().data.view(-1))
            cnt += 1

    def project(self, gradient, memories, eps=1e-3):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector
        """
        memories_np = np.nan_to_num(
            memories.t().double().cpu().numpy()
        )
        gradient_np = np.nan_to_num(
            gradient.contiguous().view(-1).double().cpu().numpy()
        )
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.margin
        try:
            v = quadprog.solve_qp(P, q, G, h)[0]
            x = np.dot(v, memories_np) + gradient_np
        except ValueError:
            x = gradient
        return torch.Tensor(x).view(-1, 1)

    def update_gradients(self):
        if len(self.observed_tasks) > 1 and self.memory_manager is not None:
            for past_task, task_mask in zip(
                self.observed_tasks[:-1], self.task_masks[:-1]
            ):
                self.zero_grad()

                memories_t, labels_t = self.memory_manager.get_task(past_task)
                output = self(torch.stack(memories_t).to(self.device))
                labels = torch.stack(labels_t).to(self.device)
                if self.task:
                    output = output[:, self.task_mask]
                    labels = update_y(labels, self.task_mask)

                batch_losses = [
                    l_f['weight'] * l_f['f'](output, labels)
                    for l_f in self.train_functions
                ]
                sum(batch_losses).backward()
                self.store_grad(past_task)

    def get_grad(self):
        indx = torch.tensor(self.observed_tasks[:-1], dtype=torch.long)
        return self.grads.index_select(1, indx)

    def overwrite_grad(self, newgrad):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in self.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                en = sum(self.grad_dims[:cnt + 1])
                this_grad = newgrad[beg:en].contiguous().view(
                    param.grad.data.size()
                )
                param.grad.data.copy_(this_grad.to(param.device))
            cnt += 1

    def constraint_check(self):
        t = self.current_task
        self.store_grad(t)
        # Copy the current gradient
        grad_t = self.grads.index_select(1, torch.tensor(t, dtype=torch.long))
        if len(self.observed_tasks) > 1:
            grad = self.get_grad()

            dotp = grad_t.t().to(self.device) @ grad.to(self.device)

            if (dotp < 0).any():
                grad_t = self.project(
                    grad_t, grad
                )
                # Copy gradients back
                self.overwrite_grad(grad_t)
        return grad_t

    def get_state(self):
        net_state = super().get_state()
        net_state['grad_dims'] = self.grad_dims
        net_state['grads'] = self.grads
        net_state['tasks'] = self.observed_tasks
        net_state['masks'] = self.task_masks
        return net_state

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.grad_dims = net_state['grad_dims']
        if type(net_state['grads']) is list:
            self.grads = [grad.cpu() for grad in net_state['grads']]
        else:
            self.grads = net_state['grads'].cpu()
        self.observed_tasks = net_state['tasks']
        self.task_masks = net_state['masks']

        return net_state

    def prebatch_update(self, batch, batches, x, y):
        super().prebatch_update(batch, batches, x, y)
        self.update_gradients()
        self.constraint_check()


class DER(IncrementalModelMemory):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, False
        )
        self.last_features = basemodel.last_features
        self.model = nn.ModuleList([deepcopy(basemodel) for _ in range(n_tasks)])
        self.fc = nn.Linear(
            self.last_features * n_tasks, self.n_classes
        )
        self.task_fc = None
        self.train_functions = self.train_functions = [
            {
                'name': 'xentr',
                'weight': 1,
                'f': self.main_loss
            },
            {
                'name': 'aux',
                'weight': 1,
                'f': self.auxiliary_loss
            },

        ]
        self.val_functions = [
            {
                'name': 'xentr',
                'weight': 1,
                'f': self.main_loss
            }
        ]

        self.update_logs()

        if self.lr is not None:
            for model in self.model:
                model.lr = self.lr
                model.reset_optimiser()
        self.first = True
        self.device = basemodel.device

    @property
    def global_mask(self):
        return torch.cat(self.task_masks).to(self.device)

    def main_loss(self, prediction, target):
        try:
            main_pred, aux_pred = prediction
        except ValueError:
            main_pred = prediction

        return F.cross_entropy(main_pred, update_y(target, self.global_mask))

    def auxiliary_loss(self, prediction, target, task_mask):
        if self.current_task > 0:
            target = torch.cat(
                [
                    torch.where(task_mask.to(target.device) == y_i)[0] + 1
                    if y_i in task_mask
                    else torch.tensor([0], dtype=y_i.dtype, device=y_i.device)
                    for y_i in target
                ]
            ).to(target.device)
            loss = F.cross_entropy(prediction[1], target)
        else:
            loss = torch.tensor(0., device=self.device)

        return loss

    def observe(self, x, y):
        return BaseModel.observe(self, x, y)

    def forward(self, *inputs):
        feature_list = [
            self.model[i].prelogits(*inputs).flatten(1).to(self.device)
            for i in range(self.current_task + 1)
        ]
        features = torch.cat(feature_list, dim=-1).to(self.device)
        n_features = features.shape[1]
        self.fc.to(self.device)
        weight = self.fc.weight[self.global_mask, :n_features].to(self.device)
        if self.fc.bias is not None:
            bias = self.fc.bias[self.global_mask].to(self.device)
        else:
            bias = None

        if self.task_fc is not None:
            self.task_fc.to(self.device)
            prediction = (
                F.linear(features, weight, bias),
                self.task_fc(feature_list[-1])
            )
        else:
            prediction = F.linear(features, weight, bias)
        return prediction

    def inference(self, data, nonbatched=True, task=None):
        # We remove all the training-specific variables
        temp_fc = self.task_fc
        tmp_masks = self.task_masks
        self.task_masks = [
            torch.stack([
                torch.tensor(i, dtype=torch.long)
                for i in range(self.n_classes)
            ])
        ]
        self.task_fc = None
        results = super().inference(data, nonbatched, task)
        # We restore the object to keep training-specific variables
        self.task_fc = temp_fc
        self.task_masks = tmp_masks
        return results

    def reset_optimiser(self, model_params=None):
        pass

    def _update_cum_grad(self, norm):
        pass

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        task=None,
        task_mask=None,
        last_step=False,
        verbose=True
    ):
        # 1) Representation learning stage
        if task not in self.observed_tasks:
            self.optimizer_alg = self.model[task].optimizer_alg
            n_classes = len(task_mask)
            self.task_fc = nn.Linear(
                self.last_features, n_classes + 1
            )
            self.train_functions = [
                {
                    'name': 'xentr',
                    'weight': 1,
                    'f': self.main_loss
                },
                {
                    'name': 'aux',
                    'weight': 1,
                    'f': partial(self.auxiliary_loss, task_mask=task_mask)
                },
            ]
            self.val_functions = [
                {
                    'name': 'xentr',
                    'weight': 1,
                    'f': self.main_loss
                }
            ]
        super().fit(
            train_loader, val_loader, epochs, patience, task, task_mask,
            last_step, verbose
        )
        if last_step:
            if (self.current_task + 1) < len(self.model):
                self.model[self.current_task + 1].load_state_dict(
                    self.model[self.current_task].state_dict()
                )
            for param in self.model[self.current_task].parameters():
                param.requires_grad = False
            self.task_fc = None

            # 2) Classifier stage
            if self.memory_manager is not None and self.current_task > 0:
                memory_sets = list(self.memory_manager.get_tasks())
                new_dataset = MultiDataset(memory_sets)
                mem_loader = DataLoader(
                    new_dataset, train_loader.batch_size, True,
                    num_workers=train_loader.num_workers, drop_last=True
                )
                self.fc = nn.Linear(
                    self.last_features * self.n_tasks, self.n_classes
                )
                self.train_functions = self.val_functions = [
                    {
                        'name': 'xentr',
                        'weight': 1,
                        'f': self.main_loss
                    }
                ]

                if patience > epochs:
                    epochs = patience
                super().fit(
                    mem_loader, mem_loader, epochs, patience, task,
                    self.global_mask, last_step, verbose
                )
        self.task_mask = None

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.task_masks = net_state['masks']

        return net_state

    def get_state(self):
        net_state = super().get_state()
        net_state['masks'] = self.task_masks
        return net_state


class iCARL(IncrementalModel):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True,
        memory_strength=0.5,
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, task
        )
        self.train_functions = self.model.train_functions + [
            {
                'name': 'dist',
                'weight': memory_strength,
                'f': lambda p, t: self.distillation_loss()
            }
        ]
        self.val_functions = self.model.val_functions

        self.update_logs()

        # Memory
        self.memx = None  # stores raw inputs, PxD
        self.memy = None
        self.last_step = False

    def prebatch_update(self, batch, batches, x, y):
        if self.task:
            y = self.task_mask[y]
        if self.epoch == 0:
            if self.memx is None:
                self.memx = x.detach().cpu().data.clone()
                self.memy = y.detach().cpu().data.clone()
            else:
                self.memx = torch.cat(
                    (self.memx, x.detach().cpu().data.clone())
                )
                self.memy = torch.cat(
                    (self.memy, y.detach().cpu().data.clone())
                )
        p = 0
        for param in self.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    flat_grad = param.grad.cpu().detach().data.flatten()
                    self.cum_grad[p] += flat_grad / batches
                p += 1

    def epoch_update(self, epochs, loader):
        last_epoch = (self.model.epoch + 1) == epochs
        if last_epoch and self.last_step:
            self.first = False
            if self.memory_manager is not None:
                training = self.model.training
                self.model.eval()
                with torch.no_grad():
                    self.memory_manager.update_memory(
                        self.memx, self.memy, self.current_task, self.model
                    )
                if training:
                    self.model.train()
            self.memx = None
            self.memy = None

    def load_model(self, net_name):
        net_state = super().load_model(net_name)

        return net_state

    def get_state(self):
        net_state = super().get_state()
        return net_state


class GSS(IncrementalModel):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True, n_recent=None
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, False
        )

        # Memory
        self.n_recent = n_recent
        self.recent_x = []
        self.recent_y = []

    def mini_batch_loop(
        self, data, train=True, verbose=True
    ):
        """
            This is the main loop. It's "generic" enough to account for multiple
            types of data (target and input) and it differentiates between
            training and testing. While inherently all networks have a training
            state to check, here the difference is applied to the kind of data
            being used (is it the validation data or the training data?). Why am
            I doing this? Because there might be different metrics for each type
            of data. There is also the fact that for training, I really don't care
            about the values of the losses, since I only want to see how the global
            value updates, while I want both (the losses and the global one) for
            validation.
            :param data: Dataloader for the network.
            :param train: Whether to use the training dataloader or the validation
             one.
            :return:
        """
        losses = list()
        mid_losses = list()
        accs = list()
        n_batches = len(data)
        for batch_i, (x, y) in enumerate(data):
            if not self.training:
                # First, we do a forward pass through the network.
                pred_labels, x_cuda, y_cuda = self.observe(x, y)

                # After that, we can compute the relevant losses.
                if train:
                    # Training losses (applied to the training data)
                    batch_losses = [
                        l_f['weight'] * l_f['f'](pred_labels, y_cuda)
                        for l_f in self.train_functions
                    ]
                    batch_loss = sum(batch_losses)
                else:
                    # Validation losses (applied to the validation data)
                    batch_losses = [
                        l_f['f'](pred_labels, y_cuda)
                        for l_f in self.val_functions
                    ]
                    batch_loss = sum([
                        l_f['weight'] * l
                        for l_f, l in zip(self.val_functions, batch_losses)
                    ])
                    mid_losses.append([l.tolist() for l in batch_losses])
                    batch_accs = [
                        l_f['f'](pred_labels, y_cuda)
                        for l_f in self.acc_functions
                    ]
                    accs.append([a.tolist() for a in batch_accs])

                # It's important to compute the global loss in both cases.
                loss_value = batch_loss.tolist()
                losses.append(loss_value)
                if verbose:
                    self.print_progress(
                        batch_i, n_batches, loss_value, np.mean(losses)
                    )
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            else:
                losses.append(self.model_update(
                    x, y, batch_i, n_batches, data.batch_size
                ))
                if verbose:
                    self.print_progress(
                        batch_i, n_batches, losses[-1], np.mean(losses)
                    )
                if self.n_recent is not None:
                    self.recent_x.append(x)
                    self.recent_y.append(y)
                    len_recent = sum([len(ri) for ri in self.recent_x])
                    if len_recent > self.n_recent:
                        self.memory_manager.update_memory(
                            torch.cat(self.recent_x, dim=0),
                            torch.cat(self.recent_y, dim=0),
                            self.current_task, self.model
                        )
                        self.recent_x = []
                        self.recent_y = []
                else:
                    self.memory_manager.update_memory(
                        x, y, self.current_task, self.model
                    )

        # Mean loss of the global loss (we don't need the loss for each batch).
        if len(losses) > 0:
            mean_loss = np.mean(losses)
        else:
            mean_loss = np.inf

        if train:
            return mean_loss
        else:
            # If using the validation data, we actually need to compute the
            # mean of each different loss.
            mean_losses = np.mean(list(zip(*mid_losses)), axis=1)
            np_accs = np.array(list(zip(*accs)))
            mean_accs = np.mean(np_accs, axis=1) if np_accs.size > 0 else []
        return mean_loss, mean_losses, mean_accs

    def update(self, x, y):
        pred_y = self.model(x.to(self.device))
        y_cuda = y.to(self.device)
        batch_losses = [
            l_f['weight'] * l_f['f'](pred_y, y_cuda)
            for l_f in self.train_functions
        ]
        batch_loss = sum(batch_losses)
        loss_value = batch_loss.tolist()
        try:
            batch_loss.backward()
            self.optimizer_alg.step()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass

        return loss_value

    def model_update(self, x, y, batch_i, n_batches, batch_size):
        if self.memory_manager is not None:
            losses = []
            memory_set = MultiDataset(
                [MemoryContainer(x, y)] + list(self.memory_manager.get_tasks())
            )
            memory_loader = DataLoader(
                memory_set, batch_size=batch_size, shuffle=True,
                drop_last=True
            )
            for x, y in memory_loader:
                self.model.optimizer_alg.zero_grad()
                loss_value = self.update(x, y)
                losses.append(loss_value)
                self.print_progress(
                    batch_i, n_batches, loss_value, np.mean(losses)
                )
            final_loss = np.mean(losses)
        else:
            final_loss = self.update(x, y)

        return final_loss

    def load_model(self, net_name):
        net_state = super().load_model(net_name)

        return net_state

    def get_state(self):
        net_state = super().get_state()
        return net_state


class GDumb(IncrementalModel):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, task
        )

    def mini_batch_loop(
        self, data, train=True, verbose=True
    ):
        """
            This is the main loop. It's "generic" enough to account for multiple
            types of data (target and input) and it differentiates between
            training and testing. While inherently all networks have a training
            state to check, here the difference is applied to the kind of data
            being used (is it the validation data or the training data?). Why am
            I doing this? Because there might be different metrics for each type
            of data. There is also the fact that for training, I really don't care
            about the values of the losses, since I only want to see how the global
            value updates, while I want both (the losses and the global one) for
            validation.
            :param data: Dataloader for the network.
            :param train: Whether to use the training dataloader or the validation
             one.
            :return:
        """
        losses = list()
        mid_losses = list()
        accs = list()
        n_batches = len(data)
        for batch_i, (x, y) in enumerate(data):
            if not self.training:
                # First, we do a forward pass through the network.
                pred_labels, x_cuda, y_cuda = self.observe(x, y)

                # After that, we can compute the relevant losses.
                if train:
                    # Training losses (applied to the training data)
                    batch_losses = [
                        l_f['weight'] * l_f['f'](pred_labels, y_cuda)
                        for l_f in self.train_functions
                    ]
                    batch_loss = sum(batch_losses)
                else:
                    # Validation losses (applied to the validation data)
                    batch_losses = [
                        l_f['f'](pred_labels, y_cuda)
                        for l_f in self.val_functions
                    ]
                    batch_loss = sum([
                        l_f['weight'] * l
                        for l_f, l in zip(self.val_functions, batch_losses)
                    ])
                    mid_losses.append([l.tolist() for l in batch_losses])
                    batch_accs = [
                        l_f['f'](pred_labels, y_cuda)
                        for l_f in self.acc_functions
                    ]
                    accs.append([a.tolist() for a in batch_accs])

                # It's important to compute the global loss in both cases.
                loss_value = batch_loss.tolist()
                losses.append(loss_value)

                if verbose:
                    self.print_progress(
                        batch_i, n_batches, loss_value, np.mean(losses)
                    )
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            else:
                updated = self.memory_manager is not None
                if updated:
                    training = self.model.training
                    self.model.eval()
                    with torch.no_grad():
                        updated = self.memory_manager.update_memory(
                            x, y, self.current_task, self.model
                        )
                    if training:
                        self.model.train()
                if updated:
                    losses.append(self.model_update(
                        batch_i, n_batches, data.batch_size
                    ))
                else:
                    if len(losses) > 0 and verbose:
                        self.print_progress(
                            batch_i, n_batches, losses[-1], np.mean(losses)
                        )

        # Mean loss of the global loss (we don't need the loss for each batch).
        if len(losses) > 0:
            mean_loss = np.mean(losses)
        else:
            mean_loss = np.inf

        if train:
            return mean_loss
        else:
            # If using the validation data, we actually need to compute the
            # mean of each different loss.
            mean_losses = np.mean(list(zip(*mid_losses)), axis=1)
            np_accs = np.array(list(zip(*accs)))
            mean_accs = np.mean(np_accs, axis=1) if np_accs.size > 0 else []
        return mean_loss, mean_losses, mean_accs

    def model_update(self, batch_i, n_batches, batch_size):
        self.model.optimizer_alg.zero_grad()
        losses = list()
        if self.memory_manager is not None:
            for task_mask, memory_set in zip(
                self.task_masks, self.memory_manager.get_tasks()
            ):
                memory_loader = DataLoader(
                    memory_set, batch_size=batch_size, shuffle=True,
                    drop_last=True
                )
                for x, y in memory_loader:
                    pred_y = self.model(x.to(self.device))
                    y_cuda = y.to(self.device)
                    if self.task:
                        pred_labels = pred_labels[:, self.task_mask]
                        y_cuda = update_y(y_cuda, self.task_mask)
                    batch_losses = [
                        l_f['weight'] * l_f['f'](pred_y, y_cuda)
                        for l_f in self.train_functions
                    ]
                    batch_loss = sum(batch_losses)
                    loss_value = batch_loss.tolist()
                    losses.append(loss_value)
                    try:
                        batch_loss.backward()
                        self.optimizer_alg.step()

                        self.print_progress(
                            batch_i, n_batches, loss_value, np.mean(losses)
                        )
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except RuntimeError:
                        pass
        return np.mean(losses)

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.task_masks = net_state['masks']

        return net_state

    def get_state(self):
        net_state = super().get_state()
        net_state['masks'] = self.task_masks
        return net_state


class Piggyback(IncrementalModel):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True,
        prune_ratio=0.5
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, task
        )
        self.first = True
        self.prune_ratio = prune_ratio
        self.weight_masks = []
        self.current_mask = None
        self.weight_buffer = []
        self.model_layers = [
            layer for layer in self.model.modules()
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)
        ]

    def prebatch_update(self, batch, batches, x, y):
        # Copy all the weight that should not be modified
        for layer, mask in zip(self.model_layers, self.current_mask):
            self.weight_buffer.append(
                deepcopy(layer.weight[mask].detach().cpu())
            )

    def batch_update(self, batch, batches, x, y):
        super().batch_update(batch, batches, x, y)
        for layer, mask, weight in zip(
            self.model_layers, self.current_mask, self.weight_buffer
        ):
            if torch.sum(mask) > 0:
                print(
                    'Restoring weights {:d}/{:d}'.format(
                        torch.sum(mask), torch.numel(mask)
                    )
                )
                layer.weight.data[mask].copy_(weight)
        self.weight_buffer = []

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        task=None,
        task_mask=None,
        last_step=False,
        verbose=True
    ):
        # 1) Train the new task normally
        if self.current_mask is None:
            # If no weight mask exists (first step) a new one is created
            # with no "safe" set of weights selected.
            self.current_mask = []
            for layer in self.model_layers:
                self.current_mask.append(
                    torch.zeros_like(
                        layer.weight, dtype=torch.bool, device=self.device
                    )
                )

        new_mask = []
        for layer in self.model_layers:
            new_mask.append(
                torch.zeros_like(
                    layer.weight, dtype=torch.bool, device=self.device
                )
            )
        super().fit(
            train_loader, val_loader, epochs, patience, task, task_mask,
            last_step, verbose
        )

        if last_step:
            # 2) Prune the network
            # We need to flatten the weights to make sure we select the lowest
            # overall magnitudes.
            # Another important detail is the need to only prune "prunable"
            # weights! To guarantee that we focus on the "non-selected" weights
            # (essentially the "not current mask" of weights).
            all_weights = torch.cat([
                layer.weight.data[torch.logical_not(mask)]
                for layer, mask in zip(self.model_layers, self.current_mask)
            ])

            # We select the highest magnitudes to keep (we prune the lowest
            # ones).
            sorted_weights = torch.argsort(torch.abs(all_weights))
            weight_indices = torch.argsort(sorted_weights)
            dropped_weights = weight_indices < self.prune_ratio * len(all_weights)
            print(
                '{:d}/{:d} to be pruned weights'.format(
                    torch.sum(dropped_weights), torch.numel(dropped_weights)
                )
            )

            mask_idx = 0
            for i, (c_mask, n_mask, layer) in enumerate(zip(
                self.current_mask, new_mask, self.model_layers
            )):
                prune_mask = torch.logical_not(c_mask)
                n_elem = torch.sum(prune_mask)
                flat_mask = dropped_weights[mask_idx:mask_idx + n_elem]
                # The mask represents the weights we want to keep secured.
                # However, as the next step involves retraining the weights
                # we will keep, for now we store an inverted matrix.

                # New mask will contain the fixed weights (previous + new).
                n_mask.data[prune_mask].copy_(torch.logical_not(flat_mask))
                n_mask.data[torch.logical_not(prune_mask)].fill_(True)
                # Current mask will contain the pruned weights (we do not want
                # to train them) plus the previous ones already stored.
                c_mask.data[prune_mask].copy_(flat_mask)

                layer.weight.data[n_mask].fill_(0.0)
                print(
                    'Filling {:d}/{:d}[{:d}] weights'
                    ' (layer {:d}) - prunable weights {:d}/{:d} <idx {:,}>'.format(
                        torch.sum(n_mask), len(all_weights), torch.numel(all_weights),
                        i, torch.sum(prune_mask), torch.numel(prune_mask),
                        mask_idx
                    )
                )
                mask_idx += n_elem

            # 3) Retrain the pruned network
            # The original paper trained for half the epochs. However, this
            # framework relies on running epochs one by one. If we divide by 2
            # the number of epochs this could lead to running this step 0
            # epochs.
            # To avoid that we assume a minimum of 1 epoch. The biggest issue
            # is that we are now essentially training for the same number of
            # epochs.
            min_epochs = max(epochs // 2, 1)
            super().fit(
                train_loader, val_loader, min_epochs, patience, task, task_mask,
                last_step, verbose
            )

            # 4) Prepare the mask for the next task
            self.current_mask = new_mask
            self.weight_masks.append(new_mask)

    def inference(self, data, nonbatched=True, task=None):
        if task is None:
            mask = self.current_mask
        else:
            mask = self.weight_masks[task]

        # We set to 0 the weights that were pruned
        for layer, mask_i in zip(self.model_layers, mask):
            self.weight_buffer.append(
                deepcopy(layer.weight[mask_i].detach().cpu())
            )
            layer.weight.data[torch.logical_not(mask_i)].fill_(0.0)

        results = super().inference(data, nonbatched, task)
        # We fill the weights again
        for layer, mask, weight in zip(
            self.model_layers, self.current_mask, self.weight_buffer
        ):
            layer.weight.data[mask].copy_(weight)
        self.weight_buffer = []

        return results
