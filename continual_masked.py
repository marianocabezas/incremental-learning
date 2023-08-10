import os
import io
import gzip
import shutil
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from base import BaseModel
from datasets import MultiDataset


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
            indx = np.random.randint(0, len(x_k) - 1)
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
                    losses += self._kl_div_loss(mask)
            else:
                losses = self._kl_div_loss(0, self.n_classes)
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
        verbose=True
    ):
        self.task_mask = task_mask
        if task is not None:
            self.current_task = task
        if self.current_task not in self.observed_tasks:
            self.observed_tasks.append(self.current_task)
        super().fit(train_loader, val_loader, epochs, patience, verbose)


class IncrementalModelMemory(IncrementalModel):
    def __init__(
            self, basemodel, best=True, memory_manager=None,
            n_classes=100, n_tasks=10, lr=None, task=True
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, task
        )

    def mini_batch_loop(self, data, train=True, verbose=True):
        if self.memory_manager is not None and self.current_task > 0 and train:
            if self.task_mask is not None:
                self.task_mask = torch.cat(self.task_masks)
            max_task = self.current_task - 1
            memory_sets = list(self.memory_manager.get_tasks(max_task))
            new_dataset = MultiDataset([data.dataset] + memory_sets)
            data = DataLoader(
                new_dataset, data.batch_size, True,
                num_workers=data.num_workers, drop_last=True
            )
        return super().mini_batch_loop(data, train, verbose)
