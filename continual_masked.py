from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import quadprog
from sklearn.decomposition import PCA
from base import BaseModel, SelfAttentionBlock
from datasets import MultiDataset


def update_y(y, mask):
    y = torch.cat(
        [torch.where(mask.to(y.device) == y_i)[0] for y_i in y]
    ).to(y.device)
    return y


class MetaModel(BaseModel):
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

    def load_model(self, net_name):
        net_state = torch.load(net_name, map_location=torch.device('cpu'))
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


class MetaModelMemory(MetaModel):
    def __init__(
            self, basemodel, best=True, memory_manager=None,
            n_classes=100, n_tasks=10, lr=None, task=True
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, task
        )

    def mini_batch_loop(self, data, train=True):
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
        return super().mini_batch_loop(data, train)


class EWC(MetaModel):
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
        verbose=True
    ):
        if self.first:
            for loss_f in self.train_functions:
                if loss_f['name'] is 'ewc':
                    loss_f['weight'] = 0
        super().fit(
            train_loader, val_loader, epochs, patience, task, task_mask,
            verbose
        )
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
            if loss_f['name'] is 'ewc':
                loss_f['weight'] = self.ewc_weight

    def epoch_update(self, epochs, loader):
        if self.ewc_alpha is not None:
            self.fisher(loader)
            if self.epoch == 0:
                for loss_f in self.train_functions:
                    if loss_f['name'] is 'ewc':
                        loss_f['weight'] = self.ewc_weight


class GEM(MetaModel):
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
        self.offsets = []

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
                    output = output[:, task_mask]
                    labels = update_y(labels, task_mask)

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
        net_state['offsets'] = self.offsets
        return net_state

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.grad_dims = net_state['grad_dims']
        if type(net_state['grads']) is list:
            self.grads = [grad.cpu() for grad in net_state['grads']]
        else:
            self.grads = net_state['grads'].cpu()
        self.observed_tasks = net_state['tasks']
        self.offsets = net_state['offsets']

        return net_state

    def prebatch_update(self, batch, batches, x, y):
        super().prebatch_update(batch, batches, x, y)
        self.update_gradients()
        self.constraint_check()

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
        self.task_masks.append(task_mask)
        super().fit(
            train_loader, val_loader, epochs, patience, task, task_mask,
            verbose
        )


class AGEM(GEM):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True,
        memory_strength=0.5,
    ):
        super().__init__(
            basemodel, best, memory_manager,
            n_classes, n_tasks, lr, task, memory_strength
        )

    def get_grad(self):
        indx = torch.tensor(self.observed_tasks[:-1], dtype=torch.long)
        return self.grads.index_select(1, indx).mean(dim=1, keepdim=True)


class SGEM(GEM):
    def __init__(
            self, basemodel, best=True, memory_manager=None,
            n_classes=100, n_tasks=10, lr=None, task=True,
            memory_strength=0.5,
    ):
        super().__init__(
            basemodel, best, memory_manager,
            n_classes, n_tasks, lr, task, memory_strength
        )

    def get_grad(self):
        indx = torch.tensor(
            [np.random.randint(len(self.observed_tasks[:-1]))],
            dtype=torch.long
        )
        return self.grads.index_select(1, indx)


class Independent(MetaModel):
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
        verbose=True
    ):
        if task is None:
            self.optimizer_alg = self.model[self.current_task + 1].optimizer_alg
        else:
            self.optimizer_alg = self.model[task].optimizer_alg
        super().fit(
            train_loader, val_loader, epochs, patience, task, task_mask,
            verbose
        )
        if (self.current_task + 1) < len(self.model):
            self.model[self.current_task + 1].load_state_dict(
                self.model[self.current_task].state_dict()
            )


class iCARL(MetaModel):
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
        self.offsets = []

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
        if last_epoch:
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
        self.offsets.append(task_mask)
        super().fit(
            train_loader, val_loader, epochs, patience, task, task_mask,
            verbose
        )

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.offsets = net_state['offsets']

        return net_state

    def get_state(self):
        net_state = super().get_state()
        net_state['offsets'] = self.offsets
        return net_state


class GDumb(MetaModel):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, task
        )

        # Memory
        self.offsets = []

    def mini_batch_loop(
            self, data, train=True
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
                    if len(losses) > 0:
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

    def model_update(self,  batch_i, n_batches, batch_size):
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
                        pred_y = pred_y[:, task_mask]
                        y_cuda = update_y(y_cuda, task_mask)
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
        self.offsets.append(task_mask)
        super().fit(
            train_loader, val_loader, epochs, patience, task, task_mask,
            verbose
        )

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.offsets = net_state['offsets']

        return net_state

    def get_state(self):
        net_state = super().get_state()
        net_state['offsets'] = self.offsets
        return net_state
