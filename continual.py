from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import quadprog
from sklearn.decomposition import PCA
from base import BaseModel, SelfAttentionBlock


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
        self.offset1 = None
        self.offset2 = None
        self.grams = []
        self.logits = []

        self.train_functions = self.model.train_functions
        self.val_functions = self.model.val_functions

        self.optimizer_alg = self.model.optimizer_alg

    def prebatch_update(self, batch, batches, x, y):
        if self.task:
            y = y + self.offset1
        if self.memory_manager is not None:
            training = self.model.training
            self.model.eval()
            with torch.no_grad():
                self.memory_manager.update_memory(
                    x, y, self.current_task, self.model
                )
            if training:
                self.model.train()

    def fill_grams(self, batch_size=None):
        if self.memory_manager is not None:
            self.grams = []
            for task_data in self.memory_manager.get_tasks():
                task_loader = DataLoader(task_data, batch_size=batch_size)
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
            'grams': self.grams,
            'logits': self.logits,
            'state': self.state_dict(),

        }
        return net_state

    def save_model(self, net_name):
        net_state = self.get_state()
        torch.save(net_state, net_name)

    def load_model(self, net_name):
        net_state = torch.load(net_name, map_location=self.device)
        self.first = net_state['first']
        self.n_classes = net_state['n_classes']
        self.current_task = net_state['task']
        self.memory_manager = net_state['manager']
        self.grams = net_state['grams']
        self.logits = net_state['logits']
        self.lr = net_state['lr']
        if self.lr is not None:
            self.model.lr = self.lr
            self.reset_optimiser()
        self.task = net_state['task_incremental']
        self.load_state_dict(net_state['state'])
        return net_state

    def forward(self, *inputs):
        return self.model(*inputs)

    def observe(self, x, y):
        pred_labels, x_cuda, y_cuda = super().observe(x, y)
        if self.task:
            pred_labels = pred_labels[:, self.offset1:self.offset2]
            y_cuda = y_cuda - self.offset1

        return pred_labels, x_cuda, y_cuda

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        task=None,
        offset1=None,
        offset2=None,
        verbose=True
    ):
        self.offset1 = offset1
        self.offset2 = offset2
        if task is not None:
            self.current_task = task
        if self.current_task not in self.observed_tasks:
            self.observed_tasks.append(self.current_task)
        super().fit(train_loader, val_loader, epochs, patience, verbose)


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
        offset1=None,
        offset2=None,
        verbose=True
    ):
        if self.first:
            for loss_f in self.train_functions:
                if loss_f['name'] is 'ewc':
                    loss_f['weight'] = 0
        super().fit(
            train_loader, val_loader, epochs, patience, task, offset1, offset2,
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
                task_loader = DataLoader(task_memory, train_loader.batch_size)
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
            memories.t().double().numpy()
        )
        gradient_np = np.nan_to_num(
            gradient.contiguous().view(-1).double().numpy()
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
            for past_task, (offset1, offset2) in zip(
                self.observed_tasks[:-1], self.offsets[:-1]
            ):
                self.zero_grad()

                memories_t, labels_t = self.memory_manager.get_task(past_task)
                output = self(torch.stack(memories_t).to(self.device))
                labels = torch.stack(labels_t).to(self.device)
                if self.task:
                    output = output[:, offset1:offset2]
                    labels = labels - offset1

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
        offset1=None,
        offset2=None,
        verbose=True
    ):
        if offset1 is None:
            offset1 = 0
        if offset2 is None:
            offset2 = self.n_classes
        self.offsets.append((offset1, offset2))
        super().fit(
            train_loader, val_loader, epochs, patience, task, offset1, offset2,
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


class NGEM(GEM):
    def __init__(
            self, basemodel, best=True, memory_manager=None,
            n_classes=100, n_tasks=10, lr=None, task=True,
            memory_strength=0.5,
    ):
        super().__init__(
            basemodel, best, memory_manager,
            n_classes, n_tasks, lr, task, memory_strength
        )
        self.block_grad_dims = []

        block_params = 0
        for ind, param in enumerate(self.parameters()):
            if ind % 5 == 0:
                block_params = 0
                block_params += param.data.numel()
                if ind == len(list(self.parameters())) - 1:
                    self.block_grad_dims.append(block_params)
            elif ind % 5 == 4:
                block_params += param.data.numel()
                self.block_grad_dims.append(block_params)
            else:
                block_params += param.data.numel()
                if ind == len(list(self.parameters())) - 1:
                    self.block_grad_dims.append(block_params)

    def project(self, gradient, memories, eps=1e-3):
        """
            First we do orthogonality of memories and then find the null space of
            their gradients
        """
        np.seterr(divide='ignore', invalid='ignore')
        memories_np = np.nan_to_num(
            memories.t().double().numpy()
        ).astype(np.float32)
        gradient_np = np.nan_to_num(
            gradient.contiguous().view(-1).double().numpy()
        ).astype(np.float32)
        memories_np_sum = np.sum(memories_np, axis=0)
        if len(memories_np) == 1:
            x = gradient_np - np.min([
                (memories_np_sum.transpose().dot(gradient_np) /
                 memories_np_sum.transpose().dot(memories_np_sum)), -self.margin
            ]) * memories_np_sum
        else:
            memories_np_mean = np.mean(memories_np, axis=0)
            memories_np_del_mean = memories_np - memories_np_mean.reshape(1, -1)
            memories_np_pca = PCA(n_components=min(3, len(memories_np)))
            memories_np_pca.fit(memories_np_del_mean)
            memories_np_orth = memories_np_pca.components_
            Pg = gradient_np - memories_np_orth.transpose().dot(
                memories_np_orth.dot(gradient_np))
            Pg_bar = memories_np_sum - memories_np_orth.transpose().dot(
                memories_np_orth.dot(memories_np_sum))
            if memories_np_sum.transpose().dot(Pg) > 0:
                x = Pg
            else:
                x = gradient_np - np.min([
                    memories_np_sum.transpose().dot(Pg) /
                    memories_np_sum.transpose().dot(Pg_bar), -self.margin
                ]) * memories_np_sum - memories_np_orth.transpose().dot(
                    memories_np_orth.dot(gradient_np)) + memories_np_sum.transpose(
                ).dot(Pg) / memories_np_sum.transpose().dot(
                    Pg_bar) * memories_np_orth.transpose().dot(
                    memories_np_orth.dot(memories_np_sum))

        gradient.copy_(
            torch.Tensor(np.nan_to_num(x).astype(np.float32)).view(-1, 1)
        )

    def constraint_check(self):
        t = self.current_task
        if len(self.observed_tasks) > 1:
            # Copy gradient
            self.store_grad(t)
            indx = torch.LongTensor(self.observed_tasks[:-1])

            for cnt in range(len(self.block_grad_dims)):
                beg = 0 if cnt == 0 else sum(self.block_grad_dims[:cnt])
                en = sum(self.block_grad_dims[:cnt + 1])
                if beg == en:
                    continue
                self.project(
                    self.grads[:, t].unsqueeze(1)[beg:en],
                    self.grads.index_select(1, indx)[beg:en],
                )
            # copy gradients back
            self.overwrite_grad(self.grads[:, t])

    def get_state(self):
        net_state = super().get_state()
        net_state['block_dims'] = self.block_grad_dims
        return net_state

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.block_grad_dims = net_state['block_dims']


class ResetGEM(GEM):
    def __init__(
            self, basemodel, best=True, memory_manager=None,
            n_classes=100, n_tasks=10, lr=None, task=True,
            memory_strength=0.5,
    ):
        super().__init__(
            basemodel, best, memory_manager,
            n_classes, n_tasks, lr, task, memory_strength
        )

        # Memory
        self.memx = None  # stores raw inputs, PxD
        self.memy = None

    def project(self, gradient, memories, eps=1e-3):
        all_gradients = torch.cat([gradient, memories], dim=1)
        abs_gradients = torch.sum(torch.abs(all_gradients), dim=1)
        gradients = torch.abs(torch.sum(all_gradients, dim=1))
        gradient[torch.abs(abs_gradients - gradients) > eps, :] = 0
        return gradient

    def prebatch_update(self, batch, batches, x, y):
        if self.task:
            y = y + self.offset1
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
        self.update_gradients()
        self.constraint_check()

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


class IndependentGEM(GEM):
    def __init__(
            self, basemodel, best=True, memory_manager=None,
            n_classes=100, n_tasks=10, lr=None, task=True,
            memory_strength=0.5,
    ):
        super().__init__(
            basemodel, best, memory_manager,
            n_classes, n_tasks, lr, task, memory_strength
        )
        self.grads = []
        for param in self.model.parameters():
            if param.requires_grad:
                self.grads.append(torch.Tensor(param.data.numel(), n_tasks))

    def project(self, gradient, memories, eps=1e-3):
        all_gradients = torch.cat([gradient, memories], dim=1)
        abs_gradients = torch.sum(torch.abs(all_gradients), dim=1)
        gradients = torch.abs(torch.sum(all_gradients, dim=1))
        gradient[torch.abs(abs_gradients - gradients) > eps, :] = 0
        return gradient

    def store_grad(self, tid):
        p = 0
        for param in self.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    self.grads[p][:, tid].copy_(
                        param.grad.cpu().data.flatten()
                    )
                p += 1

    def constraint_check(self):
        if len(self.observed_tasks) > 1:
            p = 0
            indx = torch.tensor(self.observed_tasks[:-1], dtype=torch.long)
            for param in self.parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        current_grad = param.grad.cpu().data.flatten()
                        grad = self.grads[p].index_select(1, indx)
                        dotp = torch.mm(
                            current_grad.unsqueeze(0).to(self.device),
                            grad.to(self.device)
                        )
                        if (dotp < 0).any():
                            new_grad = self.project(
                                current_grad.unsqueeze(1).to(self.device),
                                grad.to(self.device)
                            )
                            # Copy the new gradient
                            current_grad = new_grad.contiguous().view(
                                param.grad.data.size()
                            )
                            param.grad.data.copy_(current_grad.to(param.device))
                    p += 1


class ParamGEM(ResetGEM):
    def __init__(
            self, basemodel, best=True, memory_manager=None,
            n_classes=100, n_tasks=10, lr=None, task=True,
            memory_strength=0.5,
    ):
        super().__init__(
            basemodel, best, memory_manager,
            n_classes, n_tasks, lr, task, memory_strength
        )
        self.grads = []
        for param in self.model.parameters():
            if param.requires_grad:
                self.grads.append(torch.Tensor(param.data.numel(), n_tasks))

        # Memory
        self.memx = None  # stores raw inputs, PxD
        self.memy = None

    def store_grad(self, tid):
        p = 0
        for param in self.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    self.grads[p][:, tid].copy_(
                        param.grad.cpu().data.flatten()
                    )
                p += 1

    def constraint_check(self):
        if len(self.observed_tasks) > 1:
            p = 0
            indx = torch.tensor(self.observed_tasks[:-1], dtype=torch.long)
            for param in self.parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        current_grad = param.grad.cpu().data.flatten()
                        grad = self.grads[p].index_select(1, indx)
                        dotp = torch.mm(
                            current_grad.unsqueeze(0).to(self.device),
                            grad.to(self.device)
                        )
                        if (dotp < 0).any():
                            new_grad = self.project(
                                current_grad.unsqueeze(1).to(self.device),
                                grad.to(self.device)
                            )
                            # Copy the new gradient
                            current_grad = new_grad.contiguous().view(
                                param.grad.data.size()
                            )
                            param.grad.data.copy_(current_grad.to(param.device))
                    p += 1


class LoggingGEM(GEM):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True,
        memory_strength=0.5,
    ):
        super().__init__(
            basemodel, best, memory_manager,
            n_classes, n_tasks, lr, task, memory_strength
        )
        self.grad_log = {
            'mean': [],
            'mean_abs': [],
            'std': [],
            'min': [],
            '10%': [],
            '20%': [],
            '25%': [],
            '40%': [],
            'median': [],
            '60%': [],
            '75%': [],
            '80%': [],
            '90%': [],
            'max': [],
            'dot': [],
            'norm_dot': [],
            'new_dot': [],
            'norm_new_dot': [],
            'grads_dot': [],
            'norm_grads_dot': [],
        }

    def constraint_check(self):
        new_grad = super().constraint_check().numpy()
        grads = deepcopy(self.grads[:, :(self.current_task + 1)].numpy())
        old_grad = np.expand_dims(grads[:, -1], 0)
        quantiles = np.quantile(
            grads, [.1, .2, .25, .4, .5, .6, .75, .8, .9], axis=0
        )
        self.grad_log['mean'].append(np.mean(grads, axis=0))
        self.grad_log['mean_abs'].append(np.mean(np.abs(grads), axis=0))
        self.grad_log['std'].append(np.std(grads, axis=0))
        self.grad_log['min'].append(np.min(grads, axis=0))
        self.grad_log['10%'].append(quantiles[0])
        self.grad_log['20%'].append(quantiles[1])
        self.grad_log['25%'].append(quantiles[2])
        self.grad_log['40%'].append(quantiles[3])
        self.grad_log['median'].append(quantiles[4])
        self.grad_log['60%'].append(quantiles[5])
        self.grad_log['75%'].append(quantiles[6])
        self.grad_log['80%'].append(quantiles[7])
        self.grad_log['90%'].append(quantiles[8])
        self.grad_log['max'].append(np.max(grads, axis=0))
        if self.current_task > 0:
            norms = np.clip(
                np.linalg.norm(grads, axis=0, keepdims=True), 1e-6, np.inf
            )
            new_norm = new_grad / np.clip(
                np.linalg.norm(new_grad, axis=0, keepdims=True), 1e-6, np.inf
            )
            norm_grads = grads / norms
            old_norm = np.expand_dims(norm_grads[:, -1], 0)
            self.grad_log['dot'].append(old_grad @ grads[:, :-1])
            self.grad_log['norm_dot'].append(
                np.clip(old_norm @ norm_grads[:, :-1], -1, 1)
            )
            self.grad_log['new_dot'].append(
                new_grad.transpose() @ grads[:, :-1]
            )
            self.grad_log['norm_new_dot'].append(
                np.clip(new_norm.transpose() @ norm_grads[:, :-1], -1, 1)
            )
            self.grad_log['grads_dot'].append(old_grad @ new_grad)
            self.grad_log['norm_grads_dot'].append(
                np.clip(old_norm @ new_norm, -1, 1)
            )

    def get_state(self):
        net_state = super().get_state()
        net_state['log'] = self.grad_log
        return net_state

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.grad_log = net_state['log']

        return net_state


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

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        task=None,
        offset1=None,
        offset2=None,
        verbose=True
    ):
        if task is None:
            self.optimizer_alg = self.model[self.current_task + 1].optimizer_alg
        else:
            self.optimizer_alg = self.model[task].optimizer_alg
        super().fit(
            train_loader, val_loader, epochs, patience, task, offset1, offset2,
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

        # Memory
        self.memx = None  # stores raw inputs, PxD
        self.memy = None
        self.offsets = []

    def _kl_div_loss(self, offset1, offset2):
        losses = []
        x = []
        y_logits = []
        for k in range(offset1, offset2):
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
            self.model(x)[:, offset1:offset2], 1
        )
        y = F.softmax(y_logits[:, offset1:offset2], 1)
        losses.append(
            F.kl_div(
                prediction, y, reduction='batchmean'
            ) * (offset2 - offset1)
        )
        return losses

    def distillation_loss(self):
        if not self.first and self.memory_manager is not None:
            if self.task:
                losses = []
                for offset1, offset2 in self.offsets[:-1]:
                    losses += self._kl_div_loss(offset1, offset2)
            else:
                losses = self._kl_div_loss(0, self.n_classes)
        else:
            losses = []
        return sum(losses)

    def prebatch_update(self, batch, batches, x, y):
        if self.task:
            y = y + self.offset1
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
        offset1=None,
        offset2=None,
        verbose=True
    ):
        if offset1 is None:
            offset1 = 0
        if offset2 is None:
            offset2 = self.n_classes
        self.offsets.append((offset1, offset2))
        super().fit(
            train_loader, val_loader, epochs, patience, task, offset1, offset2,
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
                    self.print_progress(
                        batch_i, n_batches, losses[-1], np.mean(losses)
                    )

        # Mean loss of the global loss (we don't need the loss for each batch).
        mean_loss = np.mean(losses)

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
            for (offset1, offset2), memory_set in zip(
                self.offsets, self.memory_manager.get_tasks()
            ):
                memory_loader = DataLoader(
                    memory_set, batch_size=batch_size, shuffle=True,
                    drop_last=True
                )
                for x, y in memory_loader:
                    pred_y = self.model(x.to(self.device))
                    y_cuda = y.to(self.device)
                    if self.task:
                        pred_y = pred_y[:, offset1:offset2]
                        y_cuda = y_cuda - offset1
                    batch_losses = [
                        l_f['weight'] * l_f['f'](pred_y, y_cuda)
                        for l_f in self.train_functions
                    ]
                    batch_loss = sum(batch_losses)
                    loss_value = batch_loss.tolist()
                    losses.append(loss_value)
                    batch_loss.backward()
                    self.optimizer_alg.step()

                    self.print_progress(
                        batch_i, n_batches, loss_value, np.mean(losses)
                    )
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        return np.mean(losses)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        task=None,
        offset1=None,
        offset2=None,
        verbose=True
    ):
        if offset1 is None:
            offset1 = 0
        if offset2 is None:
            offset2 = self.n_classes
        self.offsets.append((offset1, offset2))
        super().fit(
            train_loader, val_loader, epochs, patience, task, offset1, offset2,
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


class DyTox(MetaModel):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True,
        sab=5, tab=1, heads=12, embed_dim=384, patch_size=4
    ):
        super().__init__(
            basemodel, best, memory_manager, n_classes, n_tasks, lr, task
        )
        self.sab = sab
        self.tab = tab
        self.embed_dim = embed_dim
        self.model = None

        # Transformers
        self.tokenizer = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.task_tokens = nn.ParameterList([])
        self.sab_list = nn.ModuleList([
            SelfAttentionBlock(embed_dim, embed_dim, heads)
            for _ in range(sab)
        ])
        self.tab_list = nn.ModuleList([
            SelfAttentionBlock(embed_dim, embed_dim, heads)
            for _ in range(tab)
        ])
        self.classifiers = nn.ModuleList([])

    def reset_optimiser(self, model_params=None):
        if model_params is None:
            model_params = filter(lambda p: p.requires_grad, self.parameters())
        super().reset_optimiser(model_params)
        self.model.reset_optimiser(model_params)
        self.optimizer_alg = self.model.optimizer_alg

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        task=None,
        offset1=None,
        offset2=None,
        verbose=True
    ):
        self.task_tokens.append(
            nn.Parameter(torch.rand(self.embed_dim), requires_grad=True)
        )
        self.classifiers.append(nn.Linear(self.embed_dim, self.n_classes))
        self.reset_optimiser()
        super().fit(
            train_loader, val_loader, epochs, patience, task, offset1, offset2,
            verbose
        )

    def mini_batch_loop(self, data, train=True):
        data_results = super().mini_batch_loop(data, train)
        if self.memory_manager is not None:
            max_task = self.current_task - 1
            task_results = [data_results]
            for task_data in self.memory_manager.get_tasks(max_task):
                task_loader = DataLoader(task_data, data.batch_size)
                task_results.append(super().mini_batch_loop(task_loader, train))
            data_results = np.mean(task_results, axis=1)
        return data_results

    def _class_forward(self, tokens):
        predictions = []
        for t_token, clf in zip(self.task_tokens, self.classifiers):
            query = torch.repeat_interleave(t_token, len(tokens), dim=0)
            for tab in self.tab_list:
                tab.to(self.device)
                tokens = tab(tokens, query.to(self.device))
            clf.to(self.device)
            predictions.append(clf(tokens[:, 0]))

        return torch.cat(predictions, dim=-1)

    def forward(self, x):
        self.tokenizer.to(self.device)
        tokens = self.tokenizer(x).flatten(2).permute(0, 2, 1)
        for sab in self.sab_list:
            sab.to(self.device)
            tokens = sab(tokens)
        return self._class_forward(tokens)

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.task_tokens = net_state['task_tokens']

        return net_state

    def get_state(self):
        net_state = super().get_state()
        net_state['task_tokens'] = self.task_tokens
        return net_state


class TaskGEM(DyTox, IndependentGEM):
    def __init__(
        self, basemodel, best=True, memory_manager=None,
        n_classes=100, n_tasks=10, lr=None, task=True,
        tab=1, heads=12, embed_dim=384, memory_strength=0.5,
    ):
        IndependentGEM.__init__(
            self, basemodel, best, memory_manager,
            n_classes, n_tasks, lr, task,
            memory_strength
        )
        self.tab = tab
        self.embed_dim = embed_dim
        self.model = None

        # Transformers
        self.task_tokens = nn.ParameterList([])
        self.tab_list = nn.ModuleList([
            SelfAttentionBlock(embed_dim, embed_dim, heads)
            for _ in range(tab)
        ])
        self.classifiers = nn.ModuleList([])

    def mini_batch_loop(self, data, train=True):
        return IndependentGEM.mini_batch_loop(self, data, train)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        task=None,
        offset1=None,
        offset2=None,
        verbose=True
    ):
        DyTox.fit(
            self, train_loader, val_loader, epochs, patience,
            task, offset1, offset2, verbose
        )

    def reset_optimiser(self, model_params=None):
        DyTox.reset_optimiser(self, model_params)

    def load_model(self, net_name):
        net_state = super().load_model(net_name)

        return net_state

    def get_state(self):
        net_state = IndependentGEM.get_state(self)
        net_state['task_tokens'] = self.task_tokens
        return net_state

    def forward(self, x):
        tokens = self.model.tokenize(x)
        return super()._class_forward(tokens)
