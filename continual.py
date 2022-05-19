from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import quadprog
from sklearn.decomposition import PCA
from base import BaseModel


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


def project5cone5(gradient, memories, beg, en, margin=0.5, eps=1e-3):
    """
        First we do orthognality of memories and then find the null space of
        these memories gradients
    """
    memories_np = memories[beg:en].cpu().t().double().numpy()
    gradient_np = gradient[beg:en].cpu().contiguous().view(-1).double().numpy()
    memories_np_mean = np.mean(memories_np, axis=0)
    memories_np_sum = np.sum(memories_np, axis=0)
    memories_np_del_mean = memories_np - memories_np_mean.reshape(1, -1)
    memories_np_pca = PCA(n_components=min(3, len(memories_np)))
    memories_np_pca.fit(memories_np_del_mean)
    if len(memories_np) == 1:
        x = gradient_np - np.min([
            (memories_np_sum.transpose().dot(gradient_np) /
             memories_np_sum.transpose().dot(memories_np_sum)), -margin
        ]) * memories_np_sum
    else:
        # memories_np_pca = orth(memories_np_del_mean.transpose()).transpose()

        memories_np_orth = memories_np_pca.components_
        # memories_np_orth = memories_np_pca
        Pg = gradient_np - memories_np_orth.transpose().dot(
            memories_np_orth.dot(gradient_np))
        Pg_bar = memories_np_sum - memories_np_orth.transpose().dot(
            memories_np_orth.dot(memories_np_sum))
        if memories_np_sum.transpose().dot(Pg) > 0:
            x = Pg
        else:
            x = gradient_np - np.min([
                memories_np_sum.transpose().dot(Pg) /
                memories_np_sum.transpose().dot(Pg_bar), -margin
            ]) * memories_np_sum - memories_np_orth.transpose().dot(
                memories_np_orth.dot(gradient_np)) + memories_np_sum.transpose(
                ).dot(Pg) / memories_np_sum.transpose().dot(
                    Pg_bar) * memories_np_orth.transpose().dot(
                        memories_np_orth.dot(memories_np_sum))

    print("task length: {}".format(len(memories_np)))
    gradient[beg:en].copy_(torch.Tensor(x).view(-1, 1))


class MetaModel(BaseModel):
    def __init__(
        self, basemodel, best=True, n_memories=0
    ):
        super().__init__()
        self.init = basemodel.init
        self.best = best
        self.first = True
        self.model = basemodel
        self.device = basemodel.device
        self.n_memories = n_memories
        self.memory_data = []
        self.memory_labs = []
        # Counters
        self.mem_cnt = 0
        self.observed_tasks = []
        self.current_task = -1

        self.train_functions = self.model.train_functions
        self.val_functions = self.model.val_functions

        self.optimizer_alg = self.model.optimizer_alg

    def reset_optimiser(self):
        super().reset_optimiser()
        self.model.reset_optimiser()
        self.optimizer_alg = self.model.optimizer_alg

    def save_model(self, net_name):
        net_state = {
            'first': self.first,
            'task': self.current_task,
            'state': self.state_dict()
        }
        torch.save(net_state, net_name)

    def load_model(self, net_name):
        net_state = torch.load(net_name, map_location=self.device)
        self.first = net_state['first']
        self.current_task = net_state['task']
        self.load_state_dict(net_state['state'])

    def forward(self, *inputs):
        return self.model(*inputs)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        verbose=True
    ):
        self.current_task += 1
        if self.current_task not in self.observed_tasks:
            self.observed_tasks.append(self.current_task)
        super().fit(train_loader, val_loader, epochs, patience, verbose)


class EWC(MetaModel):
    def __init__(
        self, basemodel, best=True, n_memories=0, ewc_weight=1e6, ewc_binary=True,
            ewc_alpha=None
    ):
        super().__init__(basemodel, best, n_memories)
        self.ewc_weight = ewc_weight
        self.ewc_binary = ewc_binary
        self.ewc_alpha = ewc_alpha
        self.observed_tasks = []
        self.current_task = -1

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

        self.grads = {
            n: [] for n, p in self.model.named_parameters()
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

    def reset_optimiser(self):
        super().reset_optimiser()
        self.model.reset_optimiser()
        self.optimizer_alg = self.model.optimizer_alg

    def save_model(self, net_name):
        net_state = {
            'state': self.state_dict(),
            'ewc_param': self.ewc_parameters,
            'first': self.first
        }
        torch.save(net_state, net_name)

    def load_model(self, net_name):
        net_state = torch.load(net_name, map_location=self.device)
        self.ewc_parameters = net_state['ewc_param']
        self.first = net_state['first']
        self.load_state_dict(net_state['state'])

    def forward(self, *inputs):
        return self.model(*inputs)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        verbose=True
    ):
        if self.first:
            for loss_f in self.train_functions:
                if loss_f['name'] is 'ewc':
                    loss_f['weight'] = 0
        super().fit(train_loader, val_loader, epochs, patience, verbose)
        self.fisher(train_loader)
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
        self, basemodel, best=True, n_memories=256, memory_strength=0.5,
        n_classes=100, n_tasks=1, split=False
    ):
        super().__init__(basemodel, best, n_memories)
        self.margin = memory_strength
        self.n_classes = n_classes
        self.nc_per_task = n_classes / n_tasks
        self.split = split
        self.train_functions = self.model.train_functions
        self.val_functions = self.model.val_functions
        self.memory_data = [[None] * n_memories] * n_tasks
        self.memory_labs = [[None] * n_memories] * n_tasks

        # Gradient tensors
        self.grads = {
            n: [] for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def update_memory(self, x, y):
        # Update ring buffer storing examples from current task
        t = self.current_task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t][self.mem_cnt:endcnt] = x.detach()[:effbsz]
        if bsz == 1:
            self.memory_labs[t][self.mem_cnt] = y.detach()[0:]
        else:
            self.memory_labs[t][self.mem_cnt:endcnt] = y.detach()[:effbsz]
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

    def update_gradients(self):
        if len(self.observed_tasks) > 1:
            for past_task in self.observed_tasks[:-1]:
                self.zero_grad()

                if self.split:
                    offset1 = past_task * self.nc_per_task
                    offset2 = (past_task + 1) * self.nc_per_task
                else:
                    offset1 = 0
                    offset2 = self.n_classes

                output = self.forward(
                    torch.stack(self.memory_data[past_task])
                )
                batch_losses = [
                    l_f['weight'] * l_f['f'](
                        output[:, offset1:offset2],
                        torch.stack(self.memory_labs[past_task]) - offset1)
                    for l_f in self.train_functions
                ]
                sum(batch_losses).backward()
                store_grad(
                    self.parameters, self.grads, self.grad_dims,
                    past_task
                )

    def get_grad(self, indx):
        return self.grads.index_select(1, indx)

    def constraint_check(self):
        t = self.current_task
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])

            grad = self.get_grad(indx)

            dotp = torch.mm(self.grads[:, t].unsqueeze(0), grad)

            if (dotp < 0).sum() != 0:
                project2cone2(
                    self.grads[:, t].unsqueeze(1),  grad, self.margin
                )
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)

    def save_model(self, net_name):
        net_state = {
            'mem_data': self.memory_data,
            'mem_labs': self.memory_labs,
            'mem_cnt': self.mem_cnt,
            'grads': self.grads,
            'tasks': self.observed_tasks,
            'task': self.current_task,
            'first': self.first,
            'n_classes': self.n_classes,
            'nc_per_task': self.nc_per_task,
            'split': self.split,
            'state': self.state_dict()
        }
        torch.save(net_state, net_name)

    def load_model(self, net_name):
        net_state = torch.load(net_name, map_location=self.device)
        self.memory_data = net_state['mem_data']
        self.memory_labs = net_state['mem_labs']
        self.mem_cnt = net_state['mem_cnt']
        self.grads = net_state['grads']
        self.observed_tasks = net_state['tasks']
        self.current_task = net_state['task']
        self.first = net_state['first']
        self.n_classes = net_state['n_classes']
        self.nc_per_task = net_state['nc_per_task']
        self.split = net_state['split']
        self.load_state_dict(net_state['state'])

    def prebatch_update(self, batches, x, y):
        self.update_memory(x, y)
        self.update_gradients()
        self.constraint_check()


class AGEM(GEM):
    def __init__(
            self, basemodel, best=True, n_memories=256, memory_strength=0.5,
            n_classes=100, n_tasks=1, split=False
    ):
        super().__init__(
            basemodel, best, n_memories, memory_strength, n_classes, n_tasks,
            split
        )

    def get_grad(self, indx):
        return self.grads.index_select(1, indx).mean(dim=1, keepdim=True)


class SGEM(GEM):
    def __init__(
            self, basemodel, best=True, n_memories=256, memory_strength=0.5,
            n_classes=100, n_tasks=1, split=False
    ):
        super().__init__(
            basemodel, best, n_memories, memory_strength, n_classes, n_tasks,
            split
        )

    def get_grad(self, indx):
        random_indx = np.random.randint(len(self.observed_tasks[:-1]))
        indx = indx.index_select(0,
                                 torch.Tensor([random_indx]).cuda().long())
        return self.grads.index_select(1, indx)


class NGEM(GEM):
    def __init__(
            self, basemodel, best=True, n_memories=256, memory_strength=0.5,
            n_classes=100, n_tasks=1, split=False
    ):
        super().__init__(
            basemodel, best, n_memories, memory_strength, n_classes, n_tasks,
            split
        )
        self.block_grad_dims = []

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

    def constraint_check(self):
        t = self.current_task
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])

            for cnt in range(len(self.block_grad_dims)):
                beg = 0 if cnt == 0 else sum(self.block_grad_dims[:cnt])
                en = sum(self.block_grad_dims[:cnt + 1])
                if beg == en:
                    continue
                project5cone5(
                    self.grads[:, t].unsqueeze(1),
                    self.grads.index_select(1, indx),
                    beg, en, margin=self.margin)
            # copy gradients back
            overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)


class Independent(MetaModel):
    def __init__(
        self, basemodel, best=True, n_tasks=1
    ):
        super().__init__(basemodel, best, n_tasks)
        self.model = nn.ModuleList([deepcopy(basemodel) for _ in range(n_tasks)])
        self.first = True
        self.device = basemodel.device
        # Counters
        self.observed_tasks = []
        self.current_task = -1

    def forward(self, *inputs):
        return self.model[self.current_task](*inputs)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=5,
        verbose=True
    ):
        self.optimizer_alg = self.model[self.current_task + 1].optimizer_alg
        super().fit(train_loader, val_loader, epochs, patience, verbose)
        if (self.current_task + 1) < len(self.model):
            self.model[self.current_task + 1].load_state_dict(
                self.model[self.current_task].state_dict()
            )
