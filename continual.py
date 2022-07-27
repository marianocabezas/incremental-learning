from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import quadprog
from sklearn.decomposition import PCA
from base import BaseModel


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
                param.grad.data.size()
            )
            param.grad.data.copy_(this_grad.to(param.device))
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
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    return torch.Tensor(x).view(-1, 1)
    # gradient.copy_(torch.Tensor(x).view(-1, 1))


def project5cone5(gradient, memories, beg, en, margin=0.5, eps=1e-3):
    """
        First we do orthogonality of memories and then find the null space of
        their gradients
    """
    np.seterr(divide='ignore', invalid='ignore')
    memories_np = np.nan_to_num(
        memories[beg:en].t().double().numpy()
    ).astype(np.float32)
    gradient_np = np.nan_to_num(
        gradient[beg:en].contiguous().view(-1).double().numpy()
    ).astype(np.float32)
    memories_np_sum = np.sum(memories_np, axis=0)
    if len(memories_np) == 1:
        x = gradient_np - np.min([
            (memories_np_sum.transpose().dot(gradient_np) /
             memories_np_sum.transpose().dot(memories_np_sum)), -margin
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
                memories_np_sum.transpose().dot(Pg_bar), -margin
            ]) * memories_np_sum - memories_np_orth.transpose().dot(
                memories_np_orth.dot(gradient_np)) + memories_np_sum.transpose(
                ).dot(Pg) / memories_np_sum.transpose().dot(
                    Pg_bar) * memories_np_orth.transpose().dot(
                        memories_np_orth.dot(memories_np_sum))

    gradient[beg:en].copy_(
        torch.Tensor(np.nan_to_num(x).astype(np.float32)).view(-1, 1)
    )

    # memories_tensor = memories[beg:en].t()
    # gradient_tensor = gradient[beg:en].contiguous().view(-1)
    # memories_mean = torch.mean(memories_tensor, axis=0)
    # memories_sum = torch.sum(memories_tensor, axis=0)
    #
    # if len(memories_tensor) == 1:
    #     x = gradient_tensor - np.min([
    #         (memories_sum.t().dot(gradient_tensor) /
    #          memories_sum.t().dot(memories_sum)).cpu(), - margin
    #     ]) * memories_sum
    # else:
    #     memories_del_mean = memories_tensor - memories_mean.reshape(1, -1)
    #     memories_orth, _, _ = torch.pca_lowrank(memories_del_mean, q=min(2, len(memories)))
    #     memories_orth = memories_orth.t()
    #     Pg = gradient_tensor - memories_orth.t().dot(
    #         memories_orth.dot(gradient_tensor))
    #     Pg_bar = memories_sum - memories_orth.t().dot(
    #         memories_orth.dot(memories_sum))
    #     if memories_sum.t().dot(Pg) > 0:
    #         x = Pg
    #     else:
    #         x = gradient_tensor - np.min([
    #             memories_sum.t().dot(Pg) /
    #             memories_sum.t().dot(Pg_bar), -margin
    #         ]) * memories_sum - memories_orth.t().dot(
    #             memories_orth.dot(gradient_tensor)) + memories_sum.t(
    #         ).dot(Pg) / memories_sum.t().dot(
    #             Pg_bar) * memories_orth.t().dot(
    #             memories_orth.dot(memories_sum))
    #
    # gradient[beg:en].copy_(x.view(-1, 1))


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
        self.offset1 = None
        self.offset2 = None

        self.train_functions = self.model.train_functions
        self.val_functions = self.model.val_functions

        self.optimizer_alg = self.model.optimizer_alg

    def reset_optimiser(self):
        super().reset_optimiser()
        self.model.reset_optimiser()
        self.optimizer_alg = self.model.optimizer_alg

    def get_state(self):
        net_state = {
            'first': self.first,
            'task': self.current_task,
            'state': self.state_dict()
        }
        return net_state

    def save_model(self, net_name):
        net_state = self.get_state()
        torch.save(net_state, net_name)

    def load_model(self, net_name):
        net_state = torch.load(net_name, map_location=self.device)
        self.first = net_state['first']
        self.current_task = net_state['task']
        self.load_state_dict(net_state['state'])
        return net_state

    def forward(self, *inputs):
        return self.model(*inputs)

    def observe(self, x, y):
        pred_labels, x_cuda, y_cuda = super().observe(x, y)
        if self.offset1 is not None and self.offset2 is not None:
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
        n_classes=100, n_tasks=1
    ):
        super().__init__(basemodel, best, n_memories)
        self.margin = memory_strength
        self.n_classes = n_classes
        self.nc_per_task = n_classes // n_tasks
        self.train_functions = self.model.train_functions
        self.val_functions = self.model.val_functions
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        self.memory_data = [[] for _ in range(n_tasks)]
        self.memory_labs = [[] for _ in range(n_tasks)]
        self.offsets = []

    def update_memory(self, x, y):
        # Update ring buffer storing examples from the current task
        t = self.current_task
        current_size = len(self.memory_data[t])
        empty_slots = self.n_memories - current_size
        end_slots = self.n_memories - self.mem_cnt
        batch_size = y.data.size(0)
        if empty_slots > 0 or end_slots < batch_size:
            new_slots = min(batch_size, max(empty_slots, end_slots))
            new_mem = list(torch.split(
                x[:new_slots, ...].detach().cpu(), 1
            ))
            new_labs = list(torch.split(
                y[:new_slots, ...].detach().cpu(), 1
            ))
            if empty_slots > 0:
                self.memory_data[t] += new_mem
                self.memory_labs[t] += new_labs
            else:
                self.memory_data[t][self.mem_cnt:] = new_mem
                self.memory_labs[t][self.mem_cnt:] = new_labs
            current_size = len(self.memory_data[t])
            x = x[new_slots:, ...]
            y = y[new_slots:, ...]
            if current_size < self.n_memories:
                self.mem_cnt = current_size
            else:
                self.mem_cnt = 0
        batch_size = y.data.size(0)
        if batch_size > 0:
            end_cnt = min(self.mem_cnt + batch_size, self.n_memories)
            end_mem = end_cnt - self.mem_cnt
            new_mem = list(torch.split(
                x[:end_mem, ...].detach().cpu(), 1
            ))
            new_labs = list(torch.split(
                y[:end_mem, ...].detach().cpu(), 1
            ))
            self.memory_data[t][self.mem_cnt:end_cnt] = new_mem
            self.memory_labs[t][self.mem_cnt:end_cnt] = new_labs
            self.mem_cnt += end_mem
            if self.mem_cnt == self.n_memories:
                self.mem_cnt = 0

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

    def update_gradients(self):
        if len(self.observed_tasks) > 1:
            for past_task, (offset1, offset2) in zip(
                self.observed_tasks[:-1], self.offsets[:-1]
            ):
                self.zero_grad()

                memories_t = torch.cat(self.memory_data[past_task])
                labels_t = torch.cat(self.memory_labs[past_task])

                output = self(memories_t.to(self.device))
                batch_losses = [
                    l_f['weight'] * l_f['f'](
                        output[:, offset1:offset2],
                        labels_t.to(self.device)
                    )
                    for l_f in self.train_functions
                ]
                sum(batch_losses).backward()
                self.store_grad(past_task)

    def get_grad(self, indx):
        return self.grads.index_select(1, indx)

    def constraint_check(self):
        t = self.current_task
        # Copy the current gradient
        indx = torch.LongTensor(
            self.observed_tasks[:-1]
        )
        grad = self.get_grad(indx)
        grad_t = self.get_grad(torch.LongTensor(t))
        if len(self.observed_tasks) > 1:
            dotp = torch.mm(
                grad_t.unsqueeze(0).to(self.device),
                grad.to(self.device)
            )

            if (dotp < 0).any():
                grad_t = project2cone2(
                    grad_t.unsqueeze(1), grad, self.margin
                )
                # Copy gradients back
                overwrite_grad(
                    self.parameters, grad_t, self.grad_dims
                )
        return grad_t

    def get_state(self):
        net_state = super().get_state()
        net_state['grad_dims'] = self.grad_dims
        net_state['mem_data'] = self.memory_data
        net_state['mem_labs'] = self.memory_labs
        net_state['mem_cnt'] = self.mem_cnt
        net_state['grads'] = self.grads
        net_state['tasks'] = self.observed_tasks
        net_state['n_classes'] = self.n_classes
        net_state['nc_per_task'] = self.nc_per_task
        return net_state

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.grad_dims = net_state['grad_dims']
        self.memory_data = [
            [mem.cpu() for mem in memories]
            for memories in net_state['mem_data']
        ]
        self.memory_labs = [
            [mem.cpu() for mem in memories]
            for memories in net_state['mem_labs']
        ]
        self.mem_cnt = net_state['mem_cnt']
        if type(net_state['grads']) is list:
            self.grads = [grad.cpu() for grad in net_state['grads']]
        else:
            self.grads = net_state['grads'].cpu()
        self.observed_tasks = net_state['tasks']
        self.n_classes = net_state['n_classes']
        self.nc_per_task = net_state['nc_per_task']

        return net_state

    def prebatch_update(self, batch, batches, x, y):
        self.update_memory(x, y)
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
            self, basemodel, best=True, n_memories=256, memory_strength=0.5,
            n_classes=100, n_tasks=1
    ):
        super().__init__(
            basemodel, best, n_memories, memory_strength, n_classes, n_tasks,
        )

    def get_grad(self, indx):
        return self.grads.index_select(1, indx).mean(dim=1, keepdim=True)


class SGEM(GEM):
    def __init__(
            self, basemodel, best=True, n_memories=256, memory_strength=0.5,
            n_classes=100, n_tasks=1
    ):
        super().__init__(
            basemodel, best, n_memories, memory_strength, n_classes, n_tasks,
        )

    def get_grad(self, indx):
        random_indx = np.random.randint(len(self.observed_tasks[:-1]))
        indx = indx.index_select(
            0, torch.Tensor([random_indx]).long()
        )
        return self.grads.index_select(1, indx)


class NGEM(GEM):
    def __init__(
            self, basemodel, best=True, n_memories=256, memory_strength=0.5,
            n_classes=100, n_tasks=1
    ):
        super().__init__(
            basemodel, best, n_memories, memory_strength, n_classes, n_tasks,
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
            self.store_grad(t)
            indx = torch.LongTensor(self.observed_tasks[:-1])

            for cnt in range(len(self.block_grad_dims)):
                beg = 0 if cnt == 0 else sum(self.block_grad_dims[:cnt])
                en = sum(self.block_grad_dims[:cnt + 1])
                if beg == en:
                    continue
                # gradient_gpu = self.grads[:, t].unsqueeze(1).to(self.device)
                project5cone5(
                    self.grads[:, t].unsqueeze(1),
                    self.grads.index_select(1, indx),
                    # gradient_gpu,
                    # self.grads.index_select(1, indx).to(self.device),
                    beg, en, margin=self.margin
                )
                # self.grads[:, t] = gradient_gpu.squeeze(1).cpu()
            # copy gradients back
            overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)

    def get_state(self):
        net_state = super().get_state()
        net_state['block_dims'] = self.block_grad_dims
        return net_state

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.block_grad_dims = net_state['block_dims']


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

    def reset_optimiser(self):
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


class ParamGEM(GEM):
    def __init__(
        self, basemodel, best=True, n_memories=256, memory_strength=0.5,
        n_classes=100, n_tasks=1
    ):
        super().__init__(basemodel, best, n_memories)
        self.margin = memory_strength
        self.n_classes = n_classes
        self.nc_per_task = n_classes // n_tasks
        self.train_functions = self.model.train_functions
        self.val_functions = self.model.val_functions
        self.grads = []
        for param in self.model.parameters():
            if param.requires_grad:
                self.grads.append(torch.Tensor(param.data.numel(), n_tasks))
        self.memory_data = [[] for _ in range(n_tasks)]
        self.memory_labs = [[] for _ in range(n_tasks)]
        self.offsets = []

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
            indx = torch.LongTensor(
                self.observed_tasks[:-1]
            )
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
                            new_grad = project2cone2(
                                current_grad.unsqueeze(1), grad, self.margin
                            )
                            # Copy the new gradient
                            current_grad = new_grad.contiguous().view(
                                param.grad.data.size()
                            )
                            param.grad.data.copy_(current_grad.to(param.device))
                    p += 1


class iCARL(MetaModel):
    def __init__(
        self, basemodel, best=True, n_memories=256, memory_strength=0.5,
        n_classes=100, n_tasks=1
    ):
        super().__init__(basemodel, best, n_memories)
        self.n_classes = n_classes
        self.nc_per_task = n_classes // n_tasks
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
        self.mem_class_x = {}  # stores exemplars class by class
        self.mem_class_y = {}
        self.offsets = []

    def distillation_loss(self):
        losses = []
        for past_task, (offset1, offset2) in zip(
                self.observed_tasks[:-1], self.offsets[:-1]
        ):
            # first generate a minibatch with one example per class from
            # previous tasks
            x = []
            y_logits = []
            for k in range(self.nc_per_task):
                x_k = self.mem_class_x[k + offset1]
                y_k = self.mem_class_y[k + offset1]
                indx = torch.random.randint(0, len(x_k) - 1)
                x.append(x_k[indx].clone())
                y_logits.append(y_k[indx].clone())
            x = torch.stack(x, dim=0).to(self.device)
            y_logits = torch.stack(y_logits, dim=0).to(self.device)

            # Add distillation loss
            prediction = F.log_softmax(
                self.model(x)[:, offset1:offset2], 1
            )
            y = F.softmax(y_logits[:, offset1:offset2], 1)
            losses.append(
                F.kl_div(prediction, y) * self.nc_per_task
            )
        return sum(losses)

    def prebatch_update(self, batch, batches, x, y):
        if self.memx is None:
            self.memx = x.cpu().data.clone()
            self.memy = y.cpu().data.clone()
        else:
            self.memx = torch.cat((self.memx, x.cpu().data.clone()))
            self.memy = torch.cat((self.memy, y.cpu().data.clone()))

    def batch_update(self, batch, batches, x, y):
        if (batch + 1) == batches:
            # Get labels from previous task; we assume labels are consecutive
            all_labs = torch.LongTensor(np.unique(self.memy.numpy()))
            num_classes = all_labs.size(0)
            assert (num_classes == self.nc_per_task)
            # Reduce exemplar set by updating value of num. exemplars per class
            self.num_exemplars = int(
                self.n_memories / (num_classes + len(self.mem_class_x.keys())))
            offset_slice = slice(self.offset1, self.offset2)
            for k in all_labs:
                indxs = (self.memy == k).nonzero(as_tuple=False).squeeze()
                cdata = self.memx.index_select(
                    0, indxs
                ).to(self.device)  # cdata are exemplar whose label == lab

                # Construct exemplar set for last task
                model_output = self.model(
                    cdata
                )[:, offset_slice].data.clone()
                mean_feature = model_output.mean(0)
                exemplars = torch.zeros(
                    (self.num_exemplars,) + x.shape[1:],
                    device=self.device
                )
                batch_size = cdata.size(0)
                # used to keep track of which examples we have already used
                taken = torch.zeros(batch_size)
                prev = torch.zeros(1, self.nc_per_task).to(self.device)
                for ee in range(self.num_exemplars):
                    # if ee > 0:
                    #     print(x.shape, exemplars.shape, exemplars[:ee].shape)
                    #     prev = self.model(
                    #         exemplars[:ee]
                    #     )[:, offset_slice].data.clone().sum(0)
                    mean_cost = mean_feature.expand(batch_size, self.nc_per_task)
                    output_cost = model_output + prev.expand(batch_size, self.nc_per_task)
                    cost = (mean_cost - output_cost / (ee + 1)).norm(
                            2, 1
                    ).squeeze()
                    _, indx = cost.sort(0)
                    winner = 0
                    while winner < indx.size(0) and taken[indx[winner]] == 1:
                        winner += 1
                    if winner < indx.size(0):
                        taken[indx[winner]] = 1
                        exemplars[ee] = cdata[indx[winner]].clone()
                        prev += model_output[indx[winner], offset_slice].data.clone()
                    else:
                        exemplars = exemplars[:indx.size(0), :].clone()
                        self.num_exemplars = indx.size(0)
                        break
                # update memory with exemplars
                self.mem_class_x[k] = exemplars.cpu().clone()

            # recompute outputs for distillation purposes
            for k in self.mem_class_x.keys():
                self.mem_class_y[k] = self.model(
                    self.mem_class_x[k].to(self.device)
                ).cpu().data.clone()
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
        if net_state['memx'] is not None:
            self.memx = net_state['memx'].cpu()
        else:
            self.memx = None
        if net_state['memy'] is not None:
            self.memy = net_state['memy'].cpu()
        else:
            self.memy = None
        self.mem_class_x = {
            key: data.cpu()
            for key, data in net_state['mem_class_x'].items()
        }  # stores exemplars class by class
        self.mem_class_y = {
            key: data.cpu()
            for key, data in net_state['mem_class_y'].items()
        }  # stores exemplars class by class
        self.n_classes = net_state['n_classes']
        self.nc_per_task = net_state['nc_per_task']

        return net_state

    def get_state(self):
        net_state = super().get_state()
        net_state['mem_class_x'] = self.mem_class_x
        net_state['mem_class_y'] = self.mem_class_y
        net_state['memx'] = self.memx
        net_state['memy'] = self.memy
        net_state['n_classes'] = self.n_classes
        net_state['nc_per_task'] = self.nc_per_task
        return net_state

    def save_model(self, net_name):
        net_state = {
            'mem_class_x': self.mem_class_x,
            'mem_class_y': self.mem_class_y,
            'memx': self.memx,
            'memy': self.memy,
            'n_classes': self.n_classes,
            'nc_per_task': self.nc_per_task,
        }
        torch.save(net_state, net_name)


class LoggingGEM(GEM):
    def __init__(
        self, basemodel, best=True, n_memories=256, memory_strength=0.5,
        n_classes=100, n_tasks=1
    ):
        super().__init__(
            basemodel, best, n_memories, memory_strength, n_classes, n_tasks
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
            self.grad_log['norm_dot'].append(old_norm @ norm_grads[:, :-1])
            self.grad_log['new_dot'].append(
                new_grad.transpose() @ grads[:, :-1]
            )
            self.grad_log['norm_new_dot'].append(
                new_norm.transpose() @ norm_grads[:, :-1]
            )
            self.grad_log['grads_dot'].append(old_grad @ new_grad)
            self.grad_log['norm_grads_dot'].append(old_norm @ new_norm)

    def get_state(self):
        net_state = super().get_state()
        net_state['log'] = self.grad_log
        return net_state

    def load_model(self, net_name):
        net_state = super().load_model(net_name)
        self.grad_log = net_state['log']

        return net_state
