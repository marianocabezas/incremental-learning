import numpy as np
import torch
import miosqp
import scipy.sparse as spa
from torch.utils.data.dataset import Dataset


class MemoryContainer(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index].long()

    def __len__(self):
        return len(self.data)


class ClassificationMemoryManager(Dataset):
    def __init__(self, n_memories, n_classes, n_tasks):
        assert n_memories > 0, 'Memory cannot be 0.'
        assert n_classes > 1 and n_tasks > 1,\
            'At least one split is needed (both class and task).'
        # Init
        self.n_memories = n_memories
        self.classes = n_classes
        self.tasks = n_tasks
        self.memories_x_split = self.n_memories / self.classes
        self.data = [[] for _ in range(self.classes)]
        self.task_labels = []

    def _check_index(self, index):
        if isinstance(index, tuple) and len(index) == 2:
            index, y = index
        else:
            n_class_memories = np.cumsum([len(k_i) for k_i in self.data])
            y = np.where(index < n_class_memories)[0].min()
            if y > 0:
                index -= n_class_memories[y - 1]
        return index, y

    def _update_task_labels(self, y, t):
        if t == len(self.task_labels):
            new_labels, _ = torch.sort(torch.unique(y))
            self.task_labels.append(new_labels)
        else:
            labels = torch.cat([self.task_labels[t], y])
            new_labels, _ = torch.sort(torch.unique(labels))
            self.task_labels[t] = new_labels

    def update_memory(self, x, y, t, *args, **kwargs):
        if len(y.shape) == 2:
            y = torch.argmax(y, -1)
        self._update_task_labels(y, t)
        updated = False
        for x_i, y_i in zip(x, y):
            update = len(self.data[y_i]) < self.memories_x_split
            updated = update or updated
            if update:
                self.data[y_i].append(x_i)
        return updated

    def get_split(self, split):
        return self.data[split]

    def get_tasks(self, max_task=None):
        if max_task is None:
            max_task = len(self.task_labels)
        memory_generator = (
            MemoryContainer(*self.get_task(t)) for t in range(max_task)
        )
        return memory_generator

    def get_task(self, task):
        if task < len(self.task_labels):
            labels = self.task_labels[task]
            data = [
                x_i for label in labels for x_i in self.data[label]
            ]
            labels = [
                label for label in labels for _ in self.data[label]
            ]
        else:
            data = []
            labels = []
        return data, labels

    def get_class(self, k):
        return self.data[k], [k] * len(self.data[k])

    def __getitem__(self, index):
        index, y = self._check_index(index)
        x = self.data[y][index]

        return x, y

    def __len__(self):
        return sum([len(k_i) for k_i in self.data])


class GreedyManager(ClassificationMemoryManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)

    def update_memory(self, x, y, t, *args, **kwargs):
        if len(y.shape) == 2:
            y = torch.argmax(y, -1)
        self._update_task_labels(y, t)
        updated = False
        for x_i, y_i in zip(x, y):
            n_classes = sum([len(k_i) > 0 for k_i in self.data])
            n_class_memories = [len(k_i) for k_i in self.data]
            if n_classes > 0:
                mem_x_class = self.n_memories / n_classes
            else:
                mem_x_class = self.n_memories

            class_size = len(self.data[y_i])
            update = class_size < mem_x_class
            updated = updated or update
            if update:
                if sum(n_class_memories) >= self.n_memories:
                    big_class = np.argmax(n_class_memories)
                    self.data[big_class].pop(
                        np.random.randint(len(self.data[big_class]))
                    )
                self.data[y_i].append(x_i)

        return updated


class ClassRingBuffer(ClassificationMemoryManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)

    def update_memory(self, x, y, t, *args, **kwargs):
        if len(y.shape) == 2:
            y = torch.argmax(y, -1)
        self._update_task_labels(y, t)
        for x_i, y_i in zip(x, y):
            if len(self.data[t]) == self.memories_x_split:
                self.data[t].pop(0)
            self.data[t].append(x_i)
        return True


class GSS_Greedy(ClassificationMemoryManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)
        self.data = []
        self.labels = []
        self.scores = []
        self.grads = None
        self.task_labels = [None]

    def _get_grad_tensor(self, x, y, model):
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_tensor = torch.Tensor(sum(grad_dims), len(x))
        for i, (xi, yi) in enumerate(zip(x, y)):
            model.zero_grad()
            labels = torch.stack([yi, yi]).to(model.device)
            output = model(torch.stack([xi, xi]).to(model.device))
            batch_losses = [
                l_f['weight'] * l_f['f'](output, labels)
                for l_f in model.train_functions
            ]
            sum(batch_losses).backward()
            cnt = 0
            for param in model.parameters():
                if param.grad is not None:
                    beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                    en = sum(grad_dims[:cnt + 1])
                    grad_tensor[beg:en, i].copy_(
                        param.grad.cpu().data.view(-1)
                    )
                cnt += 1

        norm_grads = grad_tensor / torch.norm(grad_tensor, dim=0, keepdim=True)

        return norm_grads

    def update_memory(self, x, y, t, model=None, n_samples=10, *args, **kwargs):
        len_buffer = len(self.data)
        if len_buffer > 0:
            # Random sampling from buffer
            rand_indx = torch.randperm(len(self.data))[:n_samples]
            rand_x = torch.stack(self.data)[rand_indx, ...]
            rand_y = torch.stack(self.labels)[rand_indx, ...]

        else:
            # Random sampling from x
            rand_indx = torch.randperm(len(x))[:n_samples]
            rand_x = x[rand_indx, ...]
            rand_y = y[rand_indx, ...]

        rand_grads = self._get_grad_tensor(rand_x, rand_y, model)
        grads = self._get_grad_tensor(x, y, model)

        scores = torch.max(grads.t() @ rand_grads, dim=1)[0]

        for xi, yi, c in zip(x, y, scores):
            if len_buffer >= self.n_memories:
                scores = torch.stack(self.scores)
                norm_scores = scores / torch.sum(scores)
                cum_scores = torch.cumsum(norm_scores, 0)
                # Bernoulli sampling
                # i ~ P(i) = Ci / sum(Cj)
                ri = torch.rand(1)
                print(cum_scores, ri)
                i = torch.where(cum_scores > ri)[0].min()
                # r ~ uniform(0, 1)
                r = torch.rand(1)
                # if r < Ci / (Ci + c) then
                #     Mi <- (x, y); Ci <- c
                if r < scores[i] / (scores[i] + c):
                    self.data[i] = xi
                    self.labels[i] = yi
                    self.scores[i] = c
            else:
                self.data.append(xi)
                self.labels.append(yi)
                self.scores.append(c)
        return True

    def get_class(self, k):
        data = []
        labels = []
        for x_i, y_i in zip(self.data, self.labels):
            if y_i == k:
                data.append(x_i)
                labels.append(y_i)
        return data, labels

    def get_task(self, task):
        return self.data, self.labels

    def get_split(self, split):
        return self.data, self.labels


class GSS_IQP(GSS_Greedy):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)
        self.data = []
        self.labels = []
        self.solver = miosqp.MIOSQP()
        self.miosqp_settings = {
            # integer feasibility tolerance
            'eps_int_feas': 1e-03,
            # maximum number of iterations
            'max_iter_bb': 1000,
            # tree exploration rule
            #   [0] depth first
            #   [1] two-phase: depth first until first incumbent and then
            #   best bound
            'tree_explor_rule': 1,
            # branching rule
            #   [0] max fractional part
            'branching_rule': 0,
            'verbose': False,
            'print_interval': 1
        }

        self.osqp_settings = {
            'eps_abs': 1e-03,
            'eps_rel': 1e-03,
            'eps_prim_inf': 1e-04,
            'verbose': False
        }

    def update_memory(self, x, y, t, model=None, *args, **kwargs):
        self.data.extend([xi for xi in x])
        self.labels.extend([yi for yi in y])
        len_buffer = len(self.data)
        if len_buffer > self.n_memories:
            grads = self._get_grad_tensor(
                torch.stack(self.data), torch.stack(self.labels), model
            )
            inds = np.arange(0, grads.shape[-1])

            G = grads.t() @ grads
            t = G.size(0)
            G = G.double().numpy()
            a = np.zeros(t)

            C = np.ones((t, 1))
            h = np.zeros(1) + self.n_memories
            C2 = np.eye(t)

            hlower = np.zeros(t)
            hupper = np.ones(t)
            idx = np.arange(t)

            #################
            C = np.concatenate((C2, C), axis=1)
            C = np.transpose(C)
            h_final_lower = np.concatenate((hlower, h), axis=0)
            h_final_upper = np.concatenate((hupper, h), axis=0)
            #################
            G = spa.csc_matrix(G)

            C = spa.csc_matrix(C)
            self.solver.setup(
                G, a, C, h_final_lower, h_final_upper, idx, hlower, hupper,
                self.miosqp_settings, self.osqp_settings
            )
            results = self.solver.solve()
            coeffiecents_np = results.x
            coeffiecents = torch.nonzero(torch.Tensor(coeffiecents_np))
            keep = inds[coeffiecents.squeeze()]
            self.data = [
                x for xi, x in enumerate(self.data) if xi in keep
            ]
            self.labels = [
                y for yi, y in enumerate(self.labels) if yi in keep
            ]

        return True


class GSS_Graph(GSS_Greedy):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)
        self.data = []
        self.labels = []

    def update_memory(self, x, y, t, model=None, *args, **kwargs):
        self.data.extend([xi for xi in x])
        self.labels.extend([yi for yi in y])
        len_buffer = len(self.data)
        if len_buffer > self.n_memories:
            grads = self._get_grad_tensor(
                torch.stack(self.data), torch.stack(self.labels), model
            )
            # Graph pruning (we want the minimum strength graph of n_memories
            # nodes).
            adj = grads.t() @ grads
            adj.fill_diagonal_(0)
            grad_cost = torch.sum(adj, dim=0, keepdim=True)
            discard = []
            for _ in range(len_buffer - self.n_memories):
                idx = torch.argmax(grad_cost)
                discard.append(idx)
                grad_cost[:, idx] = 0
                grad_cost -= adj[idx, :]
                adj[idx, :] = 0
                adj[:, idx] = 0
            self.data = [
                x for xi, x in enumerate(self.data) if xi not in discard
            ]
            self.labels = [
                y for yi, y in enumerate(self.labels) if yi not in discard
            ]
        return True


class TaskRingBuffer(ClassificationMemoryManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)
        self.memories_x_split = self.n_memories / self.tasks
        self.labels = [[] for _ in range(self.tasks)]

    def update_memory(self, x, y, t, *args, **kwargs):
        if len(y.shape) == 2:
            y = torch.argmax(y, -1)
        self._update_task_labels(y, t)
        for x_i, y_i in zip(x, y):
            if len(self.data[t]) == self.memories_x_split:
                self.data[t].pop(0)
                self.labels[t].pop(0)
            self.data[t].append(x_i)
            self.labels[t].append(y_i)
        return True

    def get_split(self, split):
        return self.data[split], self.labels[split]

    def get_task(self, task):
        return self.data[task], self.labels[task]

    def get_class(self, k):
        data = []
        labels = []
        if k in torch.cat(self.task_labels):
            tasks = [task for task in self.task_labels if k in task]
            for task in tasks:
                for x_i, y_i in zip(self.data[task], self.labels[task]):
                    if y_i == k:
                        data.append(x_i)
                        labels.append(y_i)
        return data, labels

    def __getitem__(self, index):
        index, t = self._check_index(index)
        x = self.data[t][index]
        y = self.labels[t][index]

        return x, y


class MeanClassManager(ClassificationMemoryManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)
        self.memories_x_split = self.n_memories
        self.logits = [[] for _ in range(self.classes)]

    def _update_class_exemplars(self, x_k, logits, k):
        mean_logits = torch.mean(logits, dim=0)
        x_list = list(torch.split(x_k, 1, dim=0))
        y_list = list(torch.split(logits, 1, dim=0))
        exemplar_cost = torch.zeros(1, self.classes)
        self.logits[k] = []
        self.data[k] = []
        for ex_i in range(min(self.memories_x_split, len(x_list))):
            x_samples = len(x_list)
            mean_cost = mean_logits.expand(x_samples, self.classes)
            logits = torch.cat(y_list, dim=0)
            new_cost = (logits + exemplar_cost) / (ex_i + 1)
            cost = torch.linalg.norm(mean_cost - new_cost, dim=1)
            indx = torch.argsort(cost, 0)[0]
            self.data[k].append(x_list.pop(indx).squeeze(0))
            self.logits[k].append(y_list.pop(indx).squeeze(0))
            exemplar_cost += self.logits[k][-1]

    def update_memory(self, x, y, t, model=None, *args, **kwargs):
        if len(y.shape) == 2:
            y = torch.argmax(y, -1)
        self._update_task_labels(y, t)
        # We get the labels from the new and old tasks
        new_labels = np.unique(y)
        current_labels = np.where([len(k_i) > 0 for k_i in self.data])[0]
        if len(current_labels) > 0:
            old_mask = np.isin(current_labels, new_labels, invert=True)
            old_labels = current_labels[old_mask]
        else:
            old_labels = []
        new_classes = len(new_labels)
        old_classes = sum([len(k_i) > 0 for k_i in self.data])
        n_classes = new_classes + old_classes
        # First, we should update the number of exemplars per class
        self.memories_x_split = int(self.n_memories / n_classes)

        # < New class memories >
        for k in new_labels:
            # We construct the exemplar set for the new task.
            # We assume that the classes are new and older memories will be
            # overwritten.
            x_k = x[y == k, ...]
            logits = model(x_k.to(model.device)).detach().cpu().clone()
            self._update_class_exemplars(x_k, logits, k)

        # < Old class memories >
        # We need to shrink the older classes.
        for k in old_labels:
            x_k = torch.stack(self.data[k], dim=0)
            logits = model(x_k.to(model.device)).detach().cpu().clone()
            self._update_class_exemplars(x_k, logits, k)

        return True


class iCARLManager(MeanClassManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)

    def get_task(self, task):
        if task < len(self.task_labels):
            labels = self.task_labels[task]
            data = [
                x_i for label in labels for x_i in self.data[label]
            ]
            labels = [
                y_i for label in labels for y_i in self.logits[label]
            ]
        else:
            data = []
            labels = []
        return data, labels

    def get_class(self, k):
        return self.data[k], self.logits[k]

    def get_split(self, split):
        return self.data[split], self.logits[split]

    def __getitem__(self, index):
        index, k = self._check_index(index)
        x = self.data[k][index]
        y = self.logits[k][index]

        return x, y


class GramClassManager(ClassificationMemoryManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)
        self.memories_x_split = self.n_memories // n_classes
        self.grams = [[] for _ in range(self.classes)]

    def update_memory(self, x, y, t, model=None, *args, **kwargs):
        if len(y.shape) == 2:
            y = torch.argmax(y, -1)
        self._update_task_labels(y, t)
        for x_i, y_i in zip(x, y):
            class_size = len(self.data[y_i])
            if class_size < self.memories_x_split:
                self.data[y_i].append(x_i)
            else:
                data = x_i.unsqueeze(0).to(model.device)
                new_gram = model.gram_matrix(data).detach()
                old_grams = torch.stack(
                    self.grams[y_i], dim=0
                ).to(model.device)
                gram_diff = old_grams.unsqueeze(0) - new_gram
                gram_cost = torch.linalg.norm(gram_diff.flatten(1), dim=1)
                indx = torch.argsort(gram_cost, 0)[0]
                self.grams[y_i].pop(indx)
                self.grams[y_i].append(new_gram.cpu())
                self.data[y_i].pop(indx)
                self.data[y_i].append(x_i.detach().cpu())

        return True


class PrototypeClassManager(ClassificationMemoryManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)
        self.memories_x_split = self.n_memories // n_classes
        self.grams = [[] for _ in range(self.classes)]
        self.logits = [[] for _ in range(self.classes)]
        # Prototypes per class
        self.max_types = max(1, self.memories_x_split // self.classes)

    def update_memory(self, x, y, t, model=None, *args, **kwargs):
        if len(y.shape) == 2:
            y = torch.argmax(y, -1)
        self._update_task_labels(y, t)
        # We reset the grams and logits because the network might have been
        # updated.
        self.grams = [[] for _ in range(self.classes)]
        self.logits = [[] for _ in range(self.classes)]
        for x_i, y_i in zip(x, y):
            class_size = len(self.data[y_i])
            # If the current class buffer is not full we just push the new
            # memories.
            if class_size < self.memories_x_split:
                self.data[y_i].append(x_i)
            else:
                # If the class is full, things get interesting and we need
                # to prepare the intermediate logits and grams.
                if len(self.logits[y_i]) == 0:
                    data = torch.stack(self.data[y_i]).to(model.device)
                    self.logits[y_i] = list(
                        torch.split(model(data).detach(), 1, dim=0)
                    )
                if len(self.grams[y_i]) == 0:
                    data = torch.stack(self.data[y_i]).to(model.device)
                    grams = model.gram_matrix(data).detach()
                    self.grams[y_i] = [
                        g[torch.triu(torch.ones_like(g)) > 0]
                        for g in torch.split(grams, 1, dim=0)
                    ]

                # Now we need to get logit and gram of the new memory and
                # push it into the manager. That makes comparison later
                # easier. That also means we might be pushing a new memory
                # we might need to remove.
                data = x_i.unsqueeze(0).to(model.device)
                new_gram = model.gram_matrix(data).detach()
                gram_mask = torch.triu(torch.ones_like(new_gram)) > 0
                new_logit = model(data).detach()
                prototype = torch.argmax(new_logit)
                self.grams[y_i].append(new_gram[gram_mask])
                self.data[y_i].append(x_i)
                self.logits[y_i].append(new_logit)

                # Now it's time to check the current distribution of
                # prototypes. The goal is to have a balanced number of
                # prototypes and only keep the prototypes the furthest
                # from the assumed Normal distribution of grams.
                old_types = torch.argmax(
                    torch.cat(self.logits[y_i], dim=0), dim=1
                )
                grams = torch.stack(self.grams[y_i], dim=0)
                class_mask = old_types == prototype
                # If the current prototype buffer is not at full capacity
                # we need to look for the biggest prototype buffer and
                # remove a memory from there.
                if torch.sum(class_mask) <= self.max_types:
                    type_count = torch.stack([
                        torch.sum(old_types == t)
                        for t in range(self.classes)
                    ])
                    prototype = torch.argmax(type_count)
                    class_mask = old_types == prototype
                # Once the prototypes are selected it's just a matter of
                # finding the "closest one" to the distribution with the
                # Mahalanobis distance.
                # Due to the large number of features with low values, we
                # ignore the covariances between features (which erroneously
                # assumes independence) and we ignore variances of 0.
                t_grams = grams[class_mask, ...]
                indices = torch.where(class_mask)[0]
                grams_mean = t_grams.mean(0, keepdim=True)
                gram_cov = torch.cov(t_grams.t())
                variances = torch.diag(gram_cov)
                variances[variances > 0] = 1 / variances[variances > 0]
                grams_cov_i = torch.diag(variances)
                grams_norm = t_grams - grams_mean
                distances0 = [
                    torch.sqrt(
                        (g_norm @ grams_cov_i) @ g_norm.t()
                    ).cpu().numpy().tolist()
                    for g_norm in grams_norm
                ]
                indx = indices[np.argmin(distances0)]
                self.grams[y_i].pop(indx)
                self.data[y_i].pop(indx)
                self.logits[y_i].pop(indx)

        return True


class NewPrototypeClassManager(ClassificationMemoryManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)
        self.memories_x_split = self.n_memories // n_classes
        # Prototypes per class
        self.max_types = max(1, self.memories_x_split // self.classes)

    def update_memory(self, x, y, t, model=None, *args, **kwargs):
        if len(y.shape) == 2:
            y = torch.argmax(y, -1)
        self._update_task_labels(y, t)
        # We reset the grams and logits because the network might have been
        # updated.
        for x_i, y_i in zip(x, y):
            self.data[y_i].append(x_i)

        for y_i in sorted(np.unique(y.cpu())):
            class_size = len(self.data[y_i])
            # If the buffer for the class is full, things get interesting, and
            # we need to prepare the intermediate logits and the class gram
            # matrix.
            if class_size > self.memories_x_split:
                extra = (class_size - self.memories_x_split)
                data = torch.stack(self.data[y_i]).to(model.device)
                prototypes = torch.argmax(model(data), dim=1).cpu()
                features = model.features(data).cpu()

                # Prototype check.
                # We want to represent as many "prototypes" as possible.
                # That means we need to see which "prototypes" have
                # more representation and remove examples from them.
                # This involves the following steps:
                # 1 - Counting the number of examples per prototype
                # and sorting them:
                proto_count = torch.stack([
                    torch.count_nonzero(prototypes == k)
                    for k in sorted(torch.unique(prototypes))
                ])
                proto_sort, proto_idx = torch.sort(
                    proto_count, descending=True
                )

                # 2 - Checking which of these prototypes need to be depleted.
                # While the idea in principle is simple, the implementation
                # is not. To avoid doing it example by example which takes
                # time (for loops are slow), we need to:
                # 2.1 get the differential between sorted prototypes.
                #  Note: The last prototype has no differential.
                proto_diff = proto_sort - torch.roll(proto_sort, -1)
                proto_diff[-1] = 0

                # 2.2 take into account that these differentials
                #  "increase" as we go down the list of sorted
                #  prototypes. What that means is that if we have
                #  two classes of prototypes we need to deplete,
                #  we will have to evenly distribute the number
                #  of removed examples. Consequently, the differential
                #  with respect to the 3rd class is now doubled.
                proto_diffcum = proto_diff * np.arange(1, len(proto_diff) + 1)
                proto_extra = torch.cumsum(proto_diffcum, 0) - extra

                # 2.3 determine which prototype classes need
                #  to be depleted with the previous info.
                #  The cumulative extra gives us an idea of
                #  after which prototype class we will reach
                #  enough depletion. Ideally, a prototype class
                #  with 0 would be the stopping point. However,
                #  the most likely scenario is that we go from
                #  negative to positive.
                empty_classes = torch.where(proto_extra > 0)[0]
                if len(empty_classes) > 0:
                    last_class = empty_classes[0].numpy().tolist()
                else:
                    last_class = len(proto_extra) - 1
                extra_idx = proto_idx[:last_class + 1].numpy()

                # 2.4 if we reach a positive value (not 0),
                #  we can define the fixed amount of samples
                #  to remote per class.
                fixed_array = np.cumsum(
                    proto_diff.numpy()[:last_class][::-1]
                ).tolist()[::-1]
                fixed_array += [0]
                fixed_array = np.array(fixed_array, dtype=np.uint8)
                fixed_extra = np.sum(fixed_array)

                # 2.5 finally, we only need to distribute the examples
                #  among the prototypes. We will start by penalizing
                #  big classes first. For example, if we have 3 extra
                #  memories and 4 prototype classes, we will remove
                #  memories from the first 3.
                tosplit_extra = int(extra - fixed_extra)
                tosplit_classes = last_class + 1
                split_extra = tosplit_extra // tosplit_classes
                remain_extra = tosplit_extra % tosplit_classes

                # 2.6 put all that together! It's a matter of
                #  merging the "fixed" part, the "split" fraction
                #  and the remainder.
                tosplit = np.ones(tosplit_classes, dtype=np.uint8)
                split_array = split_extra * tosplit

                remain_array = np.ones(tosplit_classes, dtype=np.uint8)
                remain_array[remain_extra:] = 0

                final_array = fixed_array + split_array + remain_array

                # Gram computation.
                # This is a proxy to see how correlated the memories are.
                # We want "unique" memories, so we need to discard memories
                # that are "similar".
                del_idx = []
                for k, n_del_mem in zip(extra_idx, final_array):
                    k_mask = prototypes == k
                    feat_k = features[k_mask]
                    k_indices = np.where(k_mask)[0]
                    gram = (feat_k @ feat_k.t()) * (1 - torch.eye(len(feat_k)))
                    gram_process = torch.mean(gram, dim=1)
                    _, gram_idx = torch.sort(gram_process, descending=True)
                    del_idx.extend(k_indices[gram_idx[:n_del_mem].numpy()].tolist())
                self.data[y_i] = [
                    elem for n, elem in enumerate(self.data[y_i])
                    if n not in del_idx
                ]

        return True
