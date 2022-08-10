import numpy as np
import torch
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

    def get_tasks(self):
        memory_generator = (
            MemoryContainer(*self.get_task(t))
            for t in range(len(self.task_labels))
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
        self._update_task_labels(y, t)
        for x_i, y_i in zip(x, y):
            if len(self.data[t]) == self.memories_x_split:
                self.data[t].pop(0)
            self.data[t].append(x_i)
        return True


class TaskRingBuffer(ClassificationMemoryManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)
        self.memories_x_split = self.n_memories / self.tasks
        self.labels = [[] for _ in range(self.tasks)]

    def update_memory(self, x, y, t, *args, **kwargs):
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
                t_grams = grams[class_mask, ...]
                indices = torch.where(class_mask)[0]
                grams_mean = t_grams.mean(0, keepdim=True)
                true_cov = torch.cov(t_grams.t())
                grams_cov = torch.diag(torch.diag(true_cov))
                grams_cov_i = torch.inverse(grams_cov)
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
