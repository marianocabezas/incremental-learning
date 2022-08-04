import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class MemoryContainer(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

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
        for x_i, y_i in zip(x, y):
            if len(self.data[y_i]) < self.memories_x_split:
                self.data[y_i].append(x_i)

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
                torch.tensor(label, dtype=torch.uint8)
                for label in labels for _ in self.data[label]
            ]
        else:
            data = []
            labels = []
        return data, labels

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
        for x_i, y_i in zip(x, y):
            n_classes = sum([len(k_i) > 0 for k_i in self.data])
            n_class_memories = [len(k_i) for k_i in self.data]
            if n_classes > 0:
                mem_x_class = self.n_memories / n_classes
            else:
                mem_x_class = self.n_memories

            class_size = len(self.data[y_i])
            if class_size == 0 or class_size < mem_x_class:
                if sum(n_class_memories) >= self.n_memories:
                    big_class = np.argmax(n_class_memories)
                    self.data[big_class].pop(
                        np.random.randint(len(self.data[big_class]))
                    )
                self.data[y_i].append(x_i)


class ClassRingBuffer(ClassificationMemoryManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)

    def update_memory(self, x, y, t, *args, **kwargs):
        self._update_task_labels(y, t)
        for x_i, y_i in zip(x, y):
            if len(self.data[t]) == self.memories_x_split:
                self.data[t].pop(0)
            self.data[t].append(x_i)


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

    def get_split(self, split):
        return self.data[split], self.labels[split]

    def get_task(self, task):
        return self.data[task], self.labels[task]

    def __getitem__(self, index):
        index, t = self._check_index(index)
        x = self.data[t][index]
        y = self.labels[t][index]

        return x, y


class iCARLManager(ClassificationMemoryManager):
    def __init__(self, n_memories, n_classes, n_tasks):
        super().__init__(n_memories, n_classes, n_tasks)
        self.memories_x_split = self.n_memories
        self.labels = [[] for _ in range(self.classes)]

    def _update_class_exemplars(self, x_k, logits, k, n_classes):
        mean_logits = torch.mean(logits, dim=0)
        x_list = torch.split(x_k, 1, dim=0)
        y_list = torch.split(logits, 1, dim=0)
        exemplar_cost = torch.zeros(1, n_classes)
        for ex_i in range(min(self.memories_x_split, len(x_list))):
            x_samples = len(x_list)
            mean_cost = mean_logits.expand(x_samples, self.classes)
            logits = torch.stack(y_list, dim=0)
            new_cost = (logits + exemplar_cost) / (ex_i + 1)
            cost = torch.linalg.norm(mean_cost - new_cost, dim=1)
            indx = torch.argsort(cost, 0)[0]
            self.data[k].append(x_list.pop(indx))
            self.labels[k].append(y_list.pop(indx))
            exemplar_cost += self.data[k][-1]

    def update_memory(self, x, y, t, model=None, *args, **kwargs):
        self._update_task_labels(y, t)
        # We get the labels from the new and old tasks
        new_labels = np.unique(y)
        old_labels = np.isin(
            np.where([len(k_i) > 0 for k_i in self.data]),
            new_labels, invert=True
        )
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
            logits = model(x_k.to(model.device)).cpu()
            self._update_class_exemplars(x_k, logits, k, n_classes)

        # < Old class memories >
        # We need to shrink the older classes.
        for k in old_labels:
            x_k = torch.stack(self.data[k], dim=0)
            logits = model(x_k.to(model.device)).cpu()
            self._update_class_exemplars(x_k, logits, k, n_classes)

    def get_task(self, task):
        if task < len(self.task_labels):
            labels = self.task_labels[task]
            data = [
                x_i for label in labels for x_i in self.data[label]
            ]
            labels = [
                y_i for label in labels for y_i in self.labels[label]
            ]
        else:
            data = []
            labels = []
        return data, labels

    def get_split(self, split):
        return self.data[split], self.labels[split]

    def __getitem__(self, index):
        index, k = self._check_index(index)
        x = self.data[k][index]
        y = self.labels[k][index]

        return x, y
