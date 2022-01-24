import time
import itertools
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel, ResConv3dBlock
from base import Autoencoder, AttentionAutoencoder, DualAttentionAutoencoder
from utils import time_to_string, to_torch_var
from criteria import gendsc_loss, similarity_loss, grad_loss, accuracy
from criteria import tp_binary_loss, tn_binary_loss, dsc_binary_loss
from criteria import focal_loss


def norm_f(n_f):
    return nn.GroupNorm(n_f // 4, n_f)


def print_batch(pi, n_patches, i, n_cases, t_in, t_case_in):
    init_c = '\033[38;5;238m'
    percent = 25 * (pi + 1) // n_patches
    progress_s = ''.join(['█'] * percent)
    remainder_s = ''.join([' '] * (25 - percent))

    t_out = time.time() - t_in
    t_case_out = time.time() - t_case_in
    time_s = time_to_string(t_out)

    t_eta = (t_case_out / (pi + 1)) * (n_patches - (pi + 1))
    eta_s = time_to_string(t_eta)
    pre_s = '{:}Case {:03d}/{:03d} ({:03d}/{:03d} - {:06.2f}%) [{:}{:}]' \
            ' {:} ETA: {:}'
    batch_s = pre_s.format(
        init_c, i + 1, n_cases, pi + 1, n_patches, 100 * (pi + 1) / n_patches,
        progress_s, remainder_s, time_s, eta_s + '\033[0m'
    )
    print('\033[K', end='', flush=True)
    print(batch_s, end='\r', flush=True)


class SimpleUNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.segmenter = nn.Sequential(
            Autoencoder(
                self.conv_filters, device, n_images, block=ResConv3dBlock,
                norm=norm_f
            ),
            nn.Conv3d(self.conv_filters[0], 1, 1)
        )
        self.segmenter.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'pdsc',
                'weight': 1,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            },
            {
                'name': 'pdsc',
                'weight': 0,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_binary_loss(p, t)
            },
            {
                'name': 'fn',
                'weight': 0,
                'f': lambda p, t: tp_binary_loss(p, t)
            },
            {
                'name': 'fp',
                'weight': 0,
                'f': lambda p, t: tn_binary_loss(p, t)
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self):
        super().reset_optimiser()
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)

    def forward(self, data):

        return torch.sigmoid(self.segmenter(data))


class MetaModel(BaseModel):
    def __init__(
        self, basemodel, ewc_weight=1e6, ewc_binary=True,
            ewc_alpha=None
    ):
        super().__init__()
        self.init = False
        self.first = True
        self.model = basemodel
        self.device = basemodel.device
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

        return sum(losses)

    def fisher(self, dataloader):
        self.model.eval()
        new_fisher = {
            n: torch.zeros_like(p.data)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        for batch_i, (x, y) in enumerate(dataloader):
            # In case we are training the the gradient to zero.
            self.model.zero_grad()

            # First, we do a forward pass through the network.
            if isinstance(x, list) or isinstance(x, tuple):
                x_cuda = tuple(x_i.to(self.device) for x_i in x)
                pred_labels = self(*x_cuda)
            else:
                pred_labels = self(x.to(self.device))

            y_cuda = y.type_as(pred_labels).to(self.device)

            if self.ewc_binary:
                loss = F.binary_cross_entropy(
                    pred_labels, y_cuda
                )
            else:
                loss = F.nll_loss(
                    torch.log(pred_labels), y_cuda
                )
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    grad = p.grad.data.detach() ** 2 / len(dataloader)
                    new_fisher[n] += grad

        if self.ewc_alpha is None:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.ewc_parameters[n]['fisher'].append(
                        new_fisher[n]
                    )
                    self.ewc_parameters[n]['means'].append(
                        p.data.detach()
                    )
        else:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    self.ewc_parameters[n]['means'] = p.data.detach()
                    if self.first:
                        self.ewc_parameters[n]['fisher'] = new_fisher[n]
                    else:
                        prev_fisher = self.ewc_parameters[n]['fisher']
                        fisher_t0 = (1 - self.ewc_alpha) * prev_fisher
                        fisher_t1 = self.ewc_alpha * new_fisher[n]
                        self.ewc_parameters[n]['fisher'] = (
                            fisher_t0 + fisher_t1
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
        if self.ewc_alpha is None:
            self.fisher(train_loader)
        for loss_f in self.train_functions:
            if loss_f['name'] is 'ewc':
                loss_f['weight'] = self.ewc_weight


class SimpleResNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            verbose=0,
    ):
        super().__init__()
        # self.init = False
        self.init = True
        # Init values
        if conv_filters is None:
            self.conv_filters = [32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.extractor = Autoencoder(
            self.conv_filters, device, n_images, block=ResConv3dBlock,
            norm=norm_f
        )
        self.extractor.down.to(device)
        self.extractor.u.to(device)
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_filters[-1], self.conv_filters[-1] // 2),
            nn.ReLU(),
            norm_f(self.conv_filters[-1] // 2),
            # nn.Linear(self.conv_filters[-1] // 2, self.conv_filters[-1] // 4),
            # nn.ReLU(),
            # norm_f(self.conv_filters[-1] // 4),
            # nn.Linear(self.conv_filters[-1] // 4, 1)
            nn.Linear(self.conv_filters[-1] // 2, 1)
        )
        self.classifier.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                # 'f': lambda p, t: focal_loss(
                    p, t.type_as(p).to(p.device)
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 0,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device)
                )
            },
            {
                'name': 'fn',
                'weight': 0.5,
                'f': lambda p, t: tp_binary_loss(p, t)
            },
            {
                'name': 'fp',
                'weight': 0.5,
                'f': lambda p, t: tn_binary_loss(p, t)
            },
            {
                'name': 'acc',
                'weight': 0,
                'f': lambda p, t: 1 - accuracy(
                    (p > 0.5).type_as(p), t.type_as(p)
                )
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=1e-4)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self):
        super().reset_optimiser()
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=1e-4)

    def forward(self, data):
        _, features = self.extractor.encode(data)
        # final_features = torch.mean(features.flatten(2), dim=2)
        final_features = torch.max(features.flatten(2), dim=2)[0]
        logits = self.classifier(final_features)
        return torch.sigmoid(logits)

    def inference(self, data, nonbatched=False):
        return super().inference(data, nonbatched=nonbatched)


class AttentionUNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.segmenter = nn.Sequential(
            AttentionAutoencoder(
                self.conv_filters, device, n_images, block=ResConv3dBlock,
                norm=norm_f
            ),
            ResConv3dBlock(
                self.conv_filters[0], self.conv_filters[0], 1,
                norm=norm_f
            ),
            nn.Conv3d(self.conv_filters[0], 1, 1)
        )
        self.segmenter.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'grad',
                'weight': 1,
                'f': lambda p, t: grad_loss(
                    p, t.type_as(p).to(p.device),
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            },
            {
                'name': 'pdsc',
                'weight': 0,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_binary_loss(p, t)
            },
            {
                'name': 'fn',
                'weight': 0,
                'f': lambda p, t: tp_binary_loss(p, t)
            },
            {
                'name': 'fp',
                'weight': 0,
                'f': lambda p, t: tn_binary_loss(p, t)
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self):
        super().reset_optimiser()
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)

    def forward(self, data):

        return torch.sigmoid(self.segmenter(data))


class DualHeadedUNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [16, 32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        self.ae = DualAttentionAutoencoder(
            self.conv_filters, device, n_images, block=ResConv3dBlock,
            norm=norm_f
        )
        self.ae.to(device)
        self.segmenter = nn.Conv3d(self.conv_filters[0], 1, 1)
        self.segmenter.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'pdsc',
                'weight': 1,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'xentropy',
                'weight': 0,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            },
            {
                'name': 'pdsc',
                'weight': 0,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_binary_loss(p, t)
            },
            {
                'name': 'fn',
                'weight': 0,
                'f': lambda p, t: tp_binary_loss(p, t)
            },
            {
                'name': 'fp',
                'weight': 0,
                'f': lambda p, t: tn_binary_loss(p, t)
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self):
        super().reset_optimiser()
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)

    def forward(self, source, target):
        features = self.ae(source, target)
        segmentation = torch.sigmoid(self.segmenter(features))
        return segmentation


class LongitudinalEncoder(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=1,
            dropout=0,
            verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [16, 32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.ae = Autoencoder(
            self.conv_filters, device, n_images, block=ResConv3dBlock,
            norm=norm_f
        )
        self.ae.to(device)
        self.final_source = ResConv3dBlock(
            self.conv_filters[0], 1, 1, nn.Identity, nn.Identity
        )
        self.final_source.to(device)
        self.final_target = ResConv3dBlock(
            self.conv_filters[0], 1, 1, nn.Identity, nn.Identity
        )
        self.final_target.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'bl',
                'weight': 1,
                'f': lambda p, t: F.mse_loss(
                    p[0], t[0],
                )
            },
            {
                'name': 'fu',
                'weight': 1,
                'f': lambda p, t: F.mse_loss(
                    p[1], t[1],
                )
            },
            {
                'name': 'sim',
                'weight': 1,
                'f': lambda p, t: similarity_loss(p[2])
            },
        ]
        self.val_functions = self.train_functions

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, source, target):
        source_out, source_feat = self.ae(source, keepfeat=True)
        target_out, target_feat = self.ae(target, keepfeat=True)

        source_out = self.final_source(source_out)
        target_out = self.final_source(target_out)

        feat = list(zip(source_feat, target_feat))

        return source_out, target_out, feat
