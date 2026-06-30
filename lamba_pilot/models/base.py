import copy
import logging
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, safe_topk_indices, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 64

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.args = args

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim
    
    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def tsne(self,showcenters=False,Normalize=False):
        import umap
        import matplotlib.pyplot as plt
        print('now draw tsne results of extracted features.')
        tot_classes=self._total_classes
        test_dataset = self.data_manager.get_dataset(np.arange(0, tot_classes), source='test', mode='test')
        valloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        vectors, y_true = self._extract_vectors(valloader)
        if showcenters:
            fc_weight=self._network.fc.proj.cpu().detach().numpy()[:tot_classes]
            print(fc_weight.shape)
            vectors=np.vstack([vectors,fc_weight])
        
        if Normalize:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(vectors)
        
        if showcenters:
            clssscenters=embedding[-tot_classes:,:]
            centerlabels=np.arange(tot_classes)
            embedding=embedding[:-tot_classes,:]
        scatter=plt.scatter(embedding[:,0],embedding[:,1],c=y_true,s=20,cmap=plt.cm.get_cmap("tab20"))
        plt.legend(*scatter.legend_elements())
        if showcenters:
            plt.scatter(clssscenters[:,0],clssscenters[:,1],marker='*',s=50,c=centerlabels,cmap=plt.cm.get_cmap("tab20"),edgecolors='black')
        
        plt.savefig(str(self.args['model_name'])+str(tot_classes)+'tsne.pdf')
        plt.close()


    def _unwrap_network(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module
        return self._network

    @staticmethod
    def _to_cpu_nested(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, dict):
            return {k: BaseLearner._to_cpu_nested(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(BaseLearner._to_cpu_nested(v) for v in obj)
        return obj

    def _checkpoint_extra(self):
        """Optional learner-specific state (override in subclasses)."""
        return {}

    def _load_checkpoint_extra(self, extra):
        """Restore learner-specific state saved by ``_checkpoint_extra``."""
        pass

    def build_checkpoint(self, trainer_state=None):
        network = self._unwrap_network()
        state = {
            "version": 1,
            "cur_task": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "model_state_dict": copy.deepcopy(network.state_dict()),
            "data_memory": self._data_memory,
            "targets_memory": self._targets_memory,
            "learner_extra": self._to_cpu_nested(self._checkpoint_extra()),
        }
        if hasattr(self, "_class_means") and getattr(self, "_class_means", None) is not None:
            state["class_means"] = self._class_means
        if trainer_state is not None:
            state["trainer_state"] = trainer_state
        return state

    def save_task_checkpoint(self, checkpoint_dir, trainer_state=None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(
            checkpoint_dir, "task_{:03d}.pt".format(self._cur_task)
        )
        torch.save(self.build_checkpoint(trainer_state), path)
        logging.info("Saved checkpoint: %s", path)
        return path

    def _prepare_network_from_checkpoint(self, ckpt):
        """Resize heads so ``load_state_dict`` can restore a mid-run model."""
        network = self._unwrap_network()
        if not hasattr(network, "update_fc"):
            return
        state = ckpt["model_state_dict"]
        fc_weight_key = next((k for k in state if k.endswith("fc.weight")), None)
        if fc_weight_key is None:
            return
        saved_out = int(state[fc_weight_key].shape[0])
        current_out = None
        if network.fc is not None:
            current_out = getattr(network.fc, "out_features", None)
            if current_out is None and hasattr(network.fc, "weight"):
                current_out = int(network.fc.weight.shape[0])
        if current_out == saved_out:
            return

        model_name = self.args.get("model_name", "")
        cur_task = int(ckpt["cur_task"])
        if model_name in ("tuna", "slca"):
            init_cls = self.args["init_cls"]
            inc = self.args["increment"]
            for task_id in range(cur_task + 1):
                network.update_fc(init_cls if task_id == 0 else inc)
        else:
            network.update_fc(int(ckpt["total_classes"]))

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._cur_task = int(ckpt["cur_task"])
        self._known_classes = int(ckpt["known_classes"])
        self._total_classes = int(ckpt["total_classes"])
        self._data_memory = ckpt.get("data_memory", np.array([]))
        self._targets_memory = ckpt.get("targets_memory", np.array([]))
        if "class_means" in ckpt:
            self._class_means = ckpt["class_means"]
        network = self._unwrap_network()
        self._prepare_network_from_checkpoint(ckpt)
        network.load_state_dict(ckpt["model_state_dict"], strict=True)
        self._load_checkpoint_extra(ckpt.get("learner_extra", {}))
        network.to(self._device)
        logging.info(
            "Loaded checkpoint %s (completed task %d, classes %d)",
            path,
            self._cur_task,
            self._total_classes,
        )
        return ckpt

    def save_checkpoint(self, filename):
        """Legacy per-task pickle save (FeCAM and other 2D baselines)."""
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.args["init_cls"], self.args["increment"])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        k = y_pred.shape[1]
        ret["top{}".format(k)] = np.around(
            (y_pred.T == np.tile(y_true, (k, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        eval_out = self._eval_cnn(self.test_loader)
        if isinstance(eval_out, tuple) and len(eval_out) == 3:
            y_pred, y_true, _y_types = eval_out
        else:
            y_pred, y_true = eval_out
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass
    
    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = safe_topk_indices(outputs, self.topk)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    
    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

        def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
            if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
                ori_classes = self._class_means.shape[0]
                assert ori_classes == self._known_classes
                new_class_means = np.zeros((self._total_classes, self.feature_dim))
                new_class_means[:self._known_classes] = self._class_means
                self._class_means = new_class_means
                # new_class_cov = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
                new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
                new_class_cov[:self._known_classes] = self._class_covs
                self._class_covs = new_class_cov
            elif not check_diff:
                self._class_means = np.zeros((self._total_classes, self.feature_dim))
                # self._class_covs = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
                self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

            for class_idx in range(self._known_classes, self._total_classes):

                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)

                try:
                    assert vectors.shape[0] > 1
                except AssertionError as e:
                    print("Size of the {}-th class is: {}, repeat it for twice.".format(class_idx, vectors.shape[0]))
                    vectors = np.tile(vectors, (2, 1))
                    print("Shape of vectors after repeating: {}".format(vectors.shape))

                # vectors = np.concatenate([vectors_aug, vectors])

                class_mean = np.mean(vectors, axis=0)
                # class_cov = np.cov(vectors.T)
                # try:
                #     class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-4
                # except UserWarning as e:
                #     logging.warning("Caught UserWarning: ", e)
               
                self._class_means[class_idx, :] = class_mean
                self._class_covs[class_idx, ...] = class_cov
