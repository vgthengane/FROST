import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def _append_topk_curve(curve, accy):
    """Record top-1 and the highest top-k metric present in *accy*."""
    curve.setdefault("top1", []).append(accy["top1"])
    topk_keys = sorted(k for k in accy if k.startswith("top") and k != "top1")
    if topk_keys:
        key = topk_keys[-1]
        curve.setdefault(key, []).append(accy[key])


def _find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    paths = [
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if name.startswith("task_") and name.endswith(".pt")
    ]
    if not paths:
        return None
    return max(paths)


def _snapshot_trainer_state(cnn_curve, nme_curve, cnn_matrix, nme_matrix):
    return {
        "cnn_curve": copy.deepcopy(cnn_curve),
        "nme_curve": copy.deepcopy(nme_curve),
        "cnn_matrix": copy.deepcopy(cnn_matrix),
        "nme_matrix": copy.deepcopy(nme_matrix),
    }


def _restore_trainer_state(state, cnn_curve, nme_curve, cnn_matrix, nme_matrix):
    cnn_curve.clear()
    for key, values in state.get("cnn_curve", {"top1": []}).items():
        cnn_curve[key] = list(values)
    nme_curve.clear()
    for key, values in state.get("nme_curve", {"top1": []}).items():
        nme_curve[key] = list(values)
    cnn_matrix[:] = copy.deepcopy(state.get("cnn_matrix", []))
    nme_matrix[:] = copy.deepcopy(state.get("nme_matrix", []))


def _is_fscil_run(args):
    """FSCIL runs keep task-0 warmup ckpts in the learner; skip tasks 1+ on disk."""
    return bool(args.get("fscil_shot_protocol") or args.get("fscil3d_protocol"))


def _prune_fscil_non_task0_checkpoints(checkpoint_dir):
    """Remove per-task checkpoints beyond task 0 (e.g. after --resume on Leonardo)."""
    if not os.path.isdir(checkpoint_dir):
        return
    for name in os.listdir(checkpoint_dir):
        if not (name.startswith("task_") and name.endswith(".pt")):
            continue
        if name == "task_000.pt":
            continue
        path = os.path.join(checkpoint_dir, name)
        os.remove(path)
        logging.info("Pruned FSCIL checkpoint: %s", path)


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    log_root = args.get("log_dir", "logs")
    if log_root == "logs":
        logs_name = os.path.join(
            log_root, args["model_name"], args["dataset"], str(init_cls), str(args["increment"])
        )
    else:
        # 3D CIL: _experiments/<dataset>/methods/<model_name>/...
        logs_name = os.path.join(
            log_root, args["model_name"], str(init_cls), str(args["increment"])
        )

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = os.path.join(
        logs_name,
        "{}_{}_{}".format(args["prefix"], args["seed"], args["backbone_type"]),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks

    checkpoint_dir = os.path.join(
        logs_name,
        "checkpoints",
        "{}_{}_{}".format(args["prefix"], args["seed"], args["backbone_type"]),
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    args["checkpoint_dir"] = checkpoint_dir
    save_checkpoint = args.get("save_checkpoint", True)

    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": []}, {"top1": []}
    cnn_matrix, nme_matrix = [], []

    start_task = 0
    if args.get("resume"):
        latest_ckpt = _find_latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            ckpt = model.load_checkpoint(latest_ckpt)
            start_task = model._cur_task + 1
            trainer_state = ckpt.get("trainer_state")
            if trainer_state:
                _restore_trainer_state(
                    trainer_state, cnn_curve, nme_curve, cnn_matrix, nme_matrix
                )
            logging.info(
                "Resuming from task %d / %d (checkpoint: %s)",
                start_task,
                data_manager.nb_tasks,
                latest_ckpt,
            )
        else:
            logging.info(
                "resume=True but no checkpoint in %s; starting from task 0",
                checkpoint_dir,
            )

    max_task = args.get("max_task")
    if max_task is not None:
        max_task = int(max_task)

    for task in range(start_task, data_manager.nb_tasks):
        if max_task is not None and task > max_task:
            logging.info("Stopping after task %d (--max-task %d)", task - 1, max_task)
            break
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            _append_topk_curve(cnn_curve, cnn_accy)
            _append_topk_curve(nme_curve, nme_accy)

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            topk_keys = [k for k in cnn_curve if k != "top1"]
            if topk_keys:
                logging.info("CNN {} curve: {}".format(topk_keys[-1], cnn_curve[topk_keys[-1]]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            if topk_keys:
                nme_topk = [k for k in nme_curve if k != "top1"]
                if nme_topk:
                    logging.info("NME {} curve: {}\n".format(nme_topk[-1], nme_curve[nme_topk[-1]]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            _append_topk_curve(cnn_curve, cnn_accy)

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            topk_keys = [k for k in cnn_curve if k != "top1"]
            if topk_keys:
                logging.info("CNN {} curve: {}\n".format(topk_keys[-1], cnn_curve[topk_keys[-1]]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

        if save_checkpoint and not (_is_fscil_run(args) and task > 0):
            trainer_state = _snapshot_trainer_state(
                cnn_curve, nme_curve, cnn_matrix, nme_matrix
            )
            model.save_task_checkpoint(checkpoint_dir, trainer_state)

    if _is_fscil_run(args):
        _prune_fscil_non_task0_checkpoints(checkpoint_dir)

    if 'print_forget' in args.keys() and args['print_forget'] is True:
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (CNN):')
            print(np_acctable)
            logging.info('Forgetting (CNN): {}'.format(forgetting))
        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (NME):')
            print(np_acctable)
        logging.info('Forgetting (NME): {}'.format(forgetting))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))