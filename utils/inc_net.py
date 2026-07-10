import copy
import logging

import torch
from torch import nn

from backbone.linears import (
    SimpleLinear,
    SplitCosineLinear,
    CosineLinear,
    EaseCosineLinear,
    SimpleContinualLinear,
    TunaLinear,
)
from backbone.prompt import CodaPrompt


def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()

    if name == "uni3d_base_patch14_448" or name == "uni3d_base_patch14_448_base":
        from backbone import uni3d_base
        model = uni3d_base.create_uni3d_base(
            pc_model=args.get("pc_model", "eva02_base_patch14_448"),
            pretrained_pc=args.get("pretrained_pc", None),
            drop_path_rate=args.get("drop_path", 0.0),
            trans_dim=args.get("trans_dim", 768),
            embed_dim=args.get("embed_dim", 1024),
            num_group=args.get("num_group", 512),
            group_size=args.get("group_size", 64),
            encoder_dim=args.get("encoder_dim", 512),
            patch_dropout=args.get("patch_dropout", 0.0),
        )
        return model.eval()

    if "_adapter" in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] in (
            "aper_adapter",
            "ranpac",
            "selfevo",
            "selfevo_bufferfree",
        ):
            from backbone import uni3d_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_num=ffn_num,
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_adapter_layernorm_option="none",
            )
            model = uni3d_adapter.create_uni3d_adapter(
                pc_model=args.get("pc_model", "eva02_base_patch14_448"),
                pretrained_pc=args.get("pretrained_pc", None),
                drop_path_rate=args.get("drop_path", 0.0),
                trans_dim=args.get("trans_dim", 768),
                embed_dim=args.get("embed_dim", 1024),
                num_group=args.get("num_group", 512),
                group_size=args.get("group_size", 64),
                encoder_dim=args.get("encoder_dim", 512),
                patch_dropout=args.get("patch_dropout", 0.0),
                tuning_config=tuning_config,
            )
            model.out_dim = args.get("embed_dim", 1024)
            return model.eval()
        raise NotImplementedError("Inconsistent model name and model type")

    if "_l2p" in name:
        if args["model_name"] != "l2p":
            raise NotImplementedError("Inconsistent model name and model type")
        from backbone import uni3d_l2p
        model = uni3d_l2p.create_uni3d_l2p(
            pc_model=args.get("pc_model", "eva02_base_patch14_448"),
            pretrained_pc=args.get("pretrained_pc", None),
            drop_path_rate=args.get("drop_path", 0.0),
            trans_dim=args.get("trans_dim", 768),
            embed_dim=args.get("embed_dim", 1024),
            num_group=args.get("num_group", 512),
            group_size=args.get("group_size", 64),
            encoder_dim=args.get("encoder_dim", 256),
            patch_dropout=args.get("patch_dropout", 0.0),
            num_classes=args["nb_classes"],
            prompt_length=args["length"],
            embedding_key=args["embedding_key"],
            prompt_init=args["prompt_key_init"],
            prompt_pool=args["prompt_pool"],
            prompt_key=args["prompt_key"],
            pool_size=args["size"],
            top_k=args["top_k"],
            batchwise_prompt=args["batchwise_prompt"],
            prompt_key_init=args["prompt_key_init"],
            use_prompt_mask=args.get("use_prompt_mask", False),
        )
        return model

    if "_dualprompt" in name:
        if args["model_name"] != "dualprompt":
            raise NotImplementedError("Inconsistent model name and model type")
        from backbone import uni3d_dualprompt
        model = uni3d_dualprompt.create_uni3d_dualprompt(
            pc_model=args.get("pc_model", "eva02_base_patch14_448"),
            pretrained_pc=args.get("pretrained_pc", None),
            drop_path_rate=args.get("drop_path", 0.0),
            trans_dim=args.get("trans_dim", 768),
            embed_dim=args.get("embed_dim", 1024),
            num_group=args.get("num_group", 512),
            group_size=args.get("group_size", 64),
            encoder_dim=args.get("encoder_dim", 512),
            patch_dropout=args.get("patch_dropout", 0.0),
            num_classes=args["nb_classes"],
            prompt_length=args["length"],
            embedding_key=args["embedding_key"],
            prompt_init=args["prompt_key_init"],
            prompt_pool=args["prompt_pool"],
            prompt_key=args["prompt_key"],
            pool_size=args["size"],
            top_k=args["top_k"],
            batchwise_prompt=args["batchwise_prompt"],
            prompt_key_init=args["prompt_key_init"],
            head_type=args["head_type"],
            use_prompt_mask=args["use_prompt_mask"],
            use_g_prompt=args["use_g_prompt"],
            g_prompt_length=args["g_prompt_length"],
            g_prompt_layer_idx=args["g_prompt_layer_idx"],
            use_prefix_tune_for_g_prompt=args.get("use_prefix_tune_for_g_prompt", False),
            use_e_prompt=args["use_e_prompt"],
            e_prompt_layer_idx=args["e_prompt_layer_idx"],
            use_prefix_tune_for_e_prompt=args.get("use_prefix_tune_for_e_prompt", False),
            same_key_value=args.get("same_key_value", False),
        )
        return model

    if "_coda_prompt" in name:
        if args["model_name"] != "coda_prompt":
            raise NotImplementedError("Inconsistent model name and model type")
        from backbone import uni3d_coda_prompt
        model = uni3d_coda_prompt.create_uni3d_coda_prompt(
            pc_model=args.get("pc_model", "eva02_base_patch14_448"),
            pretrained_pc=args.get("pretrained_pc", None),
            drop_path_rate=args.get("drop_path", 0.0),
            trans_dim=args.get("trans_dim", 768),
            num_group=args.get("num_group", 512),
            group_size=args.get("group_size", 64),
            encoder_dim=args.get("encoder_dim", 512),
            patch_dropout=args.get("patch_dropout", 0.0),
        )
        return model

    if "_ease" in name:
        if args["model_name"] != "ease":
            raise NotImplementedError("Inconsistent model name and model type")
        from backbone import uni3d_ease
        from easydict import EasyDict
        ffn_num = args["ffn_num"]
        tuning_config = EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            vpt_on=False,
            vpt_num=0,
            _device=args["device"][0],
        )
        model = uni3d_ease.create_uni3d_ease(
            pc_model=args.get("pc_model", "eva02_base_patch14_448"),
            pretrained_pc=args.get("pretrained_pc", None),
            drop_path_rate=args.get("drop_path", 0.0),
            trans_dim=args.get("trans_dim", 768),
            num_group=args.get("num_group", 512),
            group_size=args.get("group_size", 64),
            encoder_dim=args.get("encoder_dim", 512),
            patch_dropout=args.get("patch_dropout", 0.0),
            tuning_config=tuning_config,
        )
        model.out_dim = args.get("trans_dim", 768)
        return model.eval()

    if "_mos" in name:
        if args["model_name"] != "mos":
            raise NotImplementedError("Inconsistent model name and model type")
        from backbone import uni3d_mos
        from easydict import EasyDict
        ffn_num = args["ffn_num"]
        tuning_config = EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            _device=args["device"][0],
            adapter_momentum=args["adapter_momentum"],
            vpt_on=False,
            vpt_num=0,
        )
        model = uni3d_mos.create_uni3d_mos(
            pc_model=args.get("pc_model", "eva02_base_patch14_448"),
            pretrained_pc=args.get("pretrained_pc", None),
            drop_path_rate=args.get("drop_path", 0.0),
            trans_dim=args.get("trans_dim", 768),
            num_group=args.get("num_group", 512),
            group_size=args.get("group_size", 64),
            encoder_dim=args.get("encoder_dim", 512),
            patch_dropout=args.get("patch_dropout", 0.0),
            num_classes=args["nb_classes"],
            tuning_config=tuning_config,
        )
        model.out_dim = args.get("trans_dim", 768)
        return model

    if "_tuna" in name:
        from backbone import uni3d_tuna
        from easydict import EasyDict
        ffn_num = 16
        tuning_config = EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            _device=args["device"][0],
            vpt_on=False,
            vpt_num=0,
        )
        model = uni3d_tuna.create_uni3d_tuna(
            pc_model=args.get("pc_model", "eva02_base_patch14_448"),
            pretrained_pc=args.get("pretrained_pc", None),
            drop_path_rate=args.get("drop_path", 0.0),
            trans_dim=args.get("trans_dim", 768),
            num_group=args.get("num_group", 512),
            group_size=args.get("group_size", 64),
            encoder_dim=args.get("encoder_dim", 512),
            patch_dropout=args.get("patch_dropout", 0.0),
            tuning_config=tuning_config,
        )
        model.out_dim = args.get("trans_dim", 768)
        return model

    raise NotImplementedError("Unknown backbone_type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.backbone = get_backbone(args, pretrained)
        self.fc = None
        self._device = args["device"][0]
        self.model_type = "vit"

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class IncrementalNet(BaseNet):
    """Plain linear head."""

    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        gamma = torch.mean(oldnorm) / torch.mean(newnorm)
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out


class SimpleCosineIncrementalNet(BaseNet):
    """Cosine-head BaseNet."""

    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat(
                    [weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)]
                )
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        return CosineLinear(in_dim, out_dim)


class SimpleVitNet(BaseNet):
    """Cosine-head wrapper with optional random projection."""

    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.W_rand = None
        self.RP_dim = None

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat(
                    [weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)]
                )
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        return CosineLinear(in_dim, out_dim)

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        if self.W_rand is not None:
            x = torch.nn.functional.relu(x @ self.W_rand)
        out = self.fc(x)
        out.update({"features": x})
        return out


class PromptVitNet(nn.Module):
    """Wrapper for L2P / DualPrompt.

    When ``args['get_original_backbone']`` is True, build a separate non-prompt encoder
    used to extract CLS features for prompt selection.
    """

    def __init__(self, args, pretrained):
        super(PromptVitNet, self).__init__()
        self.backbone = get_backbone(args, pretrained)

        if args.get("get_original_backbone", False):
            from backbone import uni3d_l2p
            self.original_backbone = uni3d_l2p.create_uni3d_original(
                pc_model=args.get("pc_model", "eva02_base_patch14_448"),
                pretrained_pc=args.get("pretrained_pc", None),
                drop_path_rate=args.get("drop_path", 0.0),
                trans_dim=args.get("trans_dim", 768),
                embed_dim=args.get("embed_dim", 1024),
                num_group=args.get("num_group", 512),
                group_size=args.get("group_size", 64),
                encoder_dim=args.get("encoder_dim", 256),
                patch_dropout=args.get("patch_dropout", 0.0),
                num_classes=0,
            ).eval()
        else:
            self.original_backbone = None

    def forward(self, x, task_id=-1, train=False):
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features = self.original_backbone.extract_cls_transdim(x)
            else:
                cls_features = None
        return self.backbone(x, task_id=task_id, cls_features=cls_features, train=train)


class CodaPromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(CodaPromptVitNet, self).__init__()
        self.args = args
        self.backbone = get_backbone(args, pretrained)
        self.fc = nn.Linear(768, args["nb_classes"])
        self.prompt = CodaPrompt(768, args["nb_tasks"], args["prompt_param"])

    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.backbone(x)
                q = q[:, 0, :]
            out, prompt_loss = self.backbone(x, prompt=self.prompt, q=q, train=train)
            out = out[:, 0, :]
        else:
            out, _ = self.backbone(x)
            out = out[:, 0, :]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.fc(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        return out


class MultiBranchCosineIncrementalNet(BaseNet):
    """Frozen base + tuned-adapter dual branch."""

    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        self.backbone = torch.nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbones = nn.ModuleList()
        self.args = args
        self.model_type = "vit"

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat(
                    [weight, torch.zeros(nb_classes - nb_output, self._feature_dim).to(self._device)]
                )
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        return CosineLinear(in_dim, out_dim)

    def forward(self, x):
        features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        out = self.fc(features)
        out.update({"features": features})
        return out

    def construct_dual_branch_network(self, tuned_model):
        if "adapter" in self.args["backbone_type"]:
            newargs = copy.deepcopy(self.args)
            newargs["backbone_type"] = newargs["backbone_type"].replace("_adapter", "")
            self.backbones.append(get_backbone(newargs))
        else:
            self.backbones.append(get_backbone(self.args))

        self.backbones.append(tuned_model.backbone)

        self._feature_dim = self.backbones[0].out_dim * len(self.backbones)
        self.fc = self.generate_fc(self._feature_dim, self.args["init_cls"])


class EaseNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim = self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        return self.out_dim * (self._cur_task + 1)

    def update_fc(self, nb_classes):
        self._cur_task += 1
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)

        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()

        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[:old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        return EaseCosineLinear(in_dim, out_dim)

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        if test is False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
        else:
            x = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x)
            else:
                out = self.fc.forward_reweight(
                    x,
                    cur_task=self._cur_task,
                    alpha=self.alpha,
                    init_cls=self.init_cls,
                    inc=self.inc,
                    use_init_ptm=self.use_init_ptm,
                    beta=self.beta,
                )

        out.update({"features": x})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())


class SLCANet(BaseNet):
    def __init__(self, args, pretrained=True, fc_with_ln=False):
        super().__init__(args, pretrained)
        self.old_fc = None
        self.fc_with_ln = fc_with_ln

    def extract_layerwise_vector(self, x, pool=True):
        with torch.no_grad():
            features = self.backbone(x, layer_feat=True)["features"]
        for f_i in range(len(features)):
            if pool:
                features[f_i] = features[f_i].mean(1).cpu().numpy()
            else:
                features[f_i] = features[f_i][:, 0].cpu().numpy()
        return features

    def update_fc(self, nb_classes, freeze_old=True):
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old)

    def save_old_fc(self):
        if self.old_fc is None:
            self.old_fc = copy.deepcopy(self.fc)
        else:
            self.old_fc.heads.append(copy.deepcopy(self.fc.heads[-1]))

    def generate_fc(self, in_dim, out_dim):
        return SimpleContinualLinear(in_dim, out_dim)

    def forward(self, x, bcb_no_grad=False, fc_only=False):
        if fc_only:
            fc_out = self.fc(x)
            if self.old_fc is not None:
                old_fc_logits = self.old_fc(x)["logits"]
                fc_out["old_logits"] = old_fc_logits
            return fc_out
        if bcb_no_grad:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out


class MOSNet(nn.Module):
    def __init__(self, args, pretrained):
        super(MOSNet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = 768
        self.fc = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat(
                    [weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)]
                )
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc.requires_grad_(False)

    def generate_fc(self, in_dim, out_dim):
        return CosineLinear(in_dim, out_dim)

    def forward_orig(self, x):
        features = self.backbone(x, adapter_id=0)["features"]
        return {"features": features, "logits": self.fc(features)["logits"]}

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        return self.backbone(x, adapter_id, train, fc_only)


class TUNANet(nn.Module):
    def __init__(self, args, pretrained):
        super(TUNANet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = 768
        self.fc = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=False)

    def generate_fc(self, in_dim, out_dim):
        return TunaLinear(in_dim, out_dim)

    def forward_orig(self, x):
        features = self.backbone(x, adapter_id=0)["features"]
        return {"features": features, "logits": self.fc(features)["logits"]}

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        return self.backbone(x, adapter_id, train, fc_only)
