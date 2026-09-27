import importlib.util
from pathlib import Path

POINTCLOUD_DATASETS = {
    "modelnet40",
    "shapenet53",
    "shapenet55",
    "co3d",
    "scanobjectnn",
    "modelnet_scanobjectnn",
    "shapenet_scanobjectnn",
    "shapenet_co3d",
}

# LGSP image FSCIL protocols (CUB / Aircraft / iNF200). Reuse repo-root SelfEvo.
IMAGE_FSCIL_DATASETS = {
    "cub200",
    "fgvcaircraft",
    "inf200",
}

# Repo-root models/ holds SelfEvo / FSCIL learners (not a lamba_pilot subpackage).
_3D_METHODS = {
    "simplecil",
    "mos",
    "tuna",
    "ease",
    "l2p",
    "dualprompt",
    "coda_prompt",
    "ranpac",
    "slca",
    "aper_adapter",
    "selfevo",
    "selfevo_bufferfree",
    "fscil_joint",
    "fscil_finetune",
    "fscil_linprobe",
}

# Methods that live under repo-root models/ even for image FSCIL.
_REPO_ROOT_IMAGE_METHODS = {
    "selfevo",
    "selfevo_bufferfree",
    "fscil_joint",
    "fscil_finetune",
    "fscil_linprobe",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_3d_learner(method_name):
    path = REPO_ROOT / "models" / f"{method_name}.py"
    if not path.exists():
        raise NotImplementedError(f"No 3D learner at {path}")
    spec = importlib.util.spec_from_file_location(f"pilot_3d_{method_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Learner


def get_model(model_name, args):
    name = model_name.lower()
    dataset = args.get("dataset", "").lower()
    use_3d = dataset in POINTCLOUD_DATASETS
    use_image_fscil = dataset in IMAGE_FSCIL_DATASETS

    if use_3d:
        if name not in _3D_METHODS:
            raise NotImplementedError(
                f"Method '{name}' is not available for 3D dataset '{dataset}'. "
                f"Supported: {sorted(_3D_METHODS)}"
            )
        return _load_3d_learner(name)(args)

    if use_image_fscil and name in _REPO_ROOT_IMAGE_METHODS:
        return _load_3d_learner(name)(args)

    if name == "simplecil":
        from models.simplecil import Learner
    elif name == "aper_finetune":
        from models.aper_finetune import Learner
    elif name == "aper_ssf":
        from models.aper_ssf import Learner
    elif name == "aper_vpt":
        from models.aper_vpt import Learner
    elif name == "aper_adapter":
        from models.aper_adapter import Learner
    elif name == "l2p":
        from models.l2p import Learner
    elif name == "dualprompt":
        from models.dualprompt import Learner
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
    elif name == "finetune":
        from models.finetune import Learner
    elif name == "icarl":
        from models.icarl import Learner
    elif name == "der":
        from models.der import Learner
    elif name == "coil":
        from models.coil import Learner
    elif name == "foster":
        from models.foster import Learner
    elif name == "memo":
        from models.memo import Learner
    elif name == "ranpac":
        from models.ranpac import Learner
    elif name == "ease":
        from models.ease import Learner
    elif name == "slca":
        from models.slca import Learner
    elif name == "lae":
        from models.lae import Learner
    elif name == "fecam":
        from models.fecam import Learner
    elif name == "dgr":
        from models.dgr import Learner
    elif name == "mos":
        from models.mos import Learner
    elif name == "cofima":
        from models.cofima import Learner
    elif name == "duct":
        from models.duct import Learner
    elif name == "tuna":
        from models.tuna import Learner
    else:
        assert 0
    return Learner(args)
