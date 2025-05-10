import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
    
log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def build_transformation(cfg, split, mask=False):
    print(cfg)
    if cfg is None:
        return None
    t = []
    t.append(transforms.ToTensor())
    t.append(transforms.Resize((cfg.imsize, cfg.imsize)))
    
    if mask:
        return transforms.Compose(t)
    
    if cfg.norm:
        if cfg.norm == "imagenet":
            t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        elif cfg.norm == "half":
            t.append(transforms.Normalize((0.5,), (0.5,)))
        elif cfg.norm == "slake":
            t.append(transforms.Normalize((0.2469,), (0.2292,)))  # computed from 5000 images
        elif cfg.norm == "pmcoa":
            t.append(transforms.Normalize((0.1307,), (0.3081,))) 
        else:
            raise NotImplementedError("Normaliation method not implemented")

    # data augmentation 
    if split == "train":

        if cfg.random_crop and cfg.random_crop is not None:
            t.append(transforms.RandomCrop(cfg.random_crop.crop_size))
            t.append(transforms.CenterCrop(cfg.random_crop.crop_size))
        if cfg.random_horizontal_flip:
            t.append(
                transforms.RandomHorizontalFlip(p=cfg.random_horizontal_flip)
            )
        if cfg.random_affine:
            t.append(
                transforms.RandomAffine(
                    cfg.random_affine.degrees,
                    translate=[*cfg.random_affine.translate],
                    scale=[*cfg.random_affine.scale],
                )
            )
        if cfg.color_jitter:
            t.append(
                transforms.ColorJitter(
                    brightness=[*cfg.color_jitter.bightness],
                    contrast=[*cfg.color_jitter.contrast],
                )
            )
    else:
        if cfg.random_crop and cfg.random_crop is not None:
            t.append(transforms.CenterCrop(cfg.random_crop.crop_size))

    return transforms.Compose(t)




def get_slake_normalization_params():
    max_image_id = 200
    imgs_path = [f"datasets/slake/imgs/xmlab{str(i)}/source.jpg" for i in range(1, max_image_id)]

    mean = 0.0
    std = 0.0
    total_images = 0

    for img_path in imgs_path:
        # Convert the image to a tensor (if not already) and get its mean and std
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)
        mean += img.mean(dim=(1, 2))  # mean per channel
        std += img.std(dim=(1, 2))    # std per channel
        total_images += 1

    mean /= total_images
    std /= total_images
    print(mean[0], std[0])
    

def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
