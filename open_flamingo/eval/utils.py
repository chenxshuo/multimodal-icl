import getpass

import numpy as np
import torch
import random
import os
import torch.nn as nn
import time
from contextlib import suppress
import logging

logger = logging.getLogger(__name__)


def create_experiment_dir(args, model_args):
    """
    Create a directory for the experiment.

    model: OF3B, OF4B, OF4BI, OF9B
    dataset: vqav2, gqa, okvqa, textvqa, vizwiz, coco, flickr30, hatefulmemes
    demo_mode
    visual_demo_mode
    shot

    BASE_PATH/model/demo_mode_{demo_mode}/visual_demo_mode_{visual_demo_mode}/exp_time/evaluation_results.json
    BASE_PATH/model/demo_mode_{demo_mode}/visual_demo_mode_{visual_demo_mode}/exp_time/prediction_results.json
    """
    if getpass.getuser() == "di93zun":
        BASE_PATH = "/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/robustness/in-context-open-flamingo/open_flamingo_2-0/.experimental_results"
    elif getpass.getuser() == "b207dd13":
        BASE_PATH = "/home/atuin/b207dd/b207dd13/in-context/open_flamingo/.experimental_results"
    else:
        raise NotImplementedError("Unknown user. Please set BASE_PATH manually.")

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    if "9B" in model_args["checkpoint_path"]:
        model = "OF9B"
    elif "3B" in model_args["checkpoint_path"]:
        if "instruct" in model_args["checkpoint_path"]:
            model = "OF3BI"
        else:
            model = "OF3B"
    elif "4B" in model_args["checkpoint_path"]:
        if "instruct" in model_args["checkpoint_path"]:
            model = "OF4BI"
        else:
            model = "OF4B"
    else:
        raise NotImplementedError("Only OF9B, OF4B, OF4BI, OF3B, OF3BI are supported for now.")
    demo_mode = args.demo_mode
    visual_demo_mode = args.visual_demo_mode
    shot = args.shots
    shot = "_".join([str(s) for s in shot])
    evaluate_tasks = []
    if args.eval_vqav2:
        evaluate_tasks.append("vqav2")
    if args.eval_gqa:
        evaluate_tasks.append("gqa")
    if args.eval_ok_vqa:
        evaluate_tasks.append("ok_vqa")
    if args.eval_textvqa:
        evaluate_tasks.append("textvqa")
    if args.eval_vizwiz:
        evaluate_tasks.append("vizwiz")
    if args.eval_coco:
        evaluate_tasks.append("coco")
    if args.eval_flickr30:
        evaluate_tasks.append("flickr30")
    if args.eval_hateful_memes:
        evaluate_tasks.append("hateful_memes")
    if args.eval_imagenet:
        evaluate_tasks.append("imagenet")

    evaluate_tasks = "_".join(evaluate_tasks)
    # time in  format 2021-06-30_15-00-00
    experiment_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if args.rices:
        experiment_base_dir = os.path.join(
            BASE_PATH,
            f"{model}",
            f"rices",
            f"demo_mode_{demo_mode}",
            f"visual_demo_mode_{visual_demo_mode}",
            f"{evaluate_tasks}",
            f"shot_{shot}",
            f"{experiment_time}",
        )
        if args.rices_every_nth:
            experiment_base_dir = os.path.join(
                BASE_PATH,
                f"{model}",
                f"rices_every_nth",
                f"demo_mode_{demo_mode}",
                f"visual_demo_mode_{visual_demo_mode}",
                f"{evaluate_tasks}",
                f"shot_{shot}",
                f"{experiment_time}",
            )
        if args.rices_find_by_ranking_similar_text:
            experiment_base_dir = os.path.join(
                BASE_PATH,
                f"{model}",
                f"rices_find_by_ranking_similar_text",
                f"demo_mode_{demo_mode}",
                f"visual_demo_mode_{visual_demo_mode}",
                f"{evaluate_tasks}",
                f"shot_{shot}",
                f"{experiment_time}",
            )
            if args.rices_similar_with_labels:
                experiment_base_dir = os.path.join(
                    BASE_PATH,
                    f"{model}",
                    f"rices_find_by_ranking_similar_text",
                    f"rices_similar_with_labels",
                    f"demo_mode_{demo_mode}",
                    f"visual_demo_mode_{visual_demo_mode}",
                    f"{evaluate_tasks}",
                    f"shot_{shot}",
                    f"{experiment_time}",
                )
    elif args.rices_text:
        if args.rices_text_then_visual:
            experiment_base_dir = os.path.join(
                BASE_PATH,
                f"{model}",
                f"rices_text_then_visual",
                f"demo_mode_{demo_mode}",
                f"visual_demo_mode_{visual_demo_mode}",
                f"{evaluate_tasks}",
                f"shot_{shot}",
                f"{experiment_time}",
            )
        else:
            experiment_base_dir = os.path.join(
                BASE_PATH,
                f"{model}",
                f"rices_text",
                f"demo_mode_{demo_mode}",
                f"visual_demo_mode_{visual_demo_mode}",
                f"{evaluate_tasks}",
                f"shot_{shot}",
                f"{experiment_time}",
            )
    else:
        experiment_base_dir = os.path.join(
            BASE_PATH,
            f"{model}",
            f"demo_mode_{demo_mode}",
            f"visual_demo_mode_{visual_demo_mode}",
            f"{evaluate_tasks}",
            f"shot_{shot}",
            f"{experiment_time}",
        )
    if not os.path.exists(experiment_base_dir):
        try:
            os.makedirs(experiment_base_dir)
        except FileExistsError:
            pass
    if args.rank == 0:
        logger.info(f"======= Created experiment directory: {experiment_base_dir} =======")
        logger.info(f"========Arguments used for this experiment========")
        # print namespace object line by line
        for arg in vars(args):
            logger.info(f"{arg}: {getattr(args, arg)}")
        for arg in model_args:
            logger.info(f"{arg}: {model_args[arg]}")
    return experiment_base_dir


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def compute_effective_num_shots(num_shots, model_type):
    """
    Compute the effective number of shots for a given model type.
    For example, following Flamingo, 0-shot OF evaluations use two text-only shots.
    """
    if model_type == "open_flamingo":
        return num_shots if num_shots > 0 else 2
    return num_shots


def sample_batch_demos_from_query_set(query_set, num_samples, batch_size):
    """
    Sample random demonstrations from the query set.
    """
    return [random.sample(query_set, num_samples) for _ in range(batch_size)]


def get_query_set(train_dataset, query_set_size):
    """
    Get a subset of the training dataset to use as the query set.
    """
    if query_set_size > len(train_dataset):
        logger.warning(
            f"Number of query requested {query_set_size} is greater than the size of the dataset {len(train_dataset)}."
            f" Setting query_set_size to {len(train_dataset)}")
        query_set_size = len(train_dataset)
    query_set = np.random.choice(len(train_dataset), query_set_size, replace=False)
    return [train_dataset[i] for i in query_set]


def prepare_eval_samples(test_dataset, num_samples, batch_size):
    """
    Subset the test dataset and return a DataLoader.
    """
    # if num_samples == len(test_dataset):
    #     loader = torch.utils.data.DataLoader(
    #         test_dataset,
    #         batch_size=batch_size,
    #         collate_fn=custom_collate_fn,
    #     )
    #     return loader
    random_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    dataset = torch.utils.data.Subset(test_dataset, random_indices)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
    )
    return loader


def get_indices_of_unique(x):
    """
    Return the indices of x that correspond to unique elements.
    If value v is unique and two indices in x have value v, the first index is returned.
    """
    unique_elements = torch.unique(x)
    first_indices = []
    for v in unique_elements:
        indices = torch.where(x == v)[0]
        first_indices.append(indices[0])  # Take the first index for each unique element
    return torch.tensor(first_indices)


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model


def get_predicted_classnames(logprobs, k, class_id_to_name):
    """
    Args:
        - logprobs shape (B, Y) containing logprobs for each classname
        - k: number for top-k
        - class_id_to_name: dict mapping class index to classname

    Returns:
        - top-k predicted classnames shape (B, k) type str
        - top-k logprobs shape (B, k) type float
    """
    # convert indices to classnames
    _, predictions = torch.topk(logprobs, k=k, dim=1)  # shape (B, k)
    predicted_classnames = [
        [class_id_to_name[ix] for ix in item] for item in predictions.tolist()
    ]
    predicted_logprobs = torch.gather(logprobs, 1, predictions)
    return predicted_classnames, predicted_logprobs


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
