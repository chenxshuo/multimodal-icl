import open_clip
import torch
from tqdm import tqdm
import torch

from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


class RICES:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        vision_encoder_path="ViT-L-14",
        vision_encoder_pretrained="openai",
        cached_features=None,
        similar_in_topk=200,
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        # Load the model and processor
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = vision_encoder.to(self.device)
        self.image_processor = image_processor

        self.text_model = SentenceTransformer('sentence-transformers/sentence-t5-base').to(self.device)

        # Precompute features
        if cached_features is None:
            self.features = self._precompute_features()
        else:
            self.features = cached_features

        self.similar_in_topk = similar_in_topk
        # logger.critical(f"RICES: similar_in_topk: {self.similar_in_topk}")
        # assert False

    def _precompute_features(self):
        features = []

        # Switch to evaluation mode
        self.model.eval()

        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
            num_workers=8,
        )

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Precomputing features for RICES",
            ):
                batch = batch["image"]
                inputs = torch.stack(
                    [self.image_processor(image) for image in batch]
                ).to(self.device)
                image_features = self.model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features.detach())

        features = torch.cat(features)
        features = features.to("cpu")
        return features

    def find(self, batch, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            inputs = torch.stack([self.image_processor(image) for image in batch]).to(
                self.device
            )

            # Get the feature of the input image
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]

    def find_every_nth(self, batch, num_examples, n=8):
        """
        Get the top num_examples most similar examples to the images.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            inputs = torch.stack([self.image_processor(image) for image in batch]).to(
                self.device
            )

            # Get the feature of the input image
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples*n]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)[::n]] for row in indices]


    def find_by_ranking_similar_text(self, batch_image, batch_text, num_examples, with_answers=False, do_reverse=False):
        """
        RICES Images -> rank based on text similarity

        Args:
            batch ():
            num_examples ():

        Returns:

        """
        self.model.eval()

        with torch.no_grad():
            inputs = torch.stack([self.image_processor(image) for image in batch_image]).to(
                self.device
            )

            # Get the feature of the input image
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)


            # logger.debug(f"query_feature shape: {query_feature.shape}")
            # logger.debug(f"self.features shape: {self.features.shape}")
            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)
            # logger.debug(f"similarity shape: {similarity.shape}")
            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :self.similar_in_topk]
            rices_samples = [[self.dataset[i] for i in reversed(row)] for row in indices]
            # indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]
            # return [[self.dataset[i] for i in reversed(row)] for row in indices]

            # rank based on text similarity

            text_inputs = [text for text in batch_text]
            # logger.debug(f"text_inputs: {text_inputs}")
            text_query_features = self.text_model.encode(
                text_inputs,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            text_query_features /= text_query_features.norm(dim=-1, keepdim=True)

            if self.dataset.dataset_name in ["vqav2", "ok_vqa", "vizwiz", "gqa", "textvqa"]:
                if with_answers:
                    rices_samples_text = [[sample["question"] + " " + ", ".join(sample["answers"]) for sample in samples] for samples in rices_samples]
                else:
                    rices_samples_text = [[sample["question"] for sample in samples] for samples in rices_samples]
            elif self.dataset.dataset_name in ["coco", "flickr"]:
                rices_samples_text = [[sample["caption"] for sample in samples] for samples in rices_samples]
            else:
                raise NotImplementedError(f"dataset_name: {self.dataset.dataset_name} not supported")

            rices_samples_text_features = torch.stack([self.text_model.encode(
                sample_text,
                convert_to_tensor=True,
                show_progress_bar=False,
            ) for sample_text in rices_samples_text])

            rices_samples_text_features /= rices_samples_text_features.norm(dim=-1, keepdim=True)
            text_query_features = text_query_features.unsqueeze(dim=1)
            # logger.debug(f"rices_samples_text_features.shape: {rices_samples_text_features.shape}"
            #              f"text_query_features.shape: {text_query_features.shape}")
            # text_similarity = (text_query_features @ rices_samples_text_features.T).squeeze()
            text_similarity = torch.einsum("bij,bkj->bki", text_query_features, rices_samples_text_features)
            text_similarity = text_similarity.squeeze(dim=-1)
            # logger.debug(f"text_similarity.shape: {text_similarity.shape}")
            indices = text_similarity.argsort(dim=-1, descending=True)[:, :num_examples] # TODO,

            # demos = [[rices_samples[j][i] for i in reversed(row)] for j,row in enumerate(indices)]
            # sub_ind = torch.randperm(indices.shape[1])
            # indices = indices[:, sub_ind[:num_examples]]
            # logger.debug(f"indices.shape: {indices.shape}")
            # assert False
        if do_reverse:
            return [[rices_samples[j][i] for i in reversed(row)] for j,row in enumerate(indices)]
        else:
            return [[rices_samples[j][i] for i in row] for j,row in enumerate(indices)]


