# -*- coding: utf-8 -*-

"""Retrieval-based IC Example Selection based on text similarity."""

import logging
import open_clip
from sentence_transformers import SentenceTransformer
import torch
from utils import custom_collate_fn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RICESText:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        lm_model='sentence-transformers/sentence-t5-base',
        cached_features=None,
        similar_in_topk=200,
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        # Load the model and processor
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
        )
        self.vision_model = vision_encoder.to(self.device)
        self.image_processor = image_processor

        self.text_model = SentenceTransformer(lm_model).to(self.device)
        # Precompute features
        if cached_features is None:
            self.features = self._precompute_features()
        else:
            self.features = cached_features
        self.similar_in_topk = similar_in_topk

    def _precompute_features(self):
        features = []
        self.text_model.eval()
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
                batch = batch["question"] if "question" in batch else batch["caption"] #TODO for Image Classification
                text_features = self.text_model.encode(
                    [text for text in batch],
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                text_features /= text_features.norm(dim=-1, keepdim=True)
                # logger.debug(f"Text features type {type(text_features)}")
                # logger.debug(f"Text features shape: {text_features.shape}")
                # assert False
                features.append(text_features.detach())

            features = torch.cat(features)
            return features

    def find(self, batch, num_examples):
        self.text_model.eval()
        with torch.no_grad():
            query_feature = self.text_model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
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

    def find_by_ranking_similar_images(self, batch_image, batch_text, num_examples, do_reverse=False):
        """
        RICES-TEXT -> rank based on image similarity
        Args:
            batch_image ():
            batch_text ():
            num_examples ():
            do_reverse ():

        Returns:

        """
        self.text_model.eval()
        with torch.no_grad():
            query_feature = self.text_model.encode(
                batch_text,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()
            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)
            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)
            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :200] #TODO
            rices_samples = [[self.dataset[i] for i in reversed(row)] for row in indices]
            logger.debug(f"RICES samples shape: {len(rices_samples)}")
            logger.debug(f"RICES samples[0] shape: {len(rices_samples[0])}")

            image_inputs = torch.stack([self.image_processor(image) for image in batch_image]).to(
                self.device
            )
            image_query_feature = self.vision_model.encode_image(image_inputs)
            image_query_feature /= image_query_feature.norm(dim=-1, keepdim=True)
            image_query_feature = image_query_feature.detach().cpu()

            if image_query_feature.ndim == 1:
                image_query_feature = image_query_feature.unsqueeze(0)

            samples_image = [[sample["image"] for sample in samples] for samples in rices_samples]
            logger.debug(f"Samples image shape: {len(samples_image)}")
            logger.debug(f"type samples image: {type(samples_image)}")
            whole_batches = []
            for images in samples_image:
                a_batch = torch.stack([self.image_processor(image) for image in images]).to(
                    self.device
                )
                a_batch.unsqueeze_(0)
                whole_batches.append(a_batch)
            samples_image = torch.cat(whole_batches, dim=0)
            logger.debug(f"Samples image shape: {samples_image.shape}")
            samples_image_feature = []
            for b in range(samples_image.shape[0]):
                a_batch_samples_image_feature = self.vision_model.encode_image(samples_image[b])
                a_batch_samples_image_feature /= a_batch_samples_image_feature.norm(dim=-1, keepdim=True)
                a_batch_samples_image_feature.unsqueeze_(0)
                samples_image_feature.append(a_batch_samples_image_feature)
            samples_image_feature = torch.cat(samples_image_feature, dim=0)
            samples_image_feature = samples_image_feature.detach().cpu()
            image_query_feature.unsqueeze_(dim=1)
            logger.debug(f"Samples image feature shape: {samples_image_feature.shape}")
            logger.debug(f"Image query feature shape: {image_query_feature.shape}")
            image_similarity = torch.einsum("bij,bkj->bki", image_query_feature, samples_image_feature)
            image_similarity = image_similarity.squeeze(dim=-1)
            logger.debug(f"Image similarity shape: {image_similarity.shape}")
            indices = image_similarity.argsort(dim=-1, descending=True)[:, :num_examples]  # TODO,
            logger.debug(f"Indices shape: {indices.shape}")
            logger.debug(f"Indices: {indices}")
        if do_reverse:
            return [[rices_samples[j][i] for i in reversed(row)] for j,row in enumerate(indices)]
        else:
            return [[rices_samples[j][i] for i in row] for j,row in enumerate(indices)]




