import torch
import torch.nn as nn
from .helpers import GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive
from einops import repeat

import logging

logger = logging.getLogger(__name__)

class FlamingoLayer(nn.Module):
    """
    FlamingoLayer is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    """
    LAYER_COUNTER = 0
    def __init__(
        self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False
    ):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
        self.text_prompt_locations = None
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = (
                gradient_checkpointing
            )
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

        self.using_soft_prompt = False
        self.soft_prompt_text = None
        self.layer_counter = FlamingoLayer.LAYER_COUNTER
        FlamingoLayer.LAYER_COUNTER += 1

        self.do_cocoop = False
        self.meta_net = None

    def set_using_soft_prompt(self, using_soft_prompt, soft_prompt_text):
        self.using_soft_prompt = using_soft_prompt
        self.soft_prompt_text = soft_prompt_text

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None and self.media_locations is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        # if media_locations is not None:
        #     logger.debug(f"set media locations for layer {self.layer_counter}")
        #     logger.debug(f"media_locations: {media_locations.shape}")
        #     logger.debug(f"media_locations[0]: {media_locations[0]}")
        self.media_locations = media_locations

    def condition_text_prompt_locations(self, text_prompt_locations):
        self.text_prompt_locations = text_prompt_locations

    def condition_use_cached_media(self, use_cached_media):
        self.use_cached_media = use_cached_media


    def set_use_robust_prompting(self, use_robust_prompt):
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer.set_use_robust_prompting(use_robust_prompt)

    def set_attention_amplify(self, attn_amplify_layer, guide_attention=True):
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer.set_attention_amplify(attn_amplify_layer, guide_attention)
        else:
            logger.warning(f"No gated cross attention layer found in layer {self.layer_counter}")

    def set_robust_prompting_at_last(self, robust_prompting_at_last):
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer.set_robust_prompting_at_last(robust_prompting_at_last)


    def set_guide_attention(self, guide_attention, attention_amplify_factor, guide_head_index):
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer.set_guide_attention(guide_attention, attention_amplify_factor, guide_head_index)

    def set_number_of_robust_media(self, number_of_robust_media):
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer.set_number_of_robust_media(number_of_robust_media)


    def set_cocoop_meta_net(self, meta_net):
        logger.info(f"Setting meta_net for layer {self.layer_counter}")
        self.meta_net = meta_net
        self.do_cocoop = True

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # import ipdb; ipdb.set_trace()
        if self.using_soft_prompt:
            # the model layer just after the embedding layer
            # need to insert the soft text prompt here
            # logger.debug(f"adding text soft prompt embedding at layer {self.layer_counter}")
            assert self.soft_prompt_text is not None
            assert self.text_prompt_locations is not None
            assert self.layer_counter == 0
            if self.text_prompt_locations.shape[0] != lang_x.shape[0]:
                # import ipdb; ipdb.set_trace()
                # if batch size unmatches, reshape the text prompt location
                self.text_prompt_locations = repeat(
                    torch.unsqueeze(self.text_prompt_locations[0], 0),
                    "1 i -> (1 n) i", n=lang_x.shape[0]
                )
            self.soft_prompt_text = self.soft_prompt_text.to(lang_x.device)


            if self.do_cocoop:
                if self.meta_net is not None:
                    # import ipdb; ipdb.set_trace()
                    im_features = self.vis_x.squeeze(dim=1).mean(dim=1)
                    bias = self.meta_net(im_features) # shape: (batch_size, dim)
                else:
                    bias = None
                    raise ValueError("meta_net is not set")

            if lang_x.shape[1] > self.text_prompt_locations.shape[1]:
                # import ipdb; ipdb.set_trace()
                tmp = torch.zeros(lang_x.shape[:2])
                tmp[:, :self.text_prompt_locations.shape[1]] = self.text_prompt_locations[:]
                self.text_prompt_locations = tmp.bool()
            elif lang_x.shape[1] < self.text_prompt_locations.shape[1]:
                # import ipdb; ipdb.set_trace()
                tmp = torch.zeros(lang_x.shape[:2])
                tmp[:, :lang_x.shape[1]] = self.text_prompt_locations[:, :lang_x.shape[1]]
                self.text_prompt_locations = tmp.bool()
            for i in range(lang_x.shape[0]):
                if self.do_cocoop:
                    # import ipdb; ipdb.set_trace()
                    lang_x[i][self.text_prompt_locations[i]] = self.soft_prompt_text + bias[i]
                else:
                    lang_x[i][self.text_prompt_locations[i]] = self.soft_prompt_text
            # lang_x[self.text_prompt_locations] = self.soft_prompt_text

        # Cross attention
        if self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")

            if self.media_locations is None:
                raise ValueError(
                    "media_locations must be conditioned before forward pass"
                )

            lang_x = self.gated_cross_attn_layer(
                lang_x,
                self.vis_x,
                media_locations=self.media_locations,
                use_cached_media=self.use_cached_media,
            )

        # Normal decoder layer
        lang_x = self.decoder_layer(
            lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
        )
        return lang_x


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(
        self,
        media_token_id,
        lang_hidden_size,
        vis_hidden_size,
        cross_attn_every_n_layers,
        gradient_checkpointing,
        only_attend_immediate_media=True,
        hide_demo_media_embs=False,
        hide_query_media_embs=False,
        prompt_media_id=None,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        # logger.info(f"only_attend_immediate_media: {only_attend_immediate_media}")
        # assert False
        self.old_decoder_blocks = self._get_decoder_layers()
        self.prompt_media_id = prompt_media_id
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=lang_hidden_size, dim_visual=vis_hidden_size,
                    only_attend_immediate_media=only_attend_immediate_media,
                    hide_demo_media_embs=hide_demo_media_embs,
                    hide_query_media_embs=hide_query_media_embs,
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self.init_flamingo_layers(gradient_checkpointing)
        self.media_token_id = media_token_id
        self.initialized_flamingo = True
        self._use_cached_vision_x = False

    def init_flamingo_layers(self, gradient_checkpointing):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    FlamingoLayer(
                        gated_cross_attn_layer, decoder_layer, gradient_checkpointing
                    )
                    for gated_cross_attn_layer, decoder_layer in zip(
                        self.gated_cross_attn_layers, self.old_decoder_blocks
                    )
                ]
            )
        )
        # for i, decoder_layer in enumerate(self._get_decoder_layers()):
        #     logger.info(f"Initializing Flamingo {i}th layer {decoder_layer}")


    def forward(self, input_ids, attention_mask, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        # the forward of MosaicGPT (OF-3B)

        if not self.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )
        media_locations = input_ids == self.media_token_id
        if self.prompt_media_id is not None:
            prompt_media_locations = input_ids == self.prompt_media_id
            media_locations = media_locations | prompt_media_locations


        # if there are media already cached and we're generating and there are no media tokens in the input,
        # we'll assume that ALL input tokens should attend to the last previous media that is cached.
        # this is especially important for HF generate() compatibility, since generate() calls forward()
        # repeatedly one token at a time (with no media tokens).
        # without this check, the model would not attend to any images when generating (after the first token)
        use_cached_media_locations = (
            self._use_cached_vision_x
            and self.is_conditioned()
            and not media_locations.any()
        )

        for layer in self._get_decoder_layers():
            if not use_cached_media_locations:
                layer.condition_media_locations(media_locations)
            layer.condition_use_cached_media(use_cached_media_locations)

        # package arguments for the other parent's forward. since we don't know the order of the arguments,
        # make them all kwargs
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        return super().forward(**kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_use_cached_media(None)
