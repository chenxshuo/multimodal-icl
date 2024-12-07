"""
Based on: https://github.com/lucidrains/flamingo-pytorch
"""

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn
import logging

logger = logging.getLogger(__name__)

def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=64,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


# gated cross attention
class MaskedCrossAttention(nn.Module):
    LAYER_NUMBER = 0
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        only_attend_immediate_media=True,
        hide_demo_media_embs=False,
        hide_query_media_embs=False,
    ):
        super().__init__()
        self.layer_number = MaskedCrossAttention.LAYER_NUMBER
        MaskedCrossAttention.LAYER_NUMBER += 1
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # check for OF 4B
        logger.debug(f"==== Initialize MaskedCrossAttention layer {self.layer_number} ====")
        #logger.debug(f"dim is {dim}") #2560
        #logger.debug(f"dim_visual is {dim_visual}") # 1024
        #logger.debug(f"dim_head is {dim_head}") # 64
        #logger.debug(f"heads is {heads}") # 8
        #logger.debug(f"inner_dim is {inner_dim}") # 512
        #logger.debug(f"only_attend_immediate_media is {only_attend_immediate_media}") # True
        #logger.debug(f"self.norm is {self.norm}")
        #logger.debug(f"self.to_q is {self.to_q}")
        #logger.debug(f"self.to_kv is {self.to_kv}")
        #logger.debug(f"self.to_out is {self.to_out}")

        # whether for text to only attend to immediate preceding image, or all previous images
        # True
        self.only_attend_immediate_media = only_attend_immediate_media

        self.hide_demo_media_embs = hide_demo_media_embs
        self.hide_query_media_embs = hide_query_media_embs

        self.use_robust_prompting = False
        self.number_of_robust_media = None # number of extra query media, default 3 if needed for robust prompting

        """
        my previous implementation of robust prompting is to put the aug rob features of the query image 
        at the end of the input features and adjust the attention mask to let the query token see the aug features.
        """
        self.robust_prompting_at_last = False

        self.guide_attention = False
        # self.attention_amplify_factor = 1.3
        # self.guide_head_index = [0, 1, 2, 3]  # 4 heads

        self.amplified_attention = None


    def set_attention_amplify(self, attn_amplify_layer, guide_attention=True):
        self.amplified_attention = attn_amplify_layer
        self.guide_attention = guide_attention
        logger.info(f"Set amplified_attention_module to {attn_amplify_layer}"
                    f" in MaskedCrossAttention layer {self.layer_number}")

    # def set_guide_attention(self, guide_attention, attention_amplify_factor, guide_head_index):
    #     self.guide_attention = guide_attention
    #     self.attention_amplify_factor = attention_amplify_factor
    #     self.guide_head_index = guide_head_index

    # def set_attention_amplify_factor(self, attention_amplify_factor):
    #     self.attention_amplify_factor = attention_amplify_factor
    #
    # def set_guide_head_index(self, guide_head_index):
    #     self.guide_head_index = guide_head_index

    def set_robust_prompting_at_last(self, robust_prompting_at_last):
        self.robust_prompting_at_last = robust_prompting_at_last

    def set_use_robust_prompting(self, use_robust_prompting):
        self.use_robust_prompting = use_robust_prompting

    def set_number_of_robust_media(self, number_of_robust_media):
        self.number_of_robust_media = number_of_robust_media

    def forward(self, x, media, media_locations=None, use_cached_media=False):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            use_cached_media: bool
                If true, treat all of x as if they occur after the last media
                registered in media_locations. T_txt does not need to exactly
                equal media_locations.shape[1] in this case
        """
        # logger.debug(f"============In layer {self.layer_number} of MaskedCrossAttention============")
        # logger.debug(f"x shape is {x.shape}")
        # logger.debug(f"media shape is {media.shape}")
        # logger.debug(f"media_locations shape is {media_locations.shape}")
        # logger.debug(f"media_locations is {media_locations}")
        #logger.debug(f"use_cached_media is {use_cached_media}")
        if not use_cached_media:
            assert (
                media_locations.shape[1] == x.shape[1]
            ), f"media_location.shape is {media_locations.shape} but x.shape is {x.shape}"
        # number of tokens
        T_txt = x.shape[1]
        # batch size, number of image tokens?, number of latents
        # n is 64
        _, T_img, n = media.shape[:3]
        h = self.heads
        #logger.debug(f"head is {h}")

        x = self.norm(x)
        #logger.debug(f"after norm, x shape is {x.shape}")

        q = self.to_q(x) # q takes the language features
        #logger.debug(f"q shape is {q.shape}")

        if self.hide_demo_media_embs:
            media[:, :-1] = 0
            # logger.info(f"set media[:, :-1] to 0, media shape is {media.shape}")
            for i in range(media.shape[1]-1):
                # logger.info(f"i is {i}")
                # logger.info(f"media[:, {i}] sum is {media[:, i].sum()}")
                # logger.info(f"media[:, {i}] shape is {media[:, i].shape}")
                # logger.info(f"media[:, {i}] is {media[:, i]}")
                assert media[:, i].sum() == 0
            # assert media[:, -1].sum() != 0
        if self.hide_query_media_embs:
            media[:, -1] = 0
            assert media[:, -1].sum() == 0
            # logger.info(f"set media[:, -1] to 0, media shape is {media.shape}")
        media = rearrange(media, "b t n d -> b (t n) d")
        # logger.debug(f"after rearrange media shape is {media.shape}")
        k, v = self.to_kv(media).chunk(2, dim=-1) # kv, take the media features
        #logger.debug(f"k shape is {k.shape}")
        #logger.debug(f"v shape is {v.shape}")
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        #logger.debug(f"After rearrange_many")
        #logger.debug(f"q shape is {q.shape}")
        #logger.debug(f"k shape is {k.shape}")
        #logger.debug(f"v shape is {v.shape}")

        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)
        # logger.debug(f"sim = qk, sim shape is {sim.shape}")

        if exists(media_locations):
            media_time = torch.arange(T_img, device=x.device) + 1
            # if two images, then media_time = tensor([1, 2])
            # logger.debug(f"media_time shape is {media_time.shape}, media_time is {media_time}")


            if use_cached_media:
                # text time is set to the last cached media location
                text_time = repeat(
                    torch.count_nonzero(media_locations, dim=1),
                    "b -> b i",
                    i=T_txt,
                )
                # logger.debug(f"use_cached_media is True, text_time shape is {text_time.shape}")
            else:
                # at each boolean of True, increment the time counter (relative to media time)
                text_time = media_locations.cumsum(dim=-1)
                # text_time = tensor([1,1,1,1,1,2,2,2]) tokens for the first images is 1, tokens for the second image is 2 so on
                # logger.debug(f"use_cached_media is False, text_time shape is {text_time.shape}, text_time is {text_time}")


            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"), # shape (1, 8) -> shape (1, 1, 8, 1)
                repeat(media_time, "j -> 1 1 1 (j n)", n=n), # shape (1, 2) -> shape (1, 1, 1, 2*64)
            )
            # logger.debug(f"text_to_media_mask shape {text_to_media_mask.shape}, text_to_media_mask is {text_to_media_mask}")
            if self.use_robust_prompting:
                if self.robust_prompting_at_last:
                    assert False # should avoid this branch; robust prompting should not be at last
                    # logger.critical(f"Use robust prompting at last")
                    assert self.number_of_robust_media is not None
                    # text_to_media_mask[:][:][torch.eq(text_time, torch.max(text_time, dim=1)[0])][(torch.max(media_time) - self.number_of_robust_media)*n:] = True
                    text_col_mask = text_time == text_time.max(dim=1)[0].reshape(-1, 1)
                    text_col_mask = rearrange(text_col_mask, "b i -> b 1 i")
                    part = text_to_media_mask[text_col_mask]
                    part[:, (torch.max(media_time) - self.number_of_robust_media)*n:] = True
                    text_to_media_mask[text_col_mask] = part

                    # try out attention guidance
                    # import ipdb; ipdb.set_trace()
                    # ATTENTION_GUIDANCE_FLAG = False
                    if self.guide_attention:
                        amplify = self.attention_amplify_factor
                        rob_start_ind = (torch.max(media_time) - (self.number_of_robust_media-1)) * n # the 1st is the original image, only amplify the rest
                        sim[:, :, :, rob_start_ind:] = sim[:, :, :, rob_start_ind:] * amplify
                else:
                    # logger.critical(f"Robust augs are not at last")
                    assert self.number_of_robust_media is not None
                    if self.guide_attention and self.amplified_attention is not None:
                        # import ipdb; ipdb.set_trace()
                        # amplify = self.attention_amplify_factor
                        # head_index = self.guide_head_index
                        #
                        # number_of_aug = self.number_of_robust_media - 1  # TODO
                        # total_media_token_num = sim.shape[-1]
                        # total_media_num = total_media_token_num // n
                        # rob_start_ind = (total_media_num - number_of_aug - 1) * n
                        # rob_end_ind = total_media_token_num - n
                        # sim[:, head_index, :, rob_start_ind:rob_end_ind] = sim[:, head_index, :, rob_start_ind:rob_end_ind] * amplify
                        number_of_aug = self.number_of_robust_media - 1  # TODO
                        sim = self.amplified_attention(attention_scores=sim, number_of_aug=number_of_aug, media_token_num=n)


            # import ipdb;ipdb.set_trace()
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        # if self.layer_number == 7: # magic number for OF-9B, 7 is the last masked cross attention layer
        #     import ipdb; ipdb.set_trace()

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        # attention matrix
        attn = sim.softmax(dim=-1)
        #logger.debug(f"attn shape after sim_masked and softmax is {attn.shape}")

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(
                text_without_media_mask, "b i -> b 1 i 1"
            )
            attn = attn.masked_fill(text_without_media_mask, 0.0)
            # logger.debug(f"masked again attn shape after text_without_media_mask is {attn.shape}")
        # if attn.shape == torch.Size([1, 8, 1, 192]):
        #     torch.save(attn, f"./store_attention/masked_attn_after_generate_1st_token_{self.layer_number}.pt")
        #     assert False
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        #logger.debug(f"out shape is {out.shape}")
        out = rearrange(out, "b h n d -> b n (h d)")
        #logger.debug(f"out shape after rearrange is {out.shape}")
        out = self.to_out(out)
        #logger.debug(f"out shape after to_out is {out.shape}")
        return out


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True,
        hide_demo_media_embs=False,
        hide_query_media_embs=False,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
            hide_demo_media_embs=hide_demo_media_embs,
            hide_query_media_embs=hide_query_media_embs,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

        self.hide_demo_media_embs = hide_demo_media_embs
        self.hide_query_media_embs = hide_query_media_embs

        self.attn_output = []

    def set_attention_amplify(self, attn_amplify_layer, guide_attention=True):
        self.attn.set_attention_amplify(attn_amplify_layer, guide_attention)

    def set_guide_attention(self, guide_attention, attention_amplify_factor, guide_head_index):
        self.attn.set_guide_attention(guide_attention, attention_amplify_factor, guide_head_index)

    def set_robust_prompting_at_last(self, robust_prompting_at_last):
        self.attn.set_robust_prompting_at_last(robust_prompting_at_last)

    def set_use_robust_prompting(self, use_robust_prompting):
        self.attn.set_use_robust_prompting(use_robust_prompting)

    def set_number_of_robust_media(self, number_of_robust_media):
        self.attn.set_number_of_robust_media(number_of_robust_media)

    def set_attn_output(self, attn_output):
        self.attn_output.append(attn_output.cpu().detach())

    def get_attn_output(self):
        return self.attn_output

    def forward(
        self,
        x,
        media,
        media_locations=None,
        use_cached_media=False,
    ):
        #logger.debug(f"in GatedCrossAttentionBlock forward")
        # attn_out = self.attn(
        #         x,
        #         media,
        #         media_locations=media_locations,
        #         use_cached_media=use_cached_media,
        #     )
        # self.set_attn_output(attn_out)
        #logger.debug(f"x shape is {x.shape}") # 1, 22, 2560
        #logger.debug(f"attn_out shape is {attn_out.shape}") # 1, 22, 2560
        # x = attn_out * self.attn_gate.tanh() + x
        #logger.debug(f"after attn tanh gate and residual, x shape is {x.shape}")
        # import ipdb;ipdb.set_trace()

        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
                use_cached_media=use_cached_media,
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x

        #logger.debug(f"after ff tanh gate and residual, x shape is {x.shape}")
        #logger.debug("out GatedCrossAttentionBlock forward")
        return x
