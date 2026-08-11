"""Model definitions for SwiftVR: the DiT backbone and autoencoder/decoder paths."""

from .reae import ReAE, MemBlock, TPool, TGrow, Clamp
from .tiny_conditional_decoder import TinyConditionalDecoder, pack_rgb_condition
from .transformer import (
    WanTransformer3DModel,
    enable_shifted_window_self_attention,
    compile_transformer_blocks,
    set_attention_backend,
    get_attention_backend,
    list_available_attention_backends,
)
from .transformer_prompt_free import WanTransformer3DModelPromptFree
from .transformer_prompt_free_no_time import WanTransformer3DModelPromptFreeNoTime
