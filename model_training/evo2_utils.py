"""evo2_utils.py — load Evo 2 7B, run a truncated forward, inject LoRA.

Facts this relies on (read from evo2 0.6.0 / vtx 1.1.0 source, 2026-10-25):
  * Evo2(name, local_path=...) builds a vortex StripedHyena at .model. The 7B checkpoints run
    in bf16 WITHOUT Transformer Engine (FP8 projections fall back to bf16 automatically).
  * Evo2.forward() wraps everything in torch.no_grad(), so training must call the inner
    StripedHyena pieces directly: embedding_layer -> blocks[i](x, inference_params=None,
    padding_mask=None) -> returns (x, None).
  * evo2_7b_base: 32 blocks, hidden 4096. Attention blocks 3,10,17,24,31 (MHA Wqkv/out_proj);
    the rest are Hyena blocks (projections = TELinear, out_filter_dense = nn.Linear). Every
    block has an MLP with l1/l2/l3 (nn.Linear). Embedding is tied with the unembedding.
  * The model places blocks across ALL visible GPUs. Pin one GPU with CUDA_VISIBLE_DEVICES.
  * All our windows are exactly 500 bp, so batches need no padding mask.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def gpu_gb(peak=False):
    """Allocated (or peak) GPU memory in GB; 0 on CPU."""
    if not torch.cuda.is_available():
        return 0.0
    return (torch.cuda.max_memory_allocated() if peak else torch.cuda.memory_allocated()) / 1e9


def load_evo2(model_name='evo2_7b_base', local_path=None, keep_blocks=None):
    """Load Evo 2 and (optionally) drop blocks after `keep_blocks - 1` to save memory.
    Returns (evo2_wrapper, striped_hyena, tokenizer)."""
    from evo2 import Evo2
    ev = Evo2(model_name, local_path=local_path)
    m = ev.model
    if keep_blocks is not None and keep_blocks < len(m.blocks):
        m.blocks = nn.ModuleList(list(m.blocks)[:keep_blocks])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return ev, m, ev.tokenizer


def make_autograd_safe(m):
    """vortex's load_checkpoint loads weights inside torch.inference_mode() (vortex/model/utils.py),
    so parameters/buffers are 'inference tensors': fine for no_grad forward passes, but autograd
    refuses to save them for backward ("Inference tensors cannot be saved for backward").
    Replace each one with a normal copy, one tensor at a time. Needed only for training (LoRA)."""
    n = 0
    with torch.no_grad():
        for mod in m.modules():
            for name, p in list(mod._parameters.items()):
                if p is not None and p.is_inference():
                    mod._parameters[name] = nn.Parameter(p.clone(), requires_grad=p.requires_grad); n += 1
            for name, b in list(mod._buffers.items()):
                if b is not None and b.is_inference():
                    mod._buffers[name] = b.clone(); n += 1
            for name, v in list(vars(mod).items()):          # plain tensor attributes (e.g. cached filters)
                if isinstance(v, torch.Tensor) and not isinstance(v, nn.Parameter) and v.is_inference():
                    setattr(mod, name, v.clone()); n += 1
    return n


def tokenize(tok, seqs, device):
    return torch.tensor([tok.tokenize(s) for s in seqs], dtype=torch.long, device=device)


def forward_blocks(m, ids, read_layers, grad_ckpt=False):
    """Run embedding + blocks[0..max(read_layers)]; return {layer: hidden [B, L, D]}.
    Block outputs are BEFORE the model's final norm."""
    from torch.utils.checkpoint import checkpoint
    want, last = set(read_layers), max(read_layers)
    x = m.embedding_layer(ids)
    out = {}
    for i, block in enumerate(m.blocks):
        if grad_ckpt and torch.is_grad_enabled():
            x = checkpoint(lambda h, b=block: b(h, inference_params=None, padding_mask=None)[0], x,
                           use_reentrant=False)
        else:
            x, _ = block(x, inference_params=None, padding_mask=None)
        if i in want:
            out[i] = x
        if i == last:
            break
    return out


class LoRALinear(nn.Module):
    """y = base(x) + scale * B(A(dropout(x))). Base frozen (bf16); A, B trainable (fp32)."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.drop = nn.Dropout(dropout)
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        lx = F.linear(F.linear(self.drop(x), self.A.to(x.dtype)), self.B.to(x.dtype))
        return y + self.scale * lx

    @property
    def weight(self):        # some callers read .weight.dtype / .device
        return self.base.weight


def inject_lora(m, targets, max_block, r=16, alpha=32, dropout=0.05):
    """Wrap every plain nn.Linear whose leaf name is in `targets`, in blocks 0..max_block.
    Returns list of wrapped module names. Subclasses of nn.Linear (e.g. ones that return
    tuples) are skipped on purpose."""
    wrapped = []
    for bi, block in enumerate(m.blocks):
        if bi > max_block:
            break
        for name, mod in list(block.named_modules()):
            leaf = name.rsplit('.', 1)[-1]
            if leaf in targets and type(mod) is nn.Linear:
                parent = block.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else block
                setattr(parent, leaf, LoRALinear(mod, r, alpha, dropout))
                wrapped.append(f'blocks.{bi}.{name}')
    return wrapped


def lora_state_dict(m):
    return {k: v.detach().cpu() for k, v in m.state_dict().items() if k.endswith('.A') or k.endswith('.B')}
