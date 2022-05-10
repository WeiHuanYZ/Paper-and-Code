"""
This model is based on the implementation of https://github.com/lucidrains/vit-pytorch.
"""
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch,types
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):

    def __init__(self, p, **kwargs):
        super().__init__()

        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)

        return x

    def extra_repr(self):
        return "p=%s" % repr(self.p)
class Lambda(nn.Module):

    def __init__(self, lmd):
        super().__init__()
        if not isinstance(lmd, types.LambdaType):
            raise Exception("`lmd` should be lambda ftn.")
        self.lmd = lmd

    def forward(self, x):
        return self.lmd(x)


class FeedForward(nn.Module):

    def __init__(self, dim_in, hidden_dim, dim_out=None, *,
                 dropout=0.0,
                 f=nn.Linear, activation=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.net = nn.Sequential(
            f(dim_in, hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            f(hidden_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention1d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0,use_cos=False):
        super().__init__()
        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)
        self.use_cos=use_cos
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        if self.use_cos:
            q = q / q.pow(2).sum(-1, keepdim=True).sqrt()
            k = k / k.pow(2).sum(-1, keepdim=True).sqrt()
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        dots = dots + mask if mask is not None else dots
        if not self.use_cos:
            attn = dots.softmax(dim=-1)
        else:
            attn = F.leaky_relu(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn


class Attention2d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, k=1,use_cos=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out
        self.use_cos=use_cos
        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _, y = x.shape
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)
        if self.use_cos:
            q=q/q.pow(2).sum(-1,keepdim=True).sqrt()
            k=k/k.pow(2).sum(-1,keepdim=True).sqrt()
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        if not self.use_cos:
            attn = dots.softmax(dim=-1)
        else:
            attn = F.leaky_relu(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)

        out = self.to_out(out)

        return out, attn


class Transformer(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dim_mlp=1024, dropout=0.0, sd=0.0,
                 attn=Attention1d, norm=nn.LayerNorm,
                 f=nn.Linear, activation=nn.GELU,use_cos=False):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.shortcut = []
        if dim_in != dim_out:
            self.shortcut.append(norm(dim_in))
            self.shortcut.append(nn.Linear(dim_in, dim_out))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.norm1 = norm(dim_in)
        self.attn = attn(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout,use_cos=use_cos)
        self.sd1 = DropPath(sd) if sd > 0.0 else nn.Identity()

        self.norm2 = norm(dim_out)
        self.ff = FeedForward(dim_out, dim_mlp, dim_out, dropout=dropout, f=f, activation=activation)
        self.sd2 = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        skip = self.shortcut(x)
        x = self.norm1(x)
        x, attn = self.attn(x, mask=mask)
        x = self.sd1(x) + skip

        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.sd2(x) + skip

        return x

class PatchEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, dim_out, channel=3):
        super().__init__()
        if not image_size % patch_size == 0:
            raise Exception("Image dimensions must be divisible by the patch size.")
        patch_dim = channel * patch_size ** 2

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim_out),
        )

    def forward(self, x):
        x = self.patch_embedding(x)

        return x


class CLSToken(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        return x


class AbsPosEmbedding(nn.Module):

    def __init__(self, image_size, patch_size, dim, stride=None, cls=True):
        super().__init__()
        if not image_size % patch_size == 0:
            raise Exception("Image dimensions must be divisible by the patch size.")
        stride = patch_size if stride is None else stride
        output_size = self._conv_output_size(image_size, patch_size, stride)
        num_patches = output_size ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + int(cls), dim) * 0.02)

    def forward(self, x):
        x = x + self.pos_embedding

        return x

    @staticmethod
    def _conv_output_size(image_size, kernel_size, stride, padding=0):
        return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


class PatchUnembedding(nn.Module):

    def __init__(self, image_size, patch_size):
        super().__init__()
        h, w = image_size // patch_size, image_size // patch_size

        self.rearrange = nn.Sequential(
            Rearrange('b (h w) (p1 p2 d) -> b d (h p1) (w p2)',
                      h=h, w=w, p1=patch_size, p2=patch_size),
        )

    def forward(self, x):
        x = x[:, 1:]
        x = self.rearrange(x)

        return x


class ConvEmbedding(nn.Module):

    def __init__(self, patch_size, dim_out, channel=3, stride=None):
        super().__init__()
        stride = patch_size if stride is None else stride
        patch_dim = channel * patch_size ** 2

        self.patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=stride),
            Rearrange('b c n -> b n c'),
            nn.Linear(patch_dim, dim_out)
        )

    def forward(self, x):
        x = self.patch_embedding(x)

        return x


class ViT(nn.Module):

    def __init__(self, *,
                 image_size, patch_size, num_classes, depth, dim, heads, dim_mlp,
                 channel=3, dim_head=64, dropout=0.0, emb_dropout=0.0, sd=0.0,
                 embedding=None, classifier=None,use_cos=False,
                 name="vit", **block_kwargs):
        super().__init__()
        self.name = name

        self.embedding = nn.Sequential(
            PatchEmbedding(image_size, patch_size, dim, channel=channel),
            CLSToken(dim),
            AbsPosEmbedding(image_size, patch_size, dim, cls=True),
            nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity()
        ) if embedding is None else embedding

        self.transformers = []
        for i in range(depth):
            self.transformers.append(
                Transformer(dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,
                            dropout=dropout, sd=(sd * i / (depth - 1)),use_cos=use_cos)
            )
        self.transformers = nn.Sequential(*self.transformers)

        self.classifier = nn.Sequential(
            Lambda(lambda x: x[:, 0]),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) if classifier is None else classifier

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformers(x)
        x = self.classifier(x)

        return x


def tiny(num_classes=1000, name="vit_ti",
         image_size=224, patch_size=16, channel=3,
         depth=12, dim=192, heads=3, dim_head=64, dim_mlp=768,use_cos=False,
         **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,use_cos=use_cos,
        name=name, **block_kwargs,
    )


def small(num_classes=1000, name="vit_s",
          image_size=224, patch_size=16, channel=3,
          depth=12, dim=384, heads=6, dim_head=64, dim_mlp=1536,,use_cos=False,
          **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,use_cos=use_cos,
        name=name, **block_kwargs,
    )


def base(num_classes=1000, name="vit_b",
         image_size=224, patch_size=16, channel=3,
         depth=12, dim=768, heads=12, dim_head=64, dim_mlp=3072,use_cos=False,
         **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,use_cos=use_cos,
        name=name, **block_kwargs,
    )


def large(num_classes=1000, name="vit_l",
          image_size=224, patch_size=16, channel=3,
          depth=24, dim=1024, heads=16, dim_head=64, dim_mlp=4096,use_cos=False,
          **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,use_cos=use_cos,
        name=name, **block_kwargs,
    )


def huge(num_classes=1000, name="vit_h",
         image_size=224, patch_size=16, channel=3,
         depth=32, dim=1280, heads=16, dim_head=80, dim_mlp=5120,use_cos=False,
         **block_kwargs):
    return ViT(
        image_size=image_size, patch_size=patch_size, channel=channel,
        num_classes=num_classes, depth=depth,
        dim=dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp,use_cos=use_cos,
        name=name, **block_kwargs,
    )