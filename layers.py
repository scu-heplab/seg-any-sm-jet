import torch


class PositionEmbedding(torch.nn.Module):
    def __init__(self, length, d_model):
        super(PositionEmbedding, self).__init__()

        embedding = torch.arange(length).view(1, -1, 1) / torch.pow(10000, 2 * torch.arange(d_model // 2).view(1, 1, -1) / d_model)
        self._embedding = torch.nn.Parameter(torch.stack([torch.sin(embedding), torch.cos(embedding)], -1).view(1, length, -1), False)

    def forward(self, inputs):
        return inputs + self._embedding


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()

        self._eps = eps
        self._dim = dim

        self._scale = torch.nn.Parameter(torch.ones(dim))

    def forward(self, inputs):
        norm_x = inputs.norm(2, dim=-1, keepdim=True)

        rms_x = norm_x * self._dim ** (-1 / 2)
        x_normed = inputs / (rms_x + self._eps)

        return self._scale * x_normed


class RMSNorm2D(torch.nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm2D, self).__init__()

        self._eps = eps
        self._dim = dim

        self._scale = torch.nn.Parameter(torch.ones(dim, 1, 1))
        self.register_parameter("scale", self._scale)

    def forward(self, inputs):
        norm_x = inputs.norm(2, dim=1, keepdim=True)

        rms_x = norm_x * self._dim ** (-1 / 2)
        x_normed = inputs / (rms_x + self._eps)

        return self._scale * x_normed


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.nn.functional.sigmoid(x)
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, head_nums, head_size, out_dim=None, key_size=None):
        super(MultiHeadAttention, self).__init__()

        self._head_nums = head_nums
        self._head_size = head_size
        self._out_dim = out_dim or head_nums * head_size
        self._key_size = key_size or head_size

        self._query = torch.nn.Linear(d_model, self._key_size * self._head_nums)
        self._key = torch.nn.Linear(d_model, self._key_size * self._head_nums)
        self._value = torch.nn.Linear(d_model, self._head_size * self._head_nums)
        self._concat = torch.nn.Linear(self._head_size * self._head_nums, self._out_dim)

        self._rms1 = RMSNorm(d_model)

        self._mlp = MLP(d_model, d_model // 2, d_model, 2)

        self._rms2 = RMSNorm(d_model)

        self._scale = 1.0 / torch.sqrt(torch.as_tensor([d_model]))

    def forward(self, inputs, mask=None):
        query, key, value = inputs

        q = self._query(query)
        k = self._key(key)
        v = self._value(value)

        q = torch.reshape(q, (-1, q.shape[1], self._head_nums, self._key_size))
        k = torch.reshape(k, (-1, k.shape[1], self._head_nums, self._key_size))
        v = torch.reshape(v, (-1, v.shape[1], self._head_nums, self._head_size))

        a = torch.einsum("b n h d, b m h d -> b h n m", q, k) * self._scale.to(q.device)
        if mask is not None:
            a = a - (1.0 - mask[:, None, None]) * 1e12
        a = torch.softmax(a, -1)
        o = torch.einsum("b h n m, b m h d -> b n h d", a, v)
        o = self._concat(torch.reshape(o, (-1, o.shape[1], self._head_nums * self._head_size)))

        rms = self._rms1(o + query)
        mlp = self._rms2(rms + self._mlp(rms))
        return mlp


class CrossAttention(torch.nn.Module):
    def __init__(self, d_model, token_length, patch_length, head_nums, head_size, out_dim=None, key_size=None):
        super(CrossAttention, self).__init__()

        self._patch_to_token = MultiHeadAttention(d_model, head_nums, head_size, out_dim, key_size)
        self._token_to_feature = MultiHeadAttention(d_model, head_nums, head_size, out_dim, key_size)

        self._patch_position = PositionEmbedding(patch_length, d_model)
        self._token_position = PositionEmbedding(token_length, d_model)

    def forward(self, inputs):
        token, patch = inputs

        token = self._token_position(token)
        patch = self._patch_position(patch)

        token = self._patch_to_token([token, patch, patch])
        feature = self._token_to_feature([patch, token, token])

        return token, feature


class Transformer(torch.nn.Module):
    def __init__(self, d_model, token_length, patch_length, head_nums, head_size, out_dim=None, key_size=None, deep=2):
        super(Transformer, self).__init__()

        self._layers = torch.nn.ModuleList()
        for i in range(deep):
            self._layers.append(CrossAttention(d_model, token_length, patch_length, head_nums, head_size, out_dim, key_size))

        self._feature_to_token = MultiHeadAttention(d_model, head_nums, head_size, out_dim, key_size)

    def forward(self, inputs):
        token, feature = inputs

        for layer in self._layers:
            token, feature = layer([token, feature])

        token = self._feature_to_token([token, feature, feature])
        return token, feature


class MaskDecoder(torch.nn.Module):
    def __init__(self, d_model, n_mask=32, n_class=5):  # n_class = (t, H, W/Z, b, j)
        super(MaskDecoder, self).__init__()

        self._n_mask = n_mask
        self._n_out_token = self._n_mask

        self._out_token = torch.nn.Parameter(torch.rand(self._n_out_token, d_model))
        # self._label_embedding = torch.nn.Embedding(n_class + 1, d_model, 0)

        self._transformer = Transformer(d_model, self._n_out_token, 63 * 63, 4, d_model // 4)

        self._iou_mlp = MLP(d_model, d_model, 1, 3)
        self._mask_mlp = MLP(d_model, d_model, d_model // 4, 3)
        self._category_mlp = MLP(d_model, d_model, n_class + 1, 3)
        self._momentum_mlp = MLP(d_model, d_model, 4, 3)

        self._upscale_conv = torch.nn.Sequential(torch.nn.ConvTranspose2d(d_model, d_model // 2, 2, 2), RMSNorm2D(d_model // 2), torch.nn.GELU(),
                                                 torch.nn.ConvTranspose2d(d_model // 2, d_model // 4, 2, 2), torch.nn.GELU())

    def forward(self, inputs):
        patch_embedding, label = inputs
        # prompt = self._label_embedding(label)

        token = torch.tile(self._out_token[None], [label.shape[0], 1, 1])
        patch_embedding = patch_embedding.reshape(patch_embedding.shape[0], patch_embedding.shape[1], -1).permute(0, 2, 1)

        token, feature = self._transformer([token, patch_embedding])
        feature = self._upscale_conv(feature.permute(0, 2, 1).reshape(feature.shape[0], feature.shape[2], 63, 63))

        iou = self._iou_mlp(token)
        mask = self._mask_mlp(token)
        category = self._category_mlp(token)
        momentum = self._momentum_mlp(token)

        mask = torch.einsum("b n m, b m i j -> b n i j", mask, feature)

        return torch.squeeze(iou, -1), mask, category, momentum
