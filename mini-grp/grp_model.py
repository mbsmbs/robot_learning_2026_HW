import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_patches_fast(images, cfg):
    '''
    Converts images into a sequence of tokens for the Transformer:
    1. Slices the image into a grid of small squares (patches).
    2. Flattens each patch into a 1-dimensional vector (token).
    3. If there are multiple frames (history), it stacks those patches 
       into the sequence so the model can see movement over time.
    '''
    from einops import rearrange
    batch_size, height, width, channels = images.shape
    patch_size = cfg.patch_size ## n_patches = 8

    patches = rearrange(images[:,:,:,:3], 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    if channels > 3:
        ## History stacking in the channel dimension for observations only, not goal images.
        patches = rearrange(images, 'b (h p1) (w p2) (c hs) -> b (h w hs) (p1 p2 c)', p1 = patch_size, p2 = patch_size, hs=cfg.policy.obs_stacking) ## Stack the history in the channel dimension
    return patches


def calc_positional_embeddings(sequence_length, d):
    '''
    Generates a fixed pattern of sine/cosine waves to give the 
    Transformer a sense of token order and position. 
    - sequence_length: Number of tokens (patches/words).
    - d: Dimension of each token's embedding.
    '''
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        '''
        Executes the attention mechanism:
        1. Computes scores (wei) to determine how tokens relate to each other.
        2. Applies a mask to hide specific tokens (important for Multi-modal training).
        3. Returns a weighted combination of information based on these scores.
        '''
        B,T,C = x.shape
        # TODO: 
        ## Provide the block masking logic for the attention head
        k = self.key(x)     # (B, T, hs)
        q = self.query(x)   # (B, T, hs)
        wei = q @ k.transpose(-2,-1) * C**-0.5      # (B, T, T)

        # --- Block / token masking logic ---
        # mask can be:
        #   - None: no masking
        #   - (B, T): token keep mask (1 keep, 0 drop)
        #   - (B, T, T): attention mask (1 allow, 0 block)
        if mask is not None:
            if mask.dim() == 2:
                # token keep mask -> attention mask (block keys that are dropped)
                # allow attending only to kept tokens
                attn_mask = mask[:, None, :].expand(B, T, T)    # (B, T, T)
            elif mask.dim() == 3:
                attn_mask = mask
            else :
                raise ValueError(f"Unsupported mask shape : {mask.shape}")
            
            attn_mask = attn_mask.to(dtype=torch.bool)
            wei = wei.masked_fill(~attn_mask, float('-inf'))

        # wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)       # (B, T, hs)
        out = wei @ v           # (B, T, hs)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        with torch.profiler.record_function("Self-Attention"):
            out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class GRP(nn.Module):
    def __init__(self, cfg, mlp_ratio=4):
        super(GRP, self).__init__()
        self._cfg = cfg
        chars = cfg.dataset.chars_list
        cfg.vocab_size = len(chars)
        # TODO: 
        ## Provide the logic for the GRP network
        # --- basic sizes ---
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.max_block_size = cfg.max_block_size

        # action dimensions
        self.action_dim = len(cfg.env.action_mean)
        self.out_dim = self.action_dim * cfg.policy.action_stacking

        # patch embedding sizes (patch vector is p*p*3)
        patch_dim = cfg.patch_size * cfg.patch_size * 3
        self.obs_patch_proj = nn.Linear(patch_dim, cfg.n_embd)
        self.goal_patch_proj = nn.Linear(patch_dim, cfg.n_embd)

        # text embedding (non-T5)
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # optional projection for T5 embeddings -> n_embd
        # (LazyLinear figures out input dim from first forward)
        self.t5_proj = nn.LazyLinear(cfg.n_embd)

        # special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.pose_proj = nn.Sequential(
            nn.Linear(cfg.env.pose_dim, cfg.n_embd),
            nn.ReLU(),
            nn.Linear(cfg.n_embd, cfg.n_embd),
        ) if hasattr(cfg.env, "pose_dim") else None

        # --- positional embeddings ---
        # compute max sequence length
        H, W = cfg.image_shape[0], cfg.image_shape[1]
        n_patches = (H // cfg.patch_size) * (W // cfg.patch_size)
        obs_tokens = n_patches * cfg.policy.obs_stacking
        goal_img_tokens = n_patches
        text_tokens = cfg.max_block_size
        pose_tokens = 1  # if pose provided at runtime

        # We allocate assuming pose token exists; safe even if pose None
        self.max_seq_len = 1 + pose_tokens + obs_tokens + text_tokens + goal_img_tokens

        pe = calc_positional_embeddings(self.max_seq_len, cfg.n_embd)  # (L, D)
        self.pos_embedding = nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, L, D)

        self.drop = nn.Dropout(cfg.dropout)

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([
            Block(n_embd=cfg.n_embd, n_head=cfg.n_head, dropout=cfg.dropout)
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # 5) Classification MLPk
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.n_embd),
            nn.Linear(cfg.n_embd, self.out_dim),
        )

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, images, goals_txt, goal_imgs, targets=None, pose=None, mask_=False):
        n, c, h, w = images.shape
        obs_patches = get_patches_fast(images, self._cfg)
        patches_g = get_patches_fast(goal_imgs, self._cfg)
        if self._cfg.dataset.encode_with_t5:
            goals_e = goals_txt
            B, T, E = goals_txt.shape
        else:
            goals_e = self.token_embedding_table(goals_txt)
            B, E = goals_txt.shape
            T = self._cfg.max_block_size

        # TODO: 
        ## Provide the logic to produce the output and loss for the GRP
        
        # Map the vector corresponding to each patch to the hidden size dimension

        # Adding classification and goal_img tokens to the tokens

        # Adding positional embedding

        # Compute blocked masks

        # Transformer Blocks

        # Getting the classification token only

        # Compute output and loss
        return (out, loss)
    
    def resize_image(self, image):
        """
        Docstring for resize_image
        
        :param self: Description
        :param image: Description
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """
        Docstring for preprocess_state
        
        :param self: Description
        :param image: Description
        self._encode_state = lambda af:   ((af/(255.0)*2.0)-1.0) # encoder: take a float, output an integer
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        # img = _np.array(image, dtype=_np.float32)
        # img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        enc = ((image / 255.0) * 2.0) - 1.0
        # t = _torch.tensor(enc, dtype=_torch.float32, device=self._cfg.device)
        return enc
    
    def preprocess_state(self, image):
        img = self.resize_image(image)
        img = self.normalize_state(img)
        return img

    def preprocess_goal_image(self, image):
        return self.preprocess_state(image)

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            # TODO:    
            ## Provide the logic converting text goal to T5 embedding tensor
            pass
        else:
            pad = " " * self._cfg.max_block_size
            goal_ = goal[:self._cfg.max_block_size] + pad[len(goal):self._cfg.max_block_size]
            try:
                stoi = {c: i for i, c in enumerate(self._cfg.dataset.chars_list)}
                ids = [stoi.get(c, 0) for c in goal_]
            except Exception:
                ids = [0] * self._cfg.max_block_size
            return _torch.tensor(_np.expand_dims(_np.array(ids, dtype=_np.int64), axis=0), dtype=_torch.long, device=self._cfg.device)

    def process_text_embedding_for_buffer(self, goal, tokenizer=None, text_model=None):
        """
        Process text goal embedding for storing in the circular buffer.
        Returns a numpy array of shape (max_block_size, n_embd) without batch dimension.
        """
        import numpy as _np
        if tokenizer is None or text_model is None:
            raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
        
        goal_ = _np.zeros((self._cfg.max_block_size, self._cfg.n_embd), dtype=_np.float32)
        input_ids = tokenizer(goal, return_tensors="pt").input_ids
        goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy()
        goal_[:len(goal_t[0]), :] = goal_t[0][:self._cfg.max_block_size]
        return goal_

    def decode_action(self, action_tensor):
        
        """
        Docstring for decode_action
        
        :param self: Description
        :param action_tensor: Description
        self._decode_action = lambda binN: (binN * action_std) + action_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        ## The action tensor is of shape (batch_size, action_dim * action_stacking) so we need to repeat the mean and std per action stacking
        action_mean = _torch.tensor(np.repeat(self._cfg.env.action_mean, self._cfg.policy.action_stacking), dtype=action_tensor.dtype, device=action_tensor.device)
        action_std = _torch.tensor(np.repeat(self._cfg.env.action_std, self._cfg.policy.action_stacking), dtype=action_tensor.dtype, device=action_tensor.device)
        return (action_tensor * action_std) + action_mean
    
    def encode_action(self, action_float):
        """
        Docstring for encode_action
        
        :param self: Description
        :param action_float: Description
        self._encode_action = lambda af:   (af - action_mean)/(action_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        action_mean = _torch.tensor(self._cfg.env.action_mean, dtype=action_float.dtype, device=action_float.device)
        action_std = _torch.tensor(self._cfg.env.action_std, dtype=action_float.dtype, device=action_float.device)
        return (action_float - action_mean) / action_std


@torch.no_grad()
def estimate_loss(model, dataset):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_pose, x_goal, x_goal_img, Y = dataset.get_batch_grp(split, model._cfg, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y, pose=x_pose)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
