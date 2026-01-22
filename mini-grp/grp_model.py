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
        # IMPORTANT: do NOT use LazyLinear here because __init__ calls self.apply(self._init_weights)
        if cfg.dataset.encode_with_t5:
            # infer T5 hidden size from version string (covers common cases)
            t5v = str(cfg.dataset.t5_version).lower()
            if "small" in t5v:
                t5_hidden = 512
            elif "base" in t5v:
                t5_hidden = 768
            elif "large" in t5v:
                t5_hidden = 1024
            else:
                # safest default for most assignments here (t5-small)
                t5_hidden = 512
            self.t5_proj = nn.Linear(t5_hidden, cfg.n_embd)
        else:
            self.t5_proj = None

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
            for _ in range(cfg.n_blocks)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # 5) Classification MLP
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
        # n, c, h, w = images.shape
        # obs_patches = get_patches_fast(images, self._cfg)
        # patches_g = get_patches_fast(goal_imgs, self._cfg)
        # if self._cfg.dataset.encode_with_t5:
        #     goals_e = goals_txt
        #     B, T, E = goals_txt.shape
        # else:
        #     goals_e = self.token_embedding_table(goals_txt)
        #     B, E = goals_txt.shape
        #     T = self._cfg.max_block_size

        # TODO: 
        ## Provide the logic to produce the output and loss for the GRP
        
        # Map the vector corresponding to each patch to the hidden size dimension

        # Adding classification and goal_img tokens to the tokens

        # Adding positional embedding

        # Compute blocked masks

        # Transformer Blocks

        # Getting the classification token only

        # Compute output and loss
        """
        images:    usually (B, C, H, W) from the buffer/trainer
        goal_imgs: usually (B, C, H, W)
        goals_txt: either
            - (B, T) token ids (non-T5), OR
            - (B, T, E_t5) embeddings (T5 mode)
        targets:   (B, action_dim * action_stacking) for continuous baseline
        pose:      optional (B, pose_dim)
        mask_:     whether to apply block masking (drop text or goal image tokens)
        """

        # ----------------------------
        # 1) Ensure images are NHWC for get_patches_fast()
        # ----------------------------
        # get_patches_fast expects (B, H, W, C)
        if images.dim() == 4 and images.shape[1] in (3, 3 * self._cfg.policy.obs_stacking):
            images_nhwc = images.permute(0, 2, 3, 1).contiguous()   # NCHW -> NHWC
        else:
            images_nhwc = images

        if goal_imgs.dim() == 4 and goal_imgs.shape[1] == 3:
            goal_imgs_nhwc = goal_imgs.permute(0, 2, 3, 1).contiguous()
        else:
            goal_imgs_nhwc = goal_imgs

        B = images_nhwc.shape[0]

        # ----------------------------
        # 2) Patchify + project patches to n_embd
        # ----------------------------
        obs_patches = get_patches_fast(images_nhwc, self._cfg)        # (B, Nobs, patch_dim)
        goal_patches = get_patches_fast(goal_imgs_nhwc, self._cfg)    # (B, Ngoal, patch_dim)

        obs_tok = self.obs_patch_proj(obs_patches)                    # (B, Nobs, n_embd)
        goal_img_tok = self.goal_patch_proj(goal_patches)             # (B, Ngoal, n_embd)

        # ----------------------------
        # 3) Text tokens
        # ----------------------------
        if self._cfg.dataset.encode_with_t5:
            # goals_txt expected (B, T, E_t5)
            if goals_txt.dim() != 3:
                raise ValueError(f"Expected T5 embeddings (B,T,E), got {goals_txt.shape}")
            text_tok = self.t5_proj(goals_txt)                        # (B, T, n_embd)
            T_text = text_tok.shape[1]
            # pad/truncate to max_block_size
            if T_text < self._cfg.max_block_size:
                pad_len = self._cfg.max_block_size - T_text
                pad = torch.zeros(B, pad_len, self._cfg.n_embd, device=text_tok.device, dtype=text_tok.dtype)
                text_tok = torch.cat([text_tok, pad], dim=1)
            else:
                text_tok = text_tok[:, :self._cfg.max_block_size, :]
        else:
            # goals_txt expected (B, T) token ids
            if goals_txt.dim() != 2:
                raise ValueError(f"Expected token ids (B,T), got {goals_txt.shape}")
            text_tok = self.token_embedding_table(goals_txt)          # (B, T, n_embd)
            # ensure fixed length
            if text_tok.shape[1] != self._cfg.max_block_size:
                text_tok = text_tok[:, :self._cfg.max_block_size, :]

        # ----------------------------
        # 4) Special tokens: CLS (+ optional pose)
        # ----------------------------
        cls = self.cls_token.expand(B, 1, self._cfg.n_embd)           # (B, 1, n_embd)

        pose_tok = None
        if pose is not None and hasattr(self, "pose_proj") and self.pose_proj is not None:
            pose_tok = self.pose_proj(pose).unsqueeze(1)             # (B, 1, n_embd)

        # ----------------------------
        # 5) Concatenate into one sequence
        #    [CLS] + [POSE?] + OBS + TEXT + GOAL_IMG
        # ----------------------------
        parts = [cls]
        if pose_tok is not None:
            parts.append(pose_tok)
        parts.extend([obs_tok, text_tok, goal_img_tok])

        x = torch.cat(parts, dim=1)                                  # (B, Ttot, n_embd)
        Ttot = x.shape[1]

        # ----------------------------
        # 6) Positional embedding + dropout
        # ----------------------------
        if Ttot > self.pos_embedding.shape[1]:
            raise ValueError(f"Sequence too long: {Ttot} > max_seq_len {self.pos_embedding.shape[1]}")

        x = x + self.pos_embedding[:, :Ttot, :].to(x.device)
        x = self.drop(x)

        # ----------------------------
        # 7) Block masking (optional)
        #    Randomly drop TEXT tokens or GOAL_IMG tokens (per sample)
        # ----------------------------
        attn_mask = None
        if mask_:
            keep = torch.ones(B, Ttot, device=x.device, dtype=torch.bool)

            idx = 0
            idx += 1  # CLS

            if pose_tok is not None:
                idx += 1  # POSE

            n_obs = obs_tok.shape[1]
            obs_start, obs_end = idx, idx + n_obs
            idx = obs_end

            text_start, text_end = idx, idx + self._cfg.max_block_size
            idx = text_end

            n_goal = goal_img_tok.shape[1]
            goal_start, goal_end = idx, idx + n_goal

            # 0 -> drop text, 1 -> drop goal image
            drop_choice = torch.randint(0, 2, (B,), device=x.device)

            for b in range(B):
                if drop_choice[b].item() == 0:
                    keep[b, text_start:text_end] = False
                    x[b, text_start:text_end, :] = 0.0
                else:
                    keep[b, goal_start:goal_end] = False
                    x[b, goal_start:goal_end, :] = 0.0

            # attention mask: allow attention only between kept tokens
            attn_mask = keep[:, :, None] & keep[:, None, :]          # (B, T, T)

        # ----------------------------
        # 8) Transformer encoder blocks
        # ----------------------------
        for blk in self.blocks:
            x = blk(x, mask=attn_mask)
        x = self.ln_f(x)

        # ----------------------------
        # 9) Prediction head (use CLS token)
        # ----------------------------
        cls_out = x[:, 0, :]                                         # (B, n_embd)
        out = self.head(cls_out)                                     # (B, action_dim * action_stacking)

        # ----------------------------
        # 10) Loss (continuous baseline)
        # ----------------------------
        loss = None
        if targets is not None:
            loss = F.mse_loss(out, targets)
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
            # tokenize with padding/truncation
            tok = tokenizer(
                goal,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._cfg.max_block_size,
            )
            tok = {k: v.to(self._cfg.device) for k, v in tok.items()}

            with _torch.no_grad():
                enc = text_model.encoder(
                    input_ids=tok["input_ids"],
                    attention_mask=tok.get("attention_mask", None),
                ).last_hidden_state  # (1, T, E_t5)

            # Ensure exactly max_block_size
            if enc.shape[1] < self._cfg.max_block_size:
                pad_len = self._cfg.max_block_size - enc.shape[1]
                pad = _torch.zeros(1, pad_len, enc.shape[2], device=enc.device, dtype=enc.dtype)
                enc = _torch.cat([enc, pad], dim=1)
            else:
                enc = enc[:, :self._cfg.max_block_size, :]

            return enc
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
