import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, kq_dim, emb_dim,
                 qk_norm = True, qkv_bias=False, attention_drop=0.0,
                 value_proj_drop=0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.kq_dim = kq_dim
        self.emb_dim = emb_dim
        
        self.norm_const = torch.sqrt(torch.tensor([kq_dim])).to(device)
        
        # can concatenate the three matrices to get a single matrix
        self.query = nn.Linear(emb_dim, n_heads*kq_dim, bias=qkv_bias)
        self.key = nn.Linear(emb_dim, n_heads*kq_dim, bias=qkv_bias)
        self.value_down = nn.Linear(emb_dim, n_heads*kq_dim, bias=qkv_bias)
        
        self.value_up = nn.Linear(n_heads*kq_dim, emb_dim)
        
        self.q_norm = nn.LayerNorm(kq_dim) if qk_norm else nn.Identity
        self.k_norm = nn.LayerNorm(kq_dim) if qk_norm else nn.Identity
        
        self.attention_drop = nn.Dropout(attention_drop)
        self.value_proj_drop = nn.Dropout(value_proj_drop)
        
    def forward(self, emb):
        # emb: batch_size x seq_len x hid_dim
        batch_size, seq_len = emb.shape[:-1]
        
        q = self.query(emb).reshape(batch_size, seq_len, self.n_heads, self.kq_dim)
        k = self.key(emb).reshape(batch_size, seq_len, self.n_heads, self.kq_dim)
        v = self.value_down(emb).reshape(batch_size, seq_len, self.n_heads, self.kq_dim)
        
        q = q.permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len, kq_dim)
        k = k.permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len, kq_dim)
        v = v.permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len, kq_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        attention = torch.matmul(q, k.transpose(-2, -1))/(self.norm_const)
        attention = attention.softmax(dim=-1) # (batch_size, n_heads, seq_len, seq_len)
        attention = self.attention_drop(attention)
        
        value = torch.matmul(attention, v) # (batch_size, n_heads, seq_len, kq_dim)
        value = value.transpose(1, 2).reshape(batch_size, seq_len, self.n_heads*self.kq_dim)
        value = self.value_up(value)
        emb = self.value_proj_drop(value) + emb
        return emb

          
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_heads, kq_dim, emb_dim, context_dim,
                 qk_norm = True, qkv_bias=False, attention_drop=0.0,
                 value_proj_drop=0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.kq_dim = kq_dim
        self.emb_dim = emb_dim
        self.context_dim = context_dim
        
        self.norm_const = torch.sqrt(torch.tensor([kq_dim])).to(device)
        
        # can concatenate the three matrices to get a single matrix
        self.query = nn.Linear(emb_dim, n_heads*kq_dim, bias=qkv_bias)
        self.key = nn.Linear(context_dim, n_heads*kq_dim, bias=qkv_bias)
        self.value_down = nn.Linear(context_dim, n_heads*kq_dim, bias=qkv_bias)
        
        self.value_up = nn.Linear(n_heads*kq_dim, emb_dim)
        
        self.q_norm = nn.LayerNorm(kq_dim) if qk_norm else nn.Identity
        self.k_norm = nn.LayerNorm(kq_dim) if qk_norm else nn.Identity
        
        self.attention_drop = nn.Dropout(attention_drop)
        self.value_proj_drop = nn.Dropout(value_proj_drop)
        
    def forward(self, emb, context):
        # emb: batch_size x seq_len x emb_dim
        batch_size, emb_seq_len = emb.shape[:-1]
        _, context_seq_len = context.shape[:-1]
        
        q = self.query(emb).reshape(batch_size, emb_seq_len, self.n_heads, self.kq_dim)
        k = self.key(context).reshape(batch_size, context_seq_len, self.n_heads, self.kq_dim)
        v = self.value_down(context).reshape(batch_size, context_seq_len, self.n_heads, self.kq_dim)
        
        q = q.permute(0, 2, 1, 3) # (batch_size, n_heads, emb_seq_len, kq_dim)
        k = k.permute(0, 2, 1, 3) # (batch_size, n_heads, context_seq_len, kq_dim)
        v = v.permute(0, 2, 1, 3) # (batch_size, n_heads, context_seq_len, kq_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        attention = torch.matmul(q, k.transpose(-2, -1))/(self.norm_const)
        attention = attention.softmax(dim=-1) # (batch_size, n_heads, emb_seq_len, context_seq_len)
        attention = self.attention_drop(attention)
        
        value = torch.matmul(attention, v) # (batch_size, n_heads, emb_seq_len, kq_dim)
        value = value.transpose(1, 2).reshape(batch_size, emb_seq_len, self.n_heads*self.kq_dim)
        value = self.value_up(value)
        value = self.value_proj_drop(value)
        return value # same dim as emb


class MLP(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim,
                 act_fn, drop_prob=0.0) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(inp_dim, hid_dim)
        self.act_fn = act_fn()
        self.layer_2 = nn.Linear(hid_dim, out_dim)
        self.drop_layer = nn.Dropout(drop_prob)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.act_fn(x)
        x = self.drop_layer(x)
        x = self.layer_2(x)
        x = self.drop_layer(x)
        return x
          

class AttentionBlock(nn.Module):
    def __init__(self, n_heads, emb_dim, kq_dim,
                 mlp_hid_dim, mlp_act_fn, layer_norm = True,
                 qk_norm = True, qkv_bias=False, attention_drop=0.0,
                 value_proj_drop=0.0, mlp_drop = 0.0) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads, kq_dim, emb_dim, qk_norm,
                                                       qkv_bias, attention_drop, value_proj_drop)
        self.mlp = MLP(emb_dim, mlp_hid_dim, emb_dim, mlp_act_fn, mlp_drop)
        self.norm_layer_1 = nn.LayerNorm(emb_dim) if layer_norm else nn.Identity(emb_dim)
        self.norm_layer_2 = nn.LayerNorm(emb_dim) if layer_norm else nn.Identity(emb_dim)
        
    def forward(self, emb):
        # Pre-Layer Normalization
        emb = self.multi_head_attention(emb) + emb
        emb = self.norm_layer_1(emb)
        
        emb = self.mlp(emb) + emb
        emb = self.norm_layer_2(emb)
        return emb # same dimension as emb
        
        
class CrossAttentionBlock(nn.Module):
    def __init__(self, n_heads, emb_dim, kq_dim, context_dim,
                 mlp_hid_dim, mlp_act_fn, layer_norm = True,
                 qk_norm = True, qkv_bias=False, attention_drop=0.0,
                 value_proj_drop=0.0, mlp_drop = 0.0) -> None:
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(n_heads, kq_dim, emb_dim)
        self.multi_head_cross_attn = MultiHeadCrossAttention(n_heads, kq_dim, emb_dim,
                                                             context_dim, qk_norm, qkv_bias,
                                                             attention_drop, value_proj_drop)
        self.mlp = MLP(emb_dim, mlp_hid_dim, emb_dim, mlp_act_fn, mlp_drop)
        
        self.norm_layer_1 = nn.LayerNorm(emb_dim) if layer_norm else nn.Identity(emb_dim)
        self.norm_layer_2 = nn.LayerNorm(emb_dim) if layer_norm else nn.Identity(emb_dim)
        self.norm_layer_3 = nn.LayerNorm(emb_dim) if layer_norm else nn.Identity(emb_dim)
    
    def forward(self, emb, context):
        emb = emb + self.multi_head_attn(emb)
        emb = self.norm_layer_1(emb)
        
        emb = emb + self.multi_head_cross_attn(emb, context)
        emb = self.norm_layer_2(emb)
        
        emb = self.mlp(emb)
        emb = self.norm_layer_3(emb)
        
        return emb
        

class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self):
        pass
 
    
class ExampleTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.x_dim = config.x_dim
        self.y_dim = config.y_dim
        
        self.t_dim = 1
        
        self.emb_dim = config.emb_dim
        
        self.y_drop_prob = config.diff_drop_prob
        
        # x_shape: B x  x_dim_1 x x_dim_2
        self.x_emb = nn.Linear(self.x_dim, self.emb_dim)
        self.x_pos_emb = nn.Parameter(torch.randn(self.x_dim,
                                                  self.emb_dim))    
        
        # y_shape: B x y_dim_1 x y_dim_2
        self.y_emb = nn.Linear(self.y_dim, self.emb_dim)
        self.y_pos_emb = nn.Parameter(torch.randn(self.y_dim,
                                                  self.emb_dim))
        
        # t: B x 1 
        self.t_emb = nn.Linear(self.t_dim, self.emb_dim)
        
        temp_list = []
        for _ in range(config.n_x_attn):
            temp_list.append(AttentionBlock(config.n_heads, config.emb_dim,
                                            config.kq_dim, config.mlp_hid_dim,
                                            config.mlp_act_fn))
        self.x_attn = nn.ModuleList(temp_list)
        
        temp_list = []
        for _ in range(config.n_y_attn):
            temp_list.append(AttentionBlock(config.n_heads, config.emb_dim,
                                            config.kq_dim, config.mlp_hid_dim,
                                            config.mlp_act_fn))
        self.y_attn = nn.ModuleList(temp_list)
        
        temp_list = []
        for _ in range(config.n_xyt_attn):
            temp_list.append(CrossAttentionBlock(config.n_heads, config.emb_dim,
                                                 config.kq_dim, config.emb_dim, config.mlp_hid_dim,
                                                 config.mlp_act_fn))
        self.xy_cross_attn = nn.ModuleList(temp_list)
        
        temp_list = []
        for _ in range(config.n_xyt_attn):
            temp_list.append(CrossAttentionBlock(config.n_heads, config.emb_dim,
                                                 config.kq_dim, config.emb_dim, config.mlp_hid_dim,
                                                 config.mlp_act_fn))
        self.xt_cross_attn = nn.ModuleList(temp_list)
        
        self.out_layer = nn.Linear(config.emb_dim, config.x_dim)
    
    def forward(self, x, y, t, mode):
        batch_size = x.shape[0]
        
        x = x.reshape(batch_size, self.x_dim)
        x = self.x_emb(x) + self.x_pos_emb
        
        if y is None:
            y = torch.zeros((batch_size, self.emb_dim))
        else:
            y = self.y_emb(y)

        if mode=="train":
            y =y*(torch.bernoulli(torch.ones(batch_size, 1, 1)-self.y_drop_prob).to(device))
        y = y + self.y_pos_emb
        
        t = self.t_emb(t[:, None])
        
        for x_attn in self.x_attn:
            x = x_attn(x)

        for y_attn in self.y_attn:
            y = y_attn(y)
        
        for i in range(len(self.xy_cross_attn)):
            x = self.xt_cross_attn[i](x, t)
            x = self.xy_cross_attn[i](x, y)
            
        x = self.out_layer(x)
        x = x.reshape(batch_size, self.x_dim)
        return x
    
    