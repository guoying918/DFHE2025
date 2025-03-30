import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
import numpy as np

SHOT_NUM_PER_CLASS = 180

class CrossAttention_S(nn.Module):
    def __init__(self, embed_dim, num_heads, class_num):
        super(CrossAttention_S, self).__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim * num_heads
        self.class_num = class_num
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        query = query.reshape(self.class_num, self.embed_dim, -1, 1)
        key = key.reshape(self.class_num, self.embed_dim, -1, 1) 
        value = value.reshape(self.class_num, self.embed_dim, -1, 1)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.reshape(-1, attn_output.size(1))
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output

class CrossAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Queries (N, C)
        Keys (N, C)
        Values (N, C)
        '''
        b_q, c = queries.shape
        b_s, c = keys.shape

        q = self.fc_q(queries).view(b_q, self.h, self.d_k, 1).permute(0, 1, 3, 2) 
        k = self.fc_k(keys).view(1, self.h, self.d_k, b_s) 
        v = self.fc_v(values).view(1, self.h, b_s,self.d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 1, 3, 2).contiguous().view(b_q, self.h * self.d_v, 1)  # (N, h*d_v, 1)
        out = self.fc_o(out.squeeze(-1))  # Restore shape to (N, C)
        return out

class Project_Linear(nn.Module):
    def __init__(self, embed_dim):
        super(Project_Linear, self).__init__()
        self.linear = nn.Linear(embed_dim, embed_dim * 3)

    def forward(self, x):
        out = self.linear(x)
        q, k, v = torch.chunk(out, 3, dim=-1)
        return q, k, v

##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x): 
        return self.body(x)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_dim = int(dim*ffn_expansion_factor)
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, head, ffn_expansion_factor, bias, LayerNorm_type, class_num):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention_S(dim, head, class_num)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x_q, x_k, x_v, domain):

        x = self.norm1(x_q + self.attn(x_q, x_k, x_v))
        x = self.norm2(x + self.ffn(x))
        return x

class TransformerBlock_1(nn.Module):
    def __init__(self, dim, head, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_1, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(d_model=dim, d_k=dim, d_v=dim, h=head)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        
    def forward(self, x_q, x_k, x_v, domain):

        x = self.norm1(x_q + self.attn(x_q, x_k, x_v))
        
        x = self.norm2(x + self.ffn(x))
        return x

class S_enhance(nn.Module):
    def __init__(self, embed_dim, num_heads, class_num):
        super(S_enhance, self).__init__()
        self.embed_dim = embed_dim
        self.class_num = class_num
        self.proj_linears = Project_Linear(embed_dim)
        self.cross_attn_model = TransformerBlock(embed_dim, num_heads, 2, False,'BiasFree_LayerNormb', self.class_num)

    def forward(self, support_features, support_labels, domain):
        q, k, v = self.proj_linears(support_features)
        
        attn_output = self.cross_attn_model(q, k, v, domain) # (45,128)
        support_per = attn_output.reshape(self.class_num, -1, self.embed_dim)
        protype_support = support_per.mean(1)

        return protype_support, support_per, attn_output

class Network_SQHE(nn.Module):
    def __init__(self, class_num, embed_dim = 128, num_heads = 4,
                 lambda_mi=0.1, lambda_inter=0.01):
        super(Network_SQHE, self).__init__()
        self.s_enhance = S_enhance(embed_dim, num_heads, class_num)
        self.supp_query_enhance = TransformerBlock_1(embed_dim, num_heads, 2, False,'BiasFree_LayerNormb')
        
        # loss weights
        self.lambda_mi = lambda_mi
        self.lambda_proto_support = lambda_inter

    def forward(self, support_features, support_labels, query_features, query_labels, domain):
        support_proto, support_per, support_total = self.s_enhance(support_features, support_labels, domain)
        
        query_q = query_features
        key_q = support_proto
        value_q = support_proto
        query_Enhance = self.supp_query_enhance(query_q, key_q, value_q, domain) # (N, C)

        if query_labels is not None:
            ocps_loss = self.ICCL(query_Enhance, support_per, query_labels)
            inter_class_loss = self.ICOL(support_per, support_proto, support_labels)
            total_loss = (
                self.lambda_mi * ocps_loss +
                self.lambda_proto_support * inter_class_loss
            )
            return support_proto, query_Enhance, support_total, total_loss
        else:
            return support_proto, query_Enhance, support_total, None
        
    # intra-class consistency loss,  encourages features within the same class to be more compact, enhancing intra-class coherence
    def ICCL(self, query_Enhance, support_pre, query_labels):
        """
        query_Enhance: (171,128)
        support_pre: (9,5,128)
        """ 
        support_pre = support_pre.permute(0,2,1) 
        I = torch.eye(support_pre.size(1)).unsqueeze(0).repeat(support_pre.size(0), 1, 1).cuda()

        S = support_pre 
        S_t = S.transpose(2, 1) 
        S1 = S_t @ S 
        S1_I = torch.linalg.inv(S1)  
        S_star = S1_I @ S_t  
        P = I - (S @ S_star) + 1e-5 

        R = query_Enhance.float()  
        R = R.unsqueeze(-1)

        intra_domain_loss_list = []

        for i in range(R.size(0)): 
            label = query_labels[i]
            loss = torch.norm(P[label] @ R[i])
            intra_domain_loss_list.append(loss)

        loss = torch.mean(torch.stack(intra_domain_loss_list))
        return loss
    # inter-class orthogonality loss, promotes orthogonality between different class prototypes, ensuring greater separability
    def ICOL(self, support_per, support_proto, support_labels):
        """
        support_per: (9,5,128)
        support_proto: (9,128)
        """
        support_per = support_per.permute(0,2,1)
        I = torch.eye(support_per.size(1)).unsqueeze(0).repeat(support_per.size(0), 1, 1).cuda()

        S = support_per
        S_t = S.transpose(2, 1)
        S1 = S_t @ S
        S1_I = torch.linalg.inv(S1) 
        S_star = S1_I @ S_t
        P = I - (S @ S_star) + 1e-5 

        R = support_proto.float()
        R = R.unsqueeze(-1)
        inter_domain_loss_list = []
        unique_labels = torch.unique(support_labels)

        for i in range(R.size(0)): 
            for j in range(P.size(0)): 
                if unique_labels[j] != i: 
                    loss = torch.norm(P[unique_labels[j]] @ R[i]) 
                    inter_domain_loss_list.append(1.0 / (loss + 1e-5))
        if inter_domain_loss_list: 
            loss = max(inter_domain_loss_list)
        return loss