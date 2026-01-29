import torch
import torch.nn as nn
import torch.nn.functional as F
import math # 用于无穷大

class CMCM(nn.Module):
    def __init__(self, eeg_input_dim: int, bio_input_dim: int,seq_len: int, embed_dim: int, window_size: int, num_heads: int, num_classes: int,
                 dropout: float = 0, dtw_gamma: float = 1.0 ,eps: float = 1e-8):
        super(CMCM, self).__init__()

        self.intra_eeg = IntraModalityEnhance(seq_len,eeg_input_dim, embed_dim, num_heads, window_size, dropout)
        self.intra_pps = IntraModalityEnhance(seq_len,bio_input_dim, embed_dim, num_heads, window_size, dropout)

        self.inter = InterModalityCorrelation(embed_dim, num_heads, window_size, dropout)

        self.credibility = CredibilityFusion(embed_dim, window_size, dtw_gamma, eps)

        self.classifer = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, eeg, pps):
        eeg_enhanced = self.intra_eeg(eeg)
        pps_enhanced = self.intra_pps(pps)

        fused_eeg, fused_pps = self.inter(eeg_enhanced, pps_enhanced)
        final_eeg, final_pps = self.credibility(fused_eeg, fused_pps)

        combined_features = torch.cat([final_eeg, final_pps], dim=1)
        output = self.classifer(combined_features)

        return output

class CustomMultiheadAttention(nn.Module):
    def __init__(self, input_dim : int, num_heads : int, embed_dim : int, dropout : float):
        super(CustomMultiheadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = input_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        
        self.W_q_list = nn.ModuleList([nn.Linear(input_dim,self.head_dim) for _ in range(num_heads)])
        self.W_k_list = nn.ModuleList([nn.Linear(input_dim,self.head_dim) for _ in range(num_heads)]) 
        self.W_v_list = nn.ModuleList([nn.Linear(input_dim,self.head_dim) for _ in range(num_heads)])
        self.W_o = nn.Linear(num_heads*self.head_dim, embed_dim)


    def forward(self, q, k, v):
        batch_szie = q.shape[0]
        concat_attn = None
        for i in range(self.num_heads):
            q_i = self.W_q_list[i](q) # (batch, q_length, head_dim)
            k_i = self.W_k_list[i](k) # (batch, k_length, head_dim) 
            v_i = self.W_v_list[i](v) # (batch, k_length, head_dim)

            attn_score = torch.matmul(q_i, k_i.transpose(1, 2)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_score, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_out = torch.matmul(attn_weights, v_i)#(batch, q_length, head_dim)

            if concat_attn is None:
                concat_attn = attn_out
            else:
                concat_attn = torch.cat([concat_attn, attn_out], dim=-1)#(batch, q_length, num_heads*head_dim)

        multihead_out = self.W_o(concat_attn) # (batch, q_length, embed_dim)
        return multihead_out, attn_weights


def get_sliding_windows(x: torch.Tensor, window_size: int):
    """
    高效地从一个序列中创建滑动窗口。
    input: x :(batch, seq_length, feature_dim)
     window_size : 窗口大小即子序列长度
        
    ouput: 输入张量的滑动窗口视图，
        形状为 (batch, num_windows, feature_dim, window_size), 其中num_windows = seq_length
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number for symmetric padding.")
        
    padding = (window_size - 1) // 2
    
    # 1. 将序列长度维度换到最后，以便 F.pad 操作
    # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
    x = x.transpose(1, 2)
    
    # 2. 在序列长度维度上进行填充
    # x_padded shape: (batch, features, seq_len + 2*padding)
    x_padded = F.pad(x, (padding, padding), mode='constant', value = 0)
    
    # 3. 使用 unfold 创建窗口
    # unfold 在最后一个维度 (dimension=2) 上操作
    # windows shape: (batch, features, num_windows, window_size)
    windows = x_padded.unfold(dimension=2, size=window_size, step=1)
    
    # 4. 调整维度顺序以匹配期望的输出
    # (batch, features, num_windows, window_size) -> (batch, num_windows, features, window_size)
    return windows.permute(0, 2, 1, 3)

class IntraModalityEnhance(nn.Module):
    """
    模态内自注意力增强
    """
    def __init__(self, seq_len: int,input_dim: int, embed_dim: int, num_heads: int, window_size: int, dropout: float):
        super(IntraModalityEnhance, self).__init__()
        
        self.window_size = window_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 论文中描述的可学习的 token
        # 'a' token 用于特征压缩, 'b' token 用于位置编码
        self.feature_squeeze_token = nn.Parameter(torch.randn(1, seq_len, input_dim))
        self.position_token = nn.Parameter(torch.randn(1, seq_len, input_dim))
        
        self.attention =CustomMultiheadAttention(input_dim, num_heads, embed_dim, dropout)
        

    def forward(self, x: torch.Tensor):
        """
        x : (batch, seq_length, feature_dim)
         
        output: (batch, seq_length, embde_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 获取输入序列的滑动窗口
        # 形状: (batch, seq_length, feature_dim, window_size)
        windows = get_sliding_windows(x, self.window_size)
        
        output_features = []
        # 这个循环遍历每个时间步，以应用局部注意力
        for i in range(seq_len):
            # 获取当前时间步 `i` 的窗口
            window_i = windows[:, i, :, :].transpose(1, 2) # (batch, window_size, input_dim)
            
            # 获取当前时间步的可学习 token，并扩展以匹配批次大小
            a_i = self.feature_squeeze_token[:, i, :].unsqueeze(1).expand(batch_size, -1, -1) #(batch, 1, input_dim)
            b_i = self.position_token[:, i, :].unsqueeze(1).expand(batch_size, -1, -1)
            
            # 查询(Query)是特征压缩 token 'a_i'
            q = a_i
            
            # (batch, window_size + 2, feature_dim)
            kv_sequence = torch.cat([window_i, b_i, a_i], dim=1)
            
            # 应用自注意力
            #(batch, 1, embed_dim)
            attn_output, _ = self.attention(q, kv_sequence, kv_sequence)
            
            output_features.append(attn_output)
            
        # 拼接所有时间步的特征，形成最终的输出序列
        return torch.cat(output_features, dim=1)
    
class InterModalityCorrelation(nn.Module):
    '''
    跨模态相关性建模, 利用交叉注意力, 用于解决模态异质性问题
    '''
    def __init__(self,  embed_dim: int, num_heads: int, window_size: int, dropout:float):
        super(InterModalityCorrelation, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.eeg2pps_attention = CustomMultiheadAttention(embed_dim, num_heads, embed_dim, dropout)
        self.pps2eeg_attention = CustomMultiheadAttention(embed_dim, num_heads, embed_dim, dropout)
        
    def forward(self, eeg, pps):
        eeg_windows = get_sliding_windows(eeg, self.window_size)
        pps_windows = get_sliding_windows(pps, self.window_size)
        
        seq_len= eeg.shape[1]

        eeg_attn_list = []
        for i in range(seq_len):
            q_eeg = eeg[:, i, :].unsqueeze(1) #(batch, 1, embed_dim)
            kv_sequence = pps_windows[:, i, :, :].transpose(1, 2)#(batch, window_size, embed_dim)

            attn_output, _ = self.eeg2pps_attention(q_eeg, kv_sequence, kv_sequence)
            eeg_attn_list.append(attn_output)

        pps_attn_list = []
        for i in range(seq_len):
            q_pps = pps[:, i, :].unsqueeze(1)
            kv_sequence = eeg_windows[:, i, :, :].transpose(1, 2)
            attn_output, _ = self.pps2eeg_attention(q_pps, kv_sequence, kv_sequence)
            pps_attn_list.append(attn_output)

        fused_eeg = torch.cat(eeg_attn_list, dim=1)
        fused_pps = torch.cat(pps_attn_list, dim=1)

        return fused_eeg, fused_pps
    
class CredibilityFusion(nn.Module):
    '''
    使用softdtw计算序列模式一致性后得到加权特征
    '''
    def __init__(self, embed_dim:int, window_size:int, dtw_gamma: float =1.0, eps: float =1e-8):
        super(CredibilityFusion, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.gamma = dtw_gamma
        self.eps = eps

    def _soft_dtw(self, x, y):
            # x, y shape: (batch_size, seq_len_1, embed_dim)
            batch_size, n, m = x.shape
            device = x.device

            cost_matrix = torch.cdist(x, y, p=2)

            
            D = torch.full((batch_size, n + 1, n + 1), math.inf, device=device)
            D[:, 0, 0] = 0.

            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    prev_costs = torch.stack([
                        D[:, i-1, j],      
                        D[:, i, j-1],      
                        D[:, i-1, j-1]    
                    ], dim=1)
                    
                    soft_min_val = -self.gamma * torch.logsumexp(-prev_costs / self.gamma, dim=1)
                    
                    D[:, i, j] = cost_matrix[:, i-1, j-1] + soft_min_val
                    
            return D[:, n, n]

    def forward(self, fused_eeg, fuse_pps):
        batch_size, seq_len, embed_dim = fused_eeg.shape  

        me_windows = get_sliding_windows(fused_eeg, self.window_size).transpose(2,3)
        mp_windows = get_sliding_windows(fuse_pps, self.window_size).transpose(2,3)   

        me_windows = me_windows.reshape(-1, self.window_size, embed_dim)
        mp_windows = mp_windows.reshape(-1, self.window_size, embed_dim)

        w1 = self._soft_dtw(me_windows, mp_windows)       
        w1 = w1.reshape(batch_size, seq_len)

        w1_min = w1.min(dim = 1, keepdim=True)[0]
        w1_max = w1.max(dim = 1, keepdim=True)[0]

        w2 = 1 - (w1 - w1_min)/(w1_max - w1_min + self.eps)

        w2_sum = w2.sum(dim = 1, keepdim=True)
        w3 = w2/w2_sum
        w3 = w3.unsqueeze(-1) #(batch, seq_len, 1)

        final_eeg = torch.sum(fused_eeg * w3, dim = 1)#(batch, embed_dim)
        final_pps = torch.sum(fuse_pps * w3, dim = 1)

        return final_eeg, final_pps
