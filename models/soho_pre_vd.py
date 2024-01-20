import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def ema_tensor_inplace(moving_avg, new, decay):
    new_out = torch.mul(new,1.0-decay)
    moving_avg.data.mul_(decay).add_(new_out.detach())

def sum_inplace(sum_data,new):
    sum_data.data.add_(new)

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def laplace_smoothing_dim(x, n_categories,dim=1, eps=1e-5):
    return (x + eps) / (x.sum(dim=dim,keepdim=True) + n_categories * eps)

class SOHO_Pre_VD(nn.Module):
    def __init__(self,num_tokens,token_dim,decay=0.1,max_decay=1.0,eps=1e-5):
        super(SOHO_Pre_VD, self).__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        embed = torch.randn(num_tokens, token_dim)
        self.register_buffer('embed', embed)
        nn.init.normal_(self.embed)
        self.register_buffer('cluster_size', torch.zeros(num_tokens))
        self.register_buffer('cluster_sum', torch.zeros(num_tokens))
        self.register_buffer('embed_avg', torch.zeros(num_tokens,token_dim))

        self.decay = decay
        self.eps = eps
        self.curr_decay=self.decay
        self.max_decay=max_decay
        self.step=0

    def set_decay_updates(self,num_update):
        # self.curr_decay=min(self.decay*num_update,self.max_decay)
        num_update = math.ceil(num_update * 0.9)
        self.curr_decay = max(self.max_decay / num_update * self.step, self.decay)
        self.curr_decay = min(self.curr_decay, 1.0)

    def forward(self,inputs_flatten, num_update=None):
        distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embed.data ** 2, dim=1)
                     - 2 * torch.matmul(inputs_flatten, self.embed.data.t()))

        """
                encoding_indices: Tensor containing the discrete encoding indices, ie
                which element of the quantized space each input element was mapped to.
        """
        k = 4
        topk_values, topk_indices = torch.topk(distances, k, largest=False)
        topk_values = topk_values[:,1:]
        topk_values = F.normalize(topk_values, dim=1)
        # 将距离倒置当作概率分布
        topk_values = 1.0 / (topk_values + 1e-4)
        topk_values = F.softmax(topk_values, dim=1)
        topk_indices = topk_indices[:,1:]

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0],self.num_tokens, dtype=torch.float,device=inputs_flatten.device)
        encodings.scatter_(1, encoding_indices, 1)  # 将 encodings 中 encoding_indices 对应位置置为 1，相当于独热编码

        # if self.training:
        #     self.step += 1
        #     self.set_decay_updates(num_update)
        #     # print(self.curr_decay)

        #     # 统计码本中每条数据被选中的次数
        #     tmp_sum = torch.sum(encodings,dim=0,keepdim=True)
        #     encoding_sum = torch.sum(concat_all_gather(tmp_sum), dim=0)

        #     sum_inplace(self.cluster_sum,encoding_sum)  # 加和，这里的self.cluster_sum不进行加权
        #     ema_tensor_inplace(self.cluster_size, encoding_sum, self.curr_decay)    # 对码本数量进行动量更新，这里self.cluster_size保留了所有数据
        #     embed_sum_tmp = torch.matmul(encodings.t(), inputs_flatten) # 选中输入数据对应码本数据再对应输入数据的值,对应同一码本的向量相加

        #     embed_sum = torch.sum(concat_all_gather(embed_sum_tmp.unsqueeze(dim=0)),dim=0)
        #     ema_tensor_inplace(self.embed_avg, embed_sum, self.curr_decay)  # 对码本条目进行动量更新，这里self.embed_avg保留了所有数据
        #     # 平滑处理后，每个位置都有了值，原来为0的位置也有了较小值
        #     cluster_size = laplace_smoothing(self.cluster_size, self.num_tokens, self.eps) * self.cluster_size.sum()
        #     embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)

        #     if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        #         world_size = dist.get_world_size()
        #         dist.all_reduce(embed_normalized.div_(world_size))
        #     self.embed.data.copy_(embed_normalized)


        quantize = torch.matmul(encodings, self.embed)
        #quantize = inputs_flatten
        quantize = (quantize - inputs_flatten).detach() + inputs_flatten

        return quantize, encoding_indices, topk_values, topk_indices


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        tensors_gather = [
            torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
    else:
        output = tensor.clone()
    return output