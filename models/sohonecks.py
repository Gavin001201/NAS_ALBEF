import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import math
import numpy as np
# from  ..utils import ConvModule,build_norm_layer
from .soho_pre_vd import SOHO_Pre_VD


class SimpleVDforPre(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_tokens,
                 decay=0.4,
                 max_decay=0.99,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 mask_prob=0.015,
                 begin_align=False,
                 pos_align=True
                 ):
        super(SimpleVDforPre, self).__init__()
        self.linear = nn.Sequential(
                    nn.Linear(in_channels,out_channels),
                    nn.LayerNorm(out_channels),
                    nn.ReLU())
        self.ln = nn.LayerNorm(out_channels)
        self.mask_emb = nn.Embedding(1, out_channels)
        self.vq = SOHO_Pre_VD(num_tokens, out_channels, decay=decay,max_decay=max_decay)
        self.num_tokens = num_tokens
        self.out_channels = out_channels
        self.mask_prob = mask_prob
        self.total_num=0
        self.pos_align = pos_align
        self.begin_align = begin_align

        if begin_align:
            self.begin_line = nn.Linear(out_channels,out_channels)

        if pos_align:
            self.pos_line = nn.Linear(out_channels,out_channels)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, )

    def get_vis_mask(self, b, device, img_meta):
        h = max([meta['pad_shape'][0] for meta in img_meta])
        w = max([meta['pad_shape'][0] for meta in img_meta])
        mask = torch.zeros((b, 1, h, w), dtype=torch.float32, device=device)
        groups = b // len(img_meta)
        for i, meta in enumerate(img_meta):
            imh, imw, _ = meta['img_shape']
            mask[i * groups:(i + 1) * groups, 0, :imh, :imw] = 1
        return mask

    def position_encoding_sine(self, mask):
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        pos_feat_dim = self.out_channels // 2

        dim_t = torch.arange(pos_feat_dim, dtype=torch.float32, device=mask.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / pos_feat_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, img, img_meta):
        img = img[-1]
        x = F.max_pool2d(img, 2, stride=2)
        xq = self.conv(x)
        xq_img = xq

        batch_size, c, h, w = xq.size()
        b=batch_size
        inputs = xq.permute(0, 2, 3, 1).contiguous()
        inputs_flatten = inputs.view(batch_size * h * w, c)

        if self.begin_align:
            inputs_flatten=self.begin_line(inputs_flatten)

        quantized_pt, indices = self.vq(inputs_flatten)
        if self.pos_align:
            quantized_pt = self.pos_line(quantized_pt)

        embedded_pt = quantized_pt.view(b, w * h, quantized_pt.size(-1))
        embedded_pt = embedded_pt.permute(0,2,1).view(b,-1,h,w)

        embedded_pt = embedded_pt+xq_img

        visual_mask = self.get_vis_mask(batch_size, img.device, img_meta).float()
        visual_mask = F.interpolate(visual_mask, size=xq.shape[-2:]).to(dtype=torch.bool)
        pos = self.position_encoding_sine(visual_mask[:, 0, :, :])
        visual_mask = visual_mask.to(dtype=torch.float32).view(batch_size, 1, h, w)

        indices = indices.view(batch_size, 1, h, w).float()
        indices = indices * visual_mask - 100 * (1 - visual_mask)

        tmp = np.random.randint(h * w)
        tmp_label = indices[:, :, tmp // w, tmp % w].view(batch_size, 1, 1, 1)
        masked_indices = (indices == tmp_label).float()
        masked_indices = masked_indices * visual_mask

        probability_matrix = torch.full(tmp_label.shape, self.mask_prob)
        masked_indices2 = torch.bernoulli(probability_matrix).to(device=img.device).float()
        masked_indices = masked_indices * masked_indices2

        # mask_emb = torch.zeros_like(embedded_pt).to(device=xq.device).float()
        mask_emb = self.mask_emb.weight.view(1, self.out_channels, 1, 1)
        embedded_pt = embedded_pt * (1 - masked_indices) + mask_emb * masked_indices
        embedded_pt += pos

        xq = self.ln(embedded_pt.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        labels = indices * masked_indices - 100 * (1 - masked_indices)
        labels = labels.long().view(batch_size, -1)

        xq = xq.view(xq.size(0), xq.size(1), -1).contiguous()
        xq = xq.transpose(1, 2)

        visual_mask = visual_mask.view(batch_size, -1).long()



        return xq, visual_mask,labels


class SimpleVDforPreGate(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_tokens,
                 decay=0.4,
                 max_decay=0.99,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 mask_prob=0.015,
                 begin_align=False,
                 pos_align=True
                 ):
        super(SimpleVDforPreGate, self).__init__()
        self.linear = nn.Sequential(
                    nn.Linear(in_channels,out_channels),
                    nn.LayerNorm(out_channels),
                    nn.ReLU())
        self.ln = nn.LayerNorm(out_channels)
        self.vq = SOHO_Pre_VD(num_tokens, out_channels, decay=decay,max_decay=max_decay)
        self.num_tokens = num_tokens
        self.out_channels = out_channels
        self.total_num=0
        self.pos_align = pos_align
        self.begin_align = begin_align
        self.gate_fc = nn.Sequential(
                    nn.Linear(out_channels*2,2),
                    nn.LayerNorm(2),
                    nn.ReLU())

        if begin_align:
            self.begin_line = nn.Linear(out_channels,out_channels)

        if pos_align:
            self.pos_line = nn.Linear(out_channels,out_channels)

        self.init_weights()
        
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             kaiming_init(m, )

    def init_weights(self):
        # 自定义初始化方法
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 为线性层应用Xavier初始化
                torch.nn.init.xavier_normal_(m.weight)
                # 如果存在bias，则将其设置为0
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                # 对于层归一化，通常权重初始化为1，偏置初始化为0
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.constant_(m.bias, 0.0)

    def get_vis_mask(self, b, device, img_meta):
        h = max([meta['pad_shape'][0] for meta in img_meta])
        w = max([meta['pad_shape'][0] for meta in img_meta])
        mask = torch.zeros((b, 1, h, w), dtype=torch.float32, device=device)
        groups = b // len(img_meta) # 每四张相同图片为一组
        for i, meta in enumerate(img_meta):
            imh, imw, _ = meta['img_shape']
            mask[i * groups:(i + 1) * groups, 0, :imh, :imw] = 1
        return mask

    def position_encoding_sine(self, mask):
        b, l = mask.shape
        h = w = int(math.sqrt(l))
        mask = mask.reshape(b, h, w)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        pos_feat_dim = self.out_channels // 2

        dim_t = torch.arange(pos_feat_dim, dtype=torch.float32, device=mask.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / pos_feat_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        pos = pos.reshape(pos.shape[0], -1, pos.shape[3])
        return pos

    def forward(self, img, mask_indices, img_meta=None):
        xq = self.linear(img)
        xq_img = xq
        xq_img2 = xq.clone().detach()

        batch_size, l, d = xq.size()
        b=batch_size
        inputs = xq.contiguous()
        inputs_flatten = inputs.view(batch_size * l, d)

        if self.begin_align:
            inputs_flatten=self.begin_line(inputs_flatten)

        quantized_pt, indices, topk_values, topk_indices = self.vq(inputs_flatten)
        if self.pos_align:  # 过线性层
            quantized_pt = self.pos_line(quantized_pt)

        embedded_pt = quantized_pt.view(b, l, quantized_pt.size(-1))

        tmp_feature = torch.cat([embedded_pt,xq_img],dim=2)     # 量化前后的数据在通道方向拼接在一起

        tmp_s = self.gate_fc(tmp_feature)   # 通道数降为2，卷积
        tmp_score = F.softmax(tmp_s,dim=2)
        emb_score = tmp_score[:,:,0].unsqueeze(dim=2)         # 对应量化结果的得分
        img_score = tmp_score[:,:,1].unsqueeze(dim=2)         # 对应未量化结果的得分，前面拼接这里解开

        embedded_pt = embedded_pt*emb_score+xq_img*img_score    # 量化结果实际是量化前后结果的加权

        # visual_mask = self.get_vis_mask(batch_size, img.device).float()
        # visual_mask = F.interpolate(visual_mask, size=xq.shape[-2:]).to(dtype=torch.bool)
        visual_mask = torch.ones((batch_size,l,1), dtype=torch.bool)
        pos = self.position_encoding_sine(visual_mask[:, :, 0]).to(device=embedded_pt.device)  # 正余弦位置编码
        visual_mask = visual_mask.to(dtype=torch.float32).to(device=indices.device)

        indices = indices.view(batch_size, l, -1).float()
        indices = indices * visual_mask - 100 * (1 - visual_mask)   # 这里其实没变，这一步可以删掉

        # for itm
        masked_labels = torch.gather(indices.squeeze(2), 1, mask_indices)   # 获取相似度最大图像patch即掩码位置对应的量化后索引
        masked_neg_indices = (indices.squeeze(2) == masked_labels).float().unsqueeze(2) # 获取掩码矩阵
        neg_indices = torch.multinomial(topk_values, 1) # 全局候选indices的索引
        neg_indices = torch.gather(topk_indices, 1, neg_indices).reshape(batch_size, l, 1)  # 全局候选indices
        neg_indices = torch.gather(neg_indices.squeeze(2), 1, mask_indices).unsqueeze(2)    # 目标候选indices
        neg_indices = neg_indices * masked_neg_indices + indices * (1 - masked_neg_indices)
        neg_indices = neg_indices.reshape(batch_size * l, -1).long()
        encodings = torch.zeros(neg_indices.shape[0], self.vq.num_tokens, dtype=torch.float,device=neg_indices.device)
        encodings.scatter_(1, neg_indices, 1)  # 将 encodings 中 neg_indices 对应位置置为 1，相当于独热编码
        neg_quantize = torch.matmul(encodings, self.vq.embed)
        neg_quantize = neg_quantize.reshape(batch_size, l, -1)
        neg_quantize = neg_quantize*emb_score+xq_img2*img_score    # 量化结果实际是量化前后结果的加权

        embedded_pt += pos
        neg_quantize += pos

        xq = self.ln(embedded_pt).contiguous()   # layernorm
        neg_quantize = self.ln(neg_quantize).contiguous()

        # visual_mask = visual_mask.view(batch_size, -1).long()

        return xq, neg_quantize



class SimpleVDforVQA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_tokens,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 ):
        super(SimpleVDforVQA, self).__init__()
        self.conv = ConvModule(
            in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=activation,
            inplace=False
        )

        self.ln = nn.LayerNorm(out_channels)
        self.out_channels = out_channels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, )

    def get_vis_mask(self, b, device, img_meta):
        h = max([meta['pad_shape'][0] for meta in img_meta])
        w = max([meta['pad_shape'][0] for meta in img_meta])
        mask = torch.zeros((b, 1, h, w), dtype=torch.float32, device=device)
        groups = b // len(img_meta)
        for i, meta in enumerate(img_meta):
            imh, imw, _ = meta['img_shape']
            mask[i * groups:(i + 1) * groups, 0, :imh, :imw] = 1
        return mask

    def position_encoding_sine(self, mask):
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        pos_feat_dim = self.out_channels // 2

        dim_t = torch.arange(pos_feat_dim, dtype=torch.float32, device=mask.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / pos_feat_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, img, img_meta):
        img = img[-1]
        x = F.max_pool2d(img, 2, stride=2)

        xq = self.conv(x)
        xq_img = xq

        batch_size, c, h, w = xq.size()
        b = batch_size
        inputs = xq.permute(0, 2, 3, 1).contiguous()
        inputs_flatten = inputs.view(batch_size * h * w, c)

        quantized_pt = inputs_flatten
        embedded_pt = quantized_pt.view(b, w * h, quantized_pt.size(-1)) #b,w*h,c
        embedded_pt = embedded_pt.permute(0, 2, 1).view(b, -1, h, w) # b,c,h,w

        visual_mask = self.get_vis_mask(batch_size, img.device, img_meta).float()
        visual_mask = F.interpolate(visual_mask, size=xq.shape[-2:]).to(dtype=torch.bool)
        pos = self.position_encoding_sine(visual_mask[:, 0, :, :])
        visual_mask = visual_mask.to(dtype=torch.float32).view(batch_size, 1, h, w)

        xq = embedded_pt + pos
        xq = xq.view(b,-1,h*w).permute(0,2,1) # b,h*w,c
        xq = self.ln(xq)
        visual_mask = visual_mask.view(batch_size, -1).long()


        return xq, visual_mask


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)