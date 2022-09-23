from operator import index
import torch
import torch.nn.functional as F
import pdb


def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def localize_loss(center, props_glance, gauss_weight, pseudo_gauss_weight, num_props, **kwargs):
    bsz = center.shape[0]//num_props
    glance_loss = F.l1_loss(center, props_glance, reduction='mean') * kwargs["delta_1"]
    pseudo_weight = pseudo_gauss_weight.unsqueeze(dim=1).expand(bsz, num_props, -1).reshape(bsz*num_props, -1).detach()
    pseudo_loss = F.mse_loss(gauss_weight, pseudo_weight) * kwargs["delta_2"]

    loss = glance_loss  + pseudo_loss

    return loss, {'Localization Loss': loss.item(), '(1) glance_loss': glance_loss.item(), '(2) pseudo_loss': pseudo_loss.item()}


def rec_loss(words_logit, words_id, words_mask, num_props, pseudo_ref_words_logit, ref_words_logit, use_min=True, **kwargs):
    bsz = words_logit.size(0) // num_props
    
    pseudo_nll_loss, _ = cal_nll_loss(pseudo_ref_words_logit, words_id, words_mask)
    pseudo_nll_loss = pseudo_nll_loss.mean()
    if ref_words_logit is not None:
        ref_nll_loss, _ = cal_nll_loss(ref_words_logit, words_id, words_mask) 
        pseudo_nll_loss = (1.0-kwargs["theta"]) * pseudo_nll_loss + ref_nll_loss.mean() * kwargs["theta"]
    pseudo_nll_loss = pseudo_nll_loss*kwargs["beta_1"]

    
    words_mask1 = words_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    
    if use_min:
        nll_loss = nll_loss.view(bsz, num_props)
        min_nll_loss = nll_loss.min(dim=-1)[0]
    else:
        min_nll_loss = nll_loss

    min_nll_loss = min_nll_loss.mean()*kwargs["beta_2"]

    final_loss = pseudo_nll_loss + min_nll_loss
    
    loss_dict = {
        'Semantic Loss:': final_loss.item(),
        '(1) pseudo_nll_loss:': pseudo_nll_loss.item(),
        '(2) min_nll_loss:': min_nll_loss.item()
    }

    return final_loss, loss_dict

    
def ivc_loss(words_logit, words_id, words_mask, num_props, neg_words_logit_1=None, neg_words_logit_2=None, ref_words_logit=None, pseudo_ref_words_logit=None,use_div_loss=True, use_ref_words=True, **kwargs):
    bsz = words_logit.size(0) // num_props

    words_mask1 = words_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    min_nll_loss, idx = nll_loss.view(bsz, num_props).min(dim=-1)

    rank_loss = 0

    if pseudo_ref_words_logit is not None:
        pseudo_nll_loss, _ = cal_nll_loss(pseudo_ref_words_logit, words_id, words_mask)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        pseudo_neg_loss = torch.max(min_nll_loss - pseudo_nll_loss + kwargs["margin_1"], tmp_0)
        rank_loss = rank_loss + pseudo_neg_loss.mean()

    if ref_words_logit is not None and use_ref_words:
        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        ref_loss = torch.max(min_nll_loss - ref_nll_loss + kwargs["margin_2"], tmp_0)
        rank_loss = rank_loss + ref_loss.mean()
    
    if neg_words_logit_1 is not None:
        neg_nll_loss_1, neg_acc_1 = cal_nll_loss(neg_words_logit_1, words_id1, words_mask1)
        neg_nll_loss_1 = torch.gather(neg_nll_loss_1.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_1 = torch.max(min_nll_loss - neg_nll_loss_1 + kwargs["margin_3"], tmp_0)
        rank_loss = rank_loss + neg_loss_1.mean()
    
    if neg_words_logit_2 is not None:
        neg_nll_loss_2, neg_acc_2 = cal_nll_loss(neg_words_logit_2, words_id1, words_mask1)
        neg_nll_loss_2 = torch.gather(neg_nll_loss_2.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_2 = torch.max(min_nll_loss - neg_nll_loss_2 + kwargs["margin_3"], tmp_0)
        rank_loss = rank_loss + neg_loss_2.mean()

    

    loss = kwargs['alpha_1'] * rank_loss


    gauss_weight = kwargs['gauss_weight'].view(bsz, num_props, -1)
    gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)

    if use_div_loss:
        target = torch.eye(num_props).unsqueeze(0).cuda() * kwargs["lambda"]
        source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
        div_loss = torch.norm(target - source, dim=(1, 2))**2

        loss = loss + kwargs['alpha_2'] * div_loss.mean()
    return loss, {
        'Intra-Video Loss': loss.item(),
        '(1) hinge_loss_neg1': neg_loss_1.mean().item() if neg_words_logit_1 is not None else 0.0,
        '(2) hinge_loss_neg2': neg_loss_2.mean().item() if neg_words_logit_2 is not None else 0.0,
        '(3) hinge_loss_ref': ref_loss.mean().item() if ref_words_logit is not None and use_ref_words else 0.0,
        '(4) hinge_loss_pseudo_ref': pseudo_neg_loss.mean().item() if pseudo_ref_words_logit is not None else 0.0,
        '(5) div_loss': div_loss.mean().item() if use_div_loss else 0.0,
    }


    _, V_D = video_feats.shape
    bsz, Q_D = query_feats.shape

    # words_mask1 = words_mask.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    # words_id1 = words_id.unsqueeze(1).expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    # nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    # idx = nll_loss.view(bsz, num_props).min(dim=-1, keepdim=True)[1]
    # video_feats = video_feats.unsqueeze(1).reshape(-1, num_props, V_D).gather(index=idx.unsqueeze(-1).expand(bsz, -1, V_D), dim=1).squeeze()
    samilarity = torch.mm(F.normalize(video_feats), F.normalize(query_feats).T)
    scores = samilarity.exp()
    pos_scores = scores.diag()
    total_scores = scores.sum(dim=-1)
    loss = -(pos_scores/total_scores).log().mean() * 0.1

    # video_feats = video_feats.unsqueeze(1).reshape(-1, num_props, V_D).permute(1,0,2)
    # query_feats = query_feats.unsqueeze(1).expand(-1, num_props, Q_D).permute(1,0,2)
    # samilarity = torch.einsum("nld,ndm->nlm", F.normalize(video_feats), F.normalize(query_feats).permute(0,2,1))

    # scores = samilarity.exp()
    # loss = 0
    # for i in range(num_props):
    #     pos_scores = scores[i].diag()
    #     total_scores = scores[i].sum(dim=-1)
    #     loss += -(pos_scores/total_scores).log().mean()
    # loss /= num_props
    return loss, {'cvc_loss': loss.item()}