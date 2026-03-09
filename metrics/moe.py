import torch

import torch.distributed as dist


def collect_moe_metrics(model, ddp, is_master_process):
    moe_layer_stats = model.get_moe_stats()

    for layer_id, moe in moe_layer_stats:
        if ddp:
            dist.all_reduce(moe.acc_top1_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(moe.acc_topk_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(moe.acc_p_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(moe.acc_tokens, op=dist.ReduceOp.SUM)

    moe_metrics = None
    if is_master_process:
        moe_metrics = {}

        effs = []
        max_shares = []
        dead_counts = []
        cvs1 = []
        effs_k = []
        max_shares_k = []
        dead_counts_k = []
        cvsk = []
        pmean_maxes = []
        for layer_id, moe in moe_layer_stats:
            # top-1 utilization
            c1 = moe.acc_top1_counts.float()
            p1 = c1 / c1.sum().clamp(min=1.0)
            cv = (c1.std(unbiased=False) / (c1.mean() + 1e-9)).item() # coef of var

            # entropy + effective experts (from top-1 usage)
            entropy = -(p1 * (p1 + 1e-9).log()).sum()
            eff_experts = torch.exp(entropy).item()
            max_share = p1.max().item()
            dead = int((moe.acc_top1_counts == 0).sum().item())

            effs.append((layer_id, eff_experts))
            max_shares.append((layer_id, max_share))
            dead_counts.append((layer_id, dead))
            cvs1.append((layer_id, cv))

            # top-k utilization (optional)
            ck = moe.acc_topk_counts.float()
            pk = ck / ck.sum().clamp(min=1.0)
            cvk = (ck.std(unbiased=False) / (ck.mean() + 1e-9)).item()

            # entropy + effective experts (from top-k usage)
            entropy_k = -(pk * (pk + 1e-9).log()).sum()
            eff_experts_k = torch.exp(entropy_k).item()
            max_share_k = pk.max().item()
            dead_k = int((moe.acc_topk_counts == 0).sum().item())

            effs_k.append((layer_id, eff_experts_k))
            max_shares_k.append((layer_id, max_share_k))
            dead_counts_k.append((layer_id, dead_k))
            cvsk.append((layer_id, cvk))

            # mean router probs (token-weighted)
            p_mean = (moe.acc_p_sum / moe.acc_tokens.clamp(min=1)).to(torch.float32)
            pmean_max = float(p_mean.max().item())

            pmean_maxes.append((layer_id, pmean_max))

            moe_metrics.update({
                f'moe/layer_{layer_id}/eff_experts_top1': eff_experts,
                f'moe/layer_{layer_id}/max_share_top1': max_share,
                f'moe/layer_{layer_id}/dead_experts_top1': dead,
                f'moe/layer_{layer_id}/cv_top1': cv,
                f'moe/layer_{layer_id}/eff_experts_topk': eff_experts_k,
                f'moe/layer_{layer_id}/max_share_topk': max_share_k,
                f'moe/layer_{layer_id}/dead_experts_topk': dead_k,
                f'moe/layer_{layer_id}/cv_topk': cvk,
                f'moe/layer_{layer_id}/max_p_mean': pmean_max,
            })

        # Summary stats
        eff_vals = [v for _, v in effs]
        ms_vals = [v for _, v in max_shares]
        dead_vals = [v for _, v in dead_counts]
        cv1_vals = [v for _, v in cvs1]

        worst_eff_layer, worst_eff = min(effs, key=lambda x: x[1])
        worst_ms_layer, worst_ms = max(max_shares, key=lambda x: x[1])
        worst_dead_layer, worst_dead = max(dead_counts, key=lambda x: x[1])
        worst_cv1_layer, worst_cv1 = max(cvs1, key=lambda x: x[1])

        eff_vals_k = [v for _, v in effs_k]
        ms_vals_k = [v for _, v in max_shares_k]
        dead_vals_k = [v for _, v in dead_counts_k]
        cvk_vals = [v for _, v in cvsk]

        worst_eff_layer_k, worst_eff_k = min(effs_k, key=lambda x: x[1])
        worst_ms_layer_k, worst_ms_k = max(max_shares_k, key=lambda x: x[1])
        worst_dead_layer_k, worst_dead_k = max(dead_counts_k, key=lambda x: x[1])
        worst_cvk_layer, worst_cvk = max(cvsk, key=lambda x: x[1])

        pmm_vals = [v for _, v in pmean_maxes]

        moe_metrics.update({
            # top 1
            'moe_summary_top1/eff_experts_mean': float(sum(eff_vals) / len(eff_vals)),
            'moe_summary_top1/eff_experts_min': float(min(eff_vals)),
            'moe_summary_top1/max_share_mean': float(sum(ms_vals) / len(ms_vals)),
            'moe_summary_top1/max_share_max': float(max(ms_vals)),
            'moe_summary_top1/dead_experts_sum': int(sum(dead_vals)),
            'moe_summary_top1/cv_mean': float(sum(cv1_vals) / len(cv1_vals)),
            'moe_summary_top1/cv_max': float(max(cv1_vals)),
            # top 1 worst
            'moe_summary_top1_worst/eff_layer': int(worst_eff_layer),
            'moe_summary_top1_worst/eff_value': float(worst_eff),
            'moe_summary_top1_worst/max_share_layer': int(worst_ms_layer),
            'moe_summary_top1_worst/max_share_value': float(worst_ms),
            'moe_summary_top1_worst/dead_layer': int(worst_dead_layer),
            'moe_summary_top1_worst/dead_value': int(worst_dead),
            'moe_summary_top1_worst/cv_layer': int(worst_cv1_layer),
            'moe_summary_top1_worst/cv_value': float(worst_cv1),
            # top k
            'moe_summary_topk/eff_experts_mean': float(sum(eff_vals_k) / len(eff_vals_k)),
            'moe_summary_topk/eff_experts_min': float(min(eff_vals_k)),
            'moe_summary_topk/max_share_mean': float(sum(ms_vals_k) / len(ms_vals_k)),
            'moe_summary_topk/max_share_max': float(max(ms_vals_k)),
            'moe_summary_topk/dead_experts_sum': int(sum(dead_vals_k)),
            'moe_summary_topk/cv_mean': float(sum(cvk_vals) / len(cvk_vals)),
            'moe_summary_topk/cv_max': float(max(cvk_vals)),
            # top k worst
            'moe_summary_topk_worst/eff_layer': int(worst_eff_layer_k),
            'moe_summary_topk_worst/eff_value': float(worst_eff_k),
            'moe_summary_topk_worst/max_share_layer': int(worst_ms_layer_k),
            'moe_summary_topk_worst/max_share_value': float(worst_ms_k),
            'moe_summary_topk_worst/dead_layer': int(worst_dead_layer_k),
            'moe_summary_topk_worst/dead_value': int(worst_dead_k),
            'moe_summary_topk_worst/cv_layer': int(worst_cvk_layer),
            'moe_summary_topk_worst/cv_value': float(worst_cvk),
            # general
            'moe_summary/max_p_mean_mean': float(sum(pmm_vals) / len(pmm_vals)),
            'moe_summary/max_p_mean_max': float(max(pmm_vals)),

        })

    return moe_metrics
