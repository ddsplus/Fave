import torch.nn as nn
import torch
import math
from fave import Fave
import torch.nn.functional as F
import copy
import numpy as np
import torch as th
from torch.func import jvp
import time


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Att_Fave_model(nn.Module):
    def __init__(self, fave, args):
        super(Att_Fave_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.fave = fave
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
        self.mask_ratio = args.mask_ratio
        self.stage = 'pretrain'
        self.eps = args.eps

        self.train_mask_ratio = args.train_mask_ratio
        self.infer_mask_ratio = args.infer_mask_ratio
        self.loss_aux_weight = args.loss_aux_weight
        self.loss_straight_weight = args.loss_straight_weight
        self.loss_pretrain_weight = args.loss_pretrain_weight
        self.mask_end = args.mask_end

    def switch_to_finetune(self):
        """Switch to finetune mode: freeze parameters and update stage flag."""
        print("\n[Auto-Switch] Switching to Finetuning Stage...")

        self.stage = 'finetune'

        self.item_embeddings.weight.requires_grad = False
        self.position_embeddings.weight.requires_grad = False

        print("[Auto-Switch] Item & Position Embeddings Frozen.")


    def fave_pre(self, item_rep, tag_emb, mask_seq):
        x_0, decode_out, t_rf_expand, t, z0  = self.fave(item_rep, tag_emb, mask_seq)
        return x_0, decode_out, t_rf_expand, t, z0

    def reverse(self, item_rep, z0, mask_seq, stage="pretrain"):
        reverse_pre = self.fave.reverse_p_sample_rf(item_rep, z0, mask_seq, stage)
        return reverse_pre

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_fave_tat(self, rep_fave, labels):
        """Compute cross-entropy loss from the decoded representation."""
        scores = torch.matmul(rep_fave, self.item_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))

    def fave_rep_pre(self, rep_fave):
        scores = torch.matmul(rep_fave, self.item_embeddings.weight.t())
        return scores

    def loss_rmse(self, rep_fave, labels):
        rep_gt = self.item_embeddings(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_fave))

    def routing_rep_pre(self, rep_fave):
        item_norm = (self.item_embeddings.weight**2).sum(-1).view(-1, 1)  ## N x 1
        rep_norm = (rep_fave**2).sum(-1).view(-1, 1)  ## B x 1
        sim = torch.matmul(rep_fave, self.item_embeddings.weight.t())  ## B x N
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        dist = torch.clamp(dist, 0.0, np.inf)

        return -dist

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep/item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1)/torch.sum(mask_seq, dim=-1))

    def loss_fave_src(self, rep_fave, target_embeddings):
        """Compute MSE loss between denoised embedding and target embedding."""
        loss = self.loss_mse(rep_fave, target_embeddings)
        return loss

    def position_sincos_embedding(self, position_ids, dim, max_period=10000):
        device = position_ids.device
        position_ids = position_ids.float()

        dim_t = th.arange(dim, dtype=th.float32, device=device)
        dim_t = max_period ** (2 * (dim_t // 2) / dim)

        position = position_ids[:, None]  # [N, 1]
        div_term = position / dim_t  # [N, dim]

        embedding = th.zeros((position.size(0), dim), device=device)

        embedding[:, 0::2] = th.sin(div_term[:, 0::2])
        embedding[:, 1::2] = th.cos(div_term[:, 1::2])

        return embedding

    def switch_Matrix(self, sequence, device):

        batch_size, seq_len = sequence.size()

        num_items = self.item_num
        sparse_matrix = torch.zeros(batch_size, num_items, device=device)
        for i in range(batch_size):
            row_data = sequence[i]
            non_zero_indices = row_data[row_data != 0]
            sparse_matrix[i, non_zero_indices] = 1

        return sparse_matrix

    def balanced_mse_loss(self, target, output, mask_ratio=1.0):

        target_shape = target.shape

        num_ones = torch.sum(target == 1).item()

        num_zeros = torch.sum(target == 0).item()

        num_selected_zeros = int(min(num_zeros, num_ones * mask_ratio))

        zero_positions = (target == 0).nonzero(as_tuple=True)
        one_positions = (target == 1).nonzero(as_tuple=True)

        zero_rows, zero_cols = zero_positions
        one_rows, one_cols = one_positions

        selected_zero_indices = torch.randint(0, num_zeros, (num_selected_zeros,))
        selected_zero_positions = (zero_rows[selected_zero_indices], zero_cols[selected_zero_indices])

        mask = torch.zeros_like(target)
        mask[selected_zero_positions] = 1
        mask[one_positions] = 1

        masked_target = target * mask
        masked_output = output * mask

        mse_loss = self.loss_mse(masked_output, masked_target)
        return mse_loss


    def calculate_loss_pretrain(self, seq_Matrix, item_embeddings, tag_emb, mask_seq):
        rep_fave, decode_out, t_rf_expand, t_rf, z0 = self.fave_pre(item_embeddings, tag_emb, mask_seq)

        loss_mse = self.balanced_mse_loss(seq_Matrix, decode_out, self.mask_ratio)
        scores = loss_mse
        loss_fave_src = self.loss_fave_src(rep_fave, tag_emb)

        item_rep_dis = loss_fave_src

        return scores, rep_fave, t_rf_expand, t_rf, item_rep_dis, loss_mse

    def calculate_loss_finetune(self, sequence, target_emb, item_embeddings, mask_seq, seq_Matrix):
        batch_size = sequence.shape[0]
        device = sequence.device
        extra = (1 / self.eps) - 1

        t = torch.rand(batch_size, device=device)
        r = torch.rand(batch_size, device=device) * (1 - t) + t
        mask_end = torch.rand(batch_size, device=device) < self.mask_end
        r[mask_end] = 1.0

        valid_lens = mask_seq.sum(dim=1).long()
        valid_lens = torch.clamp(valid_lens, min=1)

        random_indices = (torch.rand(batch_size, device=device) * valid_lens).long()

        batch_indices = torch.arange(batch_size, device=device)
        noise_history = item_embeddings[batch_indices, random_indices] # [B, H]

        noise_mask_ratio = self.train_mask_ratio
        mask_noise = (torch.rand_like(noise_history) > (1 - noise_mask_ratio)).float()

        noise = mask_noise * noise_history

        t_b = t.view(-1, 1)
        z_t = t_b * target_emb + (1 - t_b) * noise

        v_t = target_emb - noise

        # 3. JVP
        def model_fn(z, curr_t, curr_r):
            u, _ = self.fave.xstart_model(item_embeddings, z, curr_t * extra, curr_r * extra, mask_seq)
            return u

        primals = (z_t, t, r)

        tangents = (v_t, torch.ones_like(t), torch.zeros_like(r))

        with torch.amp.autocast(device_type="cuda", enabled=False):
            u_pred, dudt = jvp(model_fn, primals, tangents)

        loss_matching = self.loss_mse(u_pred, v_t)

        loss_straightness = torch.mean(dudt ** 2)

        # 4. Decoder
        x_pred_final = z_t + (1 - t_b) * u_pred
        decode_out = self.fave.xstart_model.decoder(x_pred_final)
        loss_decoder = self.balanced_mse_loss(seq_Matrix, decode_out, self.mask_ratio)

        return x_pred_final, loss_matching, loss_decoder, loss_straightness

    def forward(self, sequence, tag, forward_mse_time = 0, train_flag=True):
        seq_length = sequence.size(1)
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings)  ## dropout first than layernorm

        item_embeddings = self.LayerNorm(item_embeddings)
        mask_seq = (sequence>0).float()

        if train_flag:
            tag_emb = self.item_embeddings(tag.squeeze(-1))  ## B x H
            seq_Matrix = self.switch_Matrix(sequence, device=sequence.device)
            if self.stage == 'pretrain':
                # Stage 1: Pretrain
                scores, rep_fave, t_rf_expand, t_rf, item_rep_dis, loss_mse  = self.calculate_loss_pretrain(
                    seq_Matrix, item_embeddings, tag_emb, mask_seq
                    )

                scores = loss_mse
                loss_main = item_rep_dis

                total_loss = loss_main + self.loss_pretrain_weight * scores

                return total_loss, rep_fave, self.stage, None, None, None

            else:
                # Stage 2: Finetune
                rep_fave, loss_main, loss_aux, loss_straightness = self.calculate_loss_finetune(
                    sequence, tag_emb, item_embeddings, mask_seq, seq_Matrix
                )
                tot_loss = loss_main + self.loss_aux_weight * loss_aux + self.loss_straight_weight * loss_straightness
                return tot_loss, rep_fave, None, None, None, None
        else:
            if self.stage == "pretrain":

                z0 = th.randn_like(item_embeddings[:,-1,:])

                rep_fave = self.reverse(item_embeddings, z0, mask_seq)

                scores = None
            else:
                batch_size = item_embeddings.shape[0]
                seq_len = item_embeddings.shape[1]

                valid_lengths = mask_seq.sum(dim=1).long() # [B]
                valid_lengths = torch.clamp(valid_lengths, min=1)

                random_indices = (torch.rand(batch_size, device=item_embeddings.device) * valid_lengths).long()
                random_indices = torch.clamp(random_indices, min=0, max=seq_len-1)

                z0_base = item_embeddings[torch.arange(batch_size, device=item_embeddings.device), random_indices, :]

                noise_mask_ratio = self.infer_mask_ratio

                mask_z0 = (torch.rand_like(z0_base) > (1 - noise_mask_ratio)).float()

                z0 = mask_z0 * z0_base

                rep_fave = self.reverse(item_embeddings, z0, mask_seq, "finetune")

            t_rf_expand, t_rf, item_rep_dis= None, None, None

            return None, rep_fave, t_rf_expand, t_rf, item_rep_dis, z0


def create_model_Fave(args):
    Fave_pre = Fave(args)
    return Fave_pre
