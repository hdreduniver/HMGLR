import torch
import torch.nn as nn
import math
from torch.nn import init
import torch.nn.functional as F
from long_seq import process_long_input
from losses import ATLoss
from torch.nn.utils.rnn import pad_sequence
from rgcn import RGCN_Layer
from utils import EmbedLayer
from ushape import AttentionUshape
from msca import MSCA

class PPM(nn.Module):
    def __init__(self, down_dim):
        super(PPM, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(512, down_dim, 3, padding=1),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(6, 6)),
            nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv1_up = F.interpolate(conv1, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv2_up = F.interpolate(conv2, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv3_up = F.interpolate(conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv4_up = F.interpolate(conv4, size=x.size()[2:], mode='bilinear', align_corners=True)
        return self.fuse(torch.cat((conv1_up, conv2_up, conv3_up, conv4_up), 1))

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=512, num_labels=-1, max_entity=22):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()
        self.extractor_trans = nn.Linear(config.hidden_size, emb_size)
        self.ht_extractor = nn.Linear(emb_size * 4, emb_size * 2)
        self.MIP_Linear = nn.Linear(emb_size * 5, emb_size * 4)
        self.MIP_Linear2 = nn.Linear(emb_size * 4, emb_size * 2)
        self.bilinear = nn.Linear(emb_size * 2, config.num_labels)
        self.emb_size = emb_size
        self.num_labels = num_labels
        self.max_entity = max_entity
        self.type_dim = 20
        self.rgcn = RGCN_Layer(emb_size + self.type_dim, emb_size, 3, 5)
        self.type_embed = EmbedLayer(num_embeddings=3, embedding_dim=self.type_dim, dropout=0.0)
        inter_channel = int(emb_size // 2)
        self.sigmoid = nn.Sigmoid()
        self.msca = MSCA(dim=256)
        self.ppm = PPM(down_dim=3)
        self.segmentation_net = AttentionUshape(input_channels=3, class_number=512, down_channel=512)
        self.reason_e1 = nn.Sequential(
            nn.Conv2d(emb_size, inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )
        self.reason_e2 = nn.Sequential(
            nn.Conv2d(inter_channel, emb_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )
        self.reason_m1 = nn.Sequential(
            nn.Conv2d(emb_size, inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, inter_channel, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )
        self.reason_m2 = nn.Sequential(
            nn.Conv2d(inter_channel, emb_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
        )

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def make_graph(self, sequence_output, attention, entity_pos, link_pos, nodes_info):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        nodes_batch = []
        entity_att_batch = []
        entity_node_batch = []
        mention_pos_batch = []
        mention_att_batch = []
        for i in range(len(entity_pos)):
            entity_nodes, mention_nodes, link_nodes = [], [], []
            entity_att = []
            mention_att = []
            mention_pos = []
            for start, end in link_pos[i]:
                if end + offset < c:
                    link_rep = sequence_output[i, start + offset: end + offset]
                    link_att = attention[i, :, start + offset: end + offset, start + offset: end + offset]
                    link_att = torch.mean(link_att, dim=0)
                    link_rep = torch.mean(torch.matmul(link_att, link_rep), dim=0)
                elif start + offset < c:
                    link_rep = sequence_output[i, start + offset:]
                    link_att = attention[i, :, start + offset:, start + offset:]
                    link_att = torch.mean(link_att, dim=0)
                    link_rep = torch.mean(torch.matmul(link_att, link_rep), dim=0)
                else:
                    link_rep = torch.zeros(self.config.hidden_size).to(sequence_output)
                link_nodes.append(link_rep)
            for e in entity_pos[i]:
                mention_pos.append(len(mention_att))
                if len(e) > 1:
                    m_emb, e_att = [], []
                    for start, end, e_id, h_lid, t_lid, sid in e:
                        if start + offset < c:
                            mention_nodes.append(sequence_output[i, start + offset])
                            m_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                            mention_att.append(attention[i, :, start + offset])
                        else:
                            mention_nodes.append(torch.zeros(self.config.hidden_size).to(sequence_output))
                            m_emb.append(torch.zeros(self.config.hidden_size).to(sequence_output))
                            e_att.append(torch.zeros(h, c).to(attention))
                            mention_att.append(torch.zeros(h, c).to(attention))
                    if len(m_emb) > 0:
                        m_emb = torch.logsumexp(torch.stack(m_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        m_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end, e_id, h_lid, t_lid, sid = e[0]
                    if start + offset < c:
                        mention_nodes.append(sequence_output[i, start + offset])
                        m_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                        mention_att.append(attention[i, :, start + offset])
                    else:
                        m_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                        mention_att.append(torch.zeros(h, c).to(attention))
                entity_nodes.append(m_emb)
                entity_att.append(e_att)
            mention_pos.append(len(mention_att))
            entity_att = torch.stack(entity_att, dim=0)
            entity_att_batch.append(entity_att)
            entity_nodes = torch.stack(entity_nodes, dim=0)
            mention_nodes = torch.stack(mention_nodes, dim=0)
            mention_att = torch.stack(mention_att, dim=0)
            link_nodes = torch.stack(link_nodes, dim=0)
            nodes = torch.cat([entity_nodes, mention_nodes, link_nodes], dim=0)
            nodes_type = self.type_embed(nodes_info[i][:, 6].to(sequence_output.device))
            nodes = torch.cat([nodes, nodes_type], dim=1)
            nodes_batch.append(nodes)
            entity_node_batch.append(entity_nodes)
            mention_att_batch.append(mention_att)
            mention_pos_batch.append(mention_pos)
        nodes_batch = pad_sequence(nodes_batch, batch_first=True, padding_value=0.0)
        return nodes_batch, entity_att_batch, entity_node_batch, mention_att_batch, mention_pos_batch

    def relation_map(self, gcn_nodes, entity, entity_att, entity_pos, sequence_output, mention_att):
        entity_cg, mention_cg = [], []
        nodes = gcn_nodes[-1]
        m_num_max = 0
        e_num_max = 0
        for i in range(len(entity_pos)):
            m_num, _, _ = mention_att[i].size()
            m_num_max = m_num if m_num > m_num_max else m_num_max
            e_num = len(entity_pos[i])
            e_num_max = e_num if e_num > e_num_max else e_num_max
        for i in range(len(entity_pos)):
            e_num = len(entity_pos[i])
            entity_stru = nodes[i][: e_num]
            m_num, head_num, dim = mention_att[i].size()
            mention_stru = nodes[i][e_num: e_num + m_num]
            e_att = entity_att[i].mean(1)
            e_att = e_att / (e_att.sum(1, keepdim=True) + 1e-5)
            e_context = torch.einsum('ij, jl->il', e_att, sequence_output[i])
            m_att = mention_att[i].mean(1)
            m_att = m_att / (m_att.sum(1, keepdim=True) + 1e-5)
            m_context = torch.einsum('ij,jl->il', m_att, sequence_output[i])
            e_cg = e_context + entity_stru
            n, h = e_cg.size()
            e_s = torch.zeros([e_num_max, h]).to(sequence_output)
            e_s[:n] = e_cg
            e_s_map = torch.einsum('ij, jk->jik', e_s, e_s.t())
            entity_cg.append(e_s_map)
            m_cg = m_context + mention_stru
            m, h = m_cg.size()
            m_s = torch.zeros([m_num_max, h]).to(sequence_output)
            m_s[:m] = m_cg
            m_s_map = torch.einsum('ij,jk->jik', m_s, m_s.t())
            mention_cg.append(m_s_map)
        entity_cg = torch.stack(entity_cg, dim=0)
        mention_cg = torch.stack(mention_cg, dim=0)
        return entity_cg, mention_cg

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                adjacency=None,
                link_pos=None,
                nodes_info=None,
                instance_mask=None,
                ):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        sequence_output = self.extractor_trans(sequence_output)
        nodes, entity_att, entity_node_batch, mention_att, mentions_pos = self.make_graph(sequence_output, attention, entity_pos, link_pos, nodes_info)
        gcn_nodes = self.rgcn(nodes, adjacency)
        entity_cg, mention_cg = self.relation_map(gcn_nodes, entity_node_batch, entity_att, entity_pos, sequence_output, mention_att)
        entity_cg_down = self.ppm(entity_cg)
        mention_cg_down = self.ppm(mention_cg)
        feature_e = self.segmentation_net(entity_cg_down).permute(0, 3, 1, 2).contiguous()
        feature_m = self.segmentation_net(mention_cg_down).permute(0, 3, 1, 2).contiguous()
        r_rep_e = self.reason_e1(feature_e)
        r_rep_e = self.msca(r_rep_e)
        r_rep_e = self.reason_e2(r_rep_e)
        r_rep_m = self.reason_m1(feature_m)
        r_rep_m = self.msca(r_rep_m)
        r_rep_m = self.reason_m2(r_rep_m)
        relation = []
        entity_h = []
        entity_t = []
        sc_feature_e = []
        nodes_re = torch.cat([gcn_nodes[0], gcn_nodes[-1]], dim=-1)
        for i in range(len(entity_pos)):
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            r_v1 = r_rep_e[i, :, ht_i[:, 0], ht_i[:, 1]].transpose(1, 0)
            r_v2 = []
            for j in range(ht_i.shape[0]):
                h_e_pos = ht_i[j, 0]
                t_e_pos = ht_i[j, 1]
                e_m_feat = r_rep_m[i, :, mentions_pos[i][h_e_pos]:mentions_pos[i][h_e_pos + 1],
                           mentions_pos[i][t_e_pos]:mentions_pos[i][t_e_pos + 1]]
                e_feat = torch.mean(e_m_feat, dim=[1, 2]).reshape([1, -1])
                r_v2.append(e_feat)
            r_v2 = torch.cat(r_v2, dim=0)
            relation.append(torch.cat([r_v1, r_v2], dim=-1))
            f_e = feature_e[i, :, ht_i[:, 0], ht_i[:, 1]].transpose(1, 0)
            sc_feature_e.append(f_e)
            e_h = torch.index_select(nodes_re[i], 0, ht_i[:, 0])
            e_t = torch.index_select(nodes_re[i], 0, ht_i[:, 1])
            entity_h.append(e_h)
            entity_t.append(e_t)
        relation = torch.cat(relation, dim=0)
        sc_feature_e = torch.cat(sc_feature_e, dim=0)
        entity_h = torch.cat(entity_h, dim=0)
        entity_t = torch.cat(entity_t, dim=0)
        entity_ht = self.ht_extractor(torch.cat([entity_h, entity_t], dim=-1))
        relation_rep = self.MIP_Linear(torch.cat([relation, sc_feature_e, entity_ht], dim=-1))
        relation_rep = torch.tanh(self.MIP_Linear2(relation_rep))
        logits = self.bilinear(relation_rep)
        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            a = loss.to(sequence_output)
            output = (loss.to(sequence_output),) + output
        return output