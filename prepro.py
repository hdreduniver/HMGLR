from tqdm import tqdm
import ujson as json
import numpy as np                                          
from adj_utils import sparse_mxs_to_torch_sparse_tensor     
import scipy.sparse as sp                                   

cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}
bio_rel2id = {'Na': 0, 'Association': 1, 'Positive_Correlation': 2, 'Negative_Correlation': 3, 'Bind': 4, 'Drug_Interaction': 5, 'Cotreatment': 6, 'Comparison': 7, 'Conversion': 8}

def read_convert_biored(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    max_entity = 0
    maxlen = 0
    entity_num = []
    entity_1, entity_2, entity_3, entity_4 = [], [], [], []
    if file_in == "":
        return None

    with open(file_in, "r") as fh:
        data = json.load(fh)

    document = data["documents"]
    for sample in tqdm(document, desc="Example"):
        if len(sample["relations"]) == 0:
            continue
        sent = ''
        sentsss = ''
        sent_map = []
        entity_id = []
        mention_pos = []
        men_ent_list = []
        for i in range(50):
            men_ent_list.append([])
            mention_pos.append([])
        relations = []
        train_triples = {}
        hts = []
        pmid = int(sample["id"])
        
        for id, text in enumerate(sample["passages"]):
            sent += text["text"]
        sents = [t.split(' ') for t in sent.split('|')]
        sents[-1] = [t for t in sents[-1] if t != '']
        sentss = [t for t in sent.split('|')]
        for t in sentss:
            sentsss = ' '.join([sentsss, t])
        sentsss = sentsss.strip()
        sentssss = sentsss.split()
        total_len = 0
        for i in range(len(sents)):
            length = len(sents[i])
            sent_map.append([total_len, total_len + length])
            total_len += length

        entity_number = 0
        men_ent_list1 = []
        for i in range(50):
            men_ent_list1.append([])
        for id, text in enumerate(sample["passages"]):
            for index, men in enumerate(text["annotations"]):
                men_text = men["text"]
                men_words = men_text.split()
                men_id = int(men["id"])
                men_ent_id = men["infons"]["identifier"]
                men_ent_type = men["infons"]["type"]
                if ',' in men_ent_id:
                    men_ent_ids = men_ent_id.split(",")
                    for id in men_ent_ids:
                        if id not in entity_id:
                            entity_id.append(id)
                            ent_id = entity_id.index(id)
                            men_ent_list1[ent_id].append(men_id)
                        else:
                            ent_id = entity_id.index(id)
                            men_ent_list1[ent_id].append(men_id)
                else:
                    if men_ent_id not in entity_id:
                        entity_id.append(men_ent_id)
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list1[ent_id].append(men_id)
                    else:
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list1[ent_id].append(men_id)
        men_ent_list1 = [t for t in men_ent_list1 if t != []]
        entity_number = len(men_ent_list1)

        docu = ''
        
        men_text_occur = {}
        search_pos = 0
        men_index = 0
        for id, text in enumerate(sample["passages"]):
            for index, men in enumerate(text["annotations"]):
                
                men_text = men["text"]
                men_words = men_text.split()
                
                men_id = int(men["id"])
                
                men_ent_id = men["infons"]["identifier"]
                men_ent_type = men["infons"]["type"]
                
                if ',' in men_ent_id:
                    men_ent_ids = men_ent_id.split(",")
                    for id in men_ent_ids:
                        if id not in entity_id:
                            entity_id.append(id)
                            ent_id = entity_id.index(id)
                            men_ent_list[ent_id].append(men_id)
                        else:
                            ent_id = entity_id.index(id)
                            men_ent_list[ent_id].append(men_id)
                else:
                    if men_ent_id not in entity_id:
                        entity_id.append(men_ent_id)
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list[ent_id].append(men_id)
                    else:
                        ent_id = entity_id.index(men_ent_id)
                        men_ent_list[ent_id].append(men_id)
            
                a = True
                while a:
                    men_start = sentssss.index(men_words[0], search_pos)
                    men_ending = men_start + len(men_words) - 1
                    if sentssss[men_ending] == men_words[-1]:
                        men_end = men_start + len(men_words)
                        a = False
                    else:
                        search_pos = men_start + 1

                for pos in sent_map:
                    if men_start >= pos[0] and men_end < pos[1] and sent_map.index(pos) <= len(sent_map) - 1:
                        men_sen_id = sent_map.index(pos)
                        break
                    elif sent_map.index(pos) == len(sent_map) - 1:
                        print("men_words:", men_words)
                        print("sentssss:", sentsss)
                        print("start,pos:", men_start, men_end)
                        print("sent_map:", sent_map)
                        print("pos:", pos)
                        print("sent_map_len, pos:", len(sent_map), sent_map.index(pos))
                        print("can not find sentence id of mention")
                        return 0
                    else:
                        continue
                
                if ',' in men_ent_id:
                    for id in men_ent_ids:
                        ent_id = entity_id.index(id)
                        mention_pos[ent_id].append(
                            (men_start, men_end, ent_id, men_sen_id, men_index, men_index + entity_number))
                else:
                    ent_id = entity_id.index(men_ent_id)
                    mention_pos[ent_id].append(
                        (men_start, men_end, ent_id, men_sen_id, men_index, men_index + entity_number))
                search_pos = men_end
                men_index += 1

        men_ent_list = [t for t in men_ent_list if t != []]
        mention_pos = [x for x in mention_pos if x != []]
        entity_pos = []
        entity_node = []
        mention_node = []
        for i in range(len(mention_pos)):
            entity_node += [[i, i, i, i, i, i, 0]]
            for men in mention_pos[i]:
                entity_pos.append(men)
                mention_node += [list(men) + [1]]

        new_sents = []
        sent1_map = {}
        i_t = 0
        i_s = 0
        sent1_pos = {}

        for sent in sents:
            sent1_pos[i_s] = len(new_sents)
            for token in sent:
                tokens_wordpiece = tokenizer.tokenize(token)
                for start, end, ent_id, sen_id, men_id, node_index in entity_pos:
                    if i_t == start:
                        tokens_wordpiece = ["^"] + tokens_wordpiece
                    if i_t + 1 == end:
                        tokens_wordpiece = tokens_wordpiece + ["^"]
                sent1_map[i_t] = len(new_sents)
                new_sents.extend(tokens_wordpiece)
                i_t += 1
            sent1_map[i_t] = len(new_sents)  
            i_s += 1
        sent1_pos[i_s] = len(new_sents)
        sents = new_sents

        ent_num = len(mention_pos)
        men_num = len(entity_pos)
  
        sentl_pos = []
        for i in range(len(sent1_pos) - 1):
            sentl_pos.append((sent1_pos[i], sent1_pos[i + 1]))

        link_node = []
        for l in range(len(sentl_pos) - 2):
            link_node += [[l, l, l, l, l, l, 2]]
        link_pos = []
        for i in range(len(sentl_pos) - 2):
            link_pos.append((sentl_pos[i], sentl_pos[i + 2]))
        
        nodes = entity_node + mention_node + link_node
        nodes = np.array(nodes)
        xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')
        l_type, r_type = nodes[xv, 6], nodes[yv, 6]
        l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]
        l_h_lid, r_h_lid = nodes[xv, 3], nodes[yv, 3]
        l_t_lid, r_t_lid = nodes[xv, 4], nodes[yv, 4]
        l_sid, r_sid = nodes[xv, 5], nodes[yv, 5]

        adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0, 'i')
        adjacency = np.full((5, l_type.shape[0], r_type.shape[0]), 0.0)
        
        adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
        adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])
        
        adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
        adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
        adjacency[1] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[1])
        adjacency[1] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[1])
        
        adj_temp = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
        adj_temp = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
        adjacency[2] = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])
        adjacency[2] = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])
        
        adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
        adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])
        
        for x, y in zip(xv.ravel(), yv.ravel()):
            if nodes[x, 5] == 0 and nodes[y, 5] == 2:
                z = np.where((l_eid == nodes[x, 2]) & (l_type == 1) & (r_type == 2) & np.logical_or(r_h_lid == nodes[y, 3], r_t_lid == nodes[y, 4]))
                temp_ = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
                temp_ = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, temp_)
                adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0
                adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0

        adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])

        for id, label in enumerate(sample["relations"]):
            infons = label["infons"]
            ent_1 = infons["entity1"]
            ent_2 = infons["entity2"]
            relation = [0] * len(bio_rel2id)
            r = bio_rel2id[infons["type"]]
            relation[r] = 1
            ent_1_id = entity_id.index(ent_1)
            ent_2_id = entity_id.index(ent_2)
            if (ent_1_id, ent_2_id) not in train_triples:
                train_triples[(ent_1_id, ent_2_id)] = [{'relation': r}]
            else:
                train_triples[(ent_1_id, ent_2_id)].append({'relation': r})
            hts.append([ent_1_id, ent_2_id])
            relations.append(relation)
            pos_samples += 1


        max_entity = max(max_entity, len(mention_pos))
        maxlen = max(maxlen, len(sents))
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        feature = {
            'input_ids': input_ids,
            'entity_pos': mention_pos,
            'labels': relations,
            'hts': hts,
            'title': pmid,
            'adjacency': adjacency,
            'link_pos': sentl_pos,
            'nodes_info': nodes,
        }
        features.append(feature)
        entity_num.append(len(mention_pos))
        print("entity_number:", len(mention_pos))
        if 1 <= len(mention_pos) < 7:
            entity_1.append(sample)
        elif 7 <= len(mention_pos) < 14:
            entity_2.append(sample)
        elif 14 <= len(mention_pos) < 21:
            entity_3.append(sample)
        else:
            entity_4.append(sample)
    
    print("features length:", len(features))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    print(len(entity_1),len(entity_2),len(entity_3),len(entity_4))
    return features

def read_convert_cdr(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    max_entity = 0
    maxlen = 0
    entity_number = []
    entity_1, entity_2, entity_3, entity_4 = [], [], [], []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)
    
    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []
        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        mention_pos = []
        ent_num = len(entities)    
        men_num = 0
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((pos[0]))
                entity_end.append((pos[1]))
                mention_pos.append((pos[0], pos[1]))
                men_num += 1

        sent_pos = {}
        i_t = 0
        i_s = 0
        sent_map = {}
        for sent in sample['sents']:
            sent_pos[i_s] = len(sents)
            for token in sent:
                tokens_wordpiece = tokenizer.tokenize(token)
                for start, end in mention_pos:
                    if start == i_t:
                        tokens_wordpiece = ["*"] + tokens_wordpiece
                    if end == i_t + 1:
                        tokens_wordpiece = tokens_wordpiece + ["*"]
                sent_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
                i_t += 1
            sent_map[i_t] = len(sents)
            i_s += 1
        sent_pos[i_s] = len(sents)

        link_node = []                              
        for l in range(len(sent_pos) - 2):          
            link_node += [[l, l, l, l, l, l, 2]]          
        mention_pos = []
        link_pos = []                               
        for i in range(len(sent_pos) - 2):          
            link_pos.append((sent_pos[i], sent_pos[i+2]))

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                r = label['r']
                dist = label['dist']
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]
                else:
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]

        entity_pos = []
        men_id = 0
        for e_id, e in enumerate(entities):
            entity_pos.append([])
            for m in e:
                start = sent_map[m["pos"][0]]
                end = sent_map[m["pos"][1]]
                s_id = m["sent_id"]

                entity_pos[-1].append((start, end, e_id, s_id, men_id, men_id+ent_num))
                men_id += 1

        entity_node = []
        mention_node = []
        for idx in range(len(entity_pos)):
            entity_node += [[idx, idx, idx, idx, idx, idx, 0]]
            for item in entity_pos[idx]:
                mention_node += [list(item) + [1]]

        nodes = entity_node + mention_node + link_node      
        nodes = np.array(nodes)         

        xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')       
        l_type, r_type = nodes[xv, 6], nodes[yv, 6]         
        l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]           
        l_h_lid, r_h_lid = nodes[xv, 3], nodes[yv, 3]       
        l_t_lid, r_t_lid = nodes[xv, 4], nodes[yv, 4]       
        l_sid, r_sid = nodes[xv, 5], nodes[yv, 5]           


        adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0, 'i')
        adjacency = np.full((5, l_type.shape[0], r_type.shape[0]), 0.0)
        adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
        adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])
        adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
        adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
        adjacency[1] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[1])      
        adjacency[1] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[1])
        adj_temp = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
        adj_temp = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
        adjacency[2] = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])      
        adjacency[2] = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])      
        adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
        adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])
        for x, y in zip(xv.ravel(), yv.ravel()):                                                       
            if nodes[x, 5] == 0 and nodes[y, 5] == 2:                                                   
                z = np.where((l_eid == nodes[x, 2]) & (l_type == 1) & (r_type == 2) & np.logical_or(r_h_lid == nodes[y, 3], r_t_lid == nodes[y, 4]))   
                temp_ = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)                
                temp_ = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, temp_)                   
                adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0                                  
                adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0                                  

        adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])      

        relations, hts, dists = [], [], []           
        for h, t in train_triples.keys():
            relation = [0] * len(cdr_rel2id)
            for mention in train_triples[h, t]:
                relation[mention["relation"]] = 1
                if mention["dist"] == "CROSS":
                    dist = 1                            
                elif mention["dist"] == "NON-CROSS":
                    dist = 0                            
            relations.append(relation)
            hts.append([h, t])
            dists.append(dist)                  

    maxlen = max(maxlen, len(sents))
    sents = sents[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(sents)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

    if len(hts) > 0:
        feature = {'input_ids': input_ids,
                    'entity_pos': mention_pos,
                    'labels': relations,
                    'dists': dists,              
                    'hts': hts,
                    'title': pmid,
                    'adjacency': adjacency,      
                    'link_pos': link_pos,       
                    'nodes_info': nodes,         
                }
        features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features


def read_convert_gda(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    max_entity = 0
    maxlen = 0
    entity_number = []
    entity_1, entity_2, entity_3, entity_4 = [], [], [], []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)
    
    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []
        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        mention_pos = []
        ent_num = len(entities)    
        men_num = 0
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((pos[0]))
                entity_end.append((pos[1]))
                mention_pos.append((pos[0], pos[1]))
                men_num += 1

        sent_pos = {}
        i_t = 0
        i_s = 0
        sent_map = {}
        for sent in sample['sents']:
            sent_pos[i_s] = len(sents)
            for token in sent:
                tokens_wordpiece = tokenizer.tokenize(token)
                for start, end in mention_pos:
                    if start == i_t:
                        tokens_wordpiece = ["*"] + tokens_wordpiece
                    if end == i_t + 1:
                        tokens_wordpiece = tokens_wordpiece + ["*"]
                sent_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
                i_t += 1
            sent_map[i_t] = len(sents)
            i_s += 1
        sent_pos[i_s] = len(sents)

        link_node = []                              
        for l in range(len(sent_pos) - 2):          
            link_node += [[l, l, l, l, l, l, 2]]          
        mention_pos = []
        link_pos = []                               
        for i in range(len(sent_pos) - 2):          
            link_pos.append((sent_pos[i], sent_pos[i+2]))

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                r = label['r']
                dist = label['dist']
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]
                else:
                    train_triple[(label['h'], label['t'])] = [{'relation': r, 'dist': dist}]

        entity_pos = []
        men_id = 0
        for e_id, e in enumerate(entities):
            entity_pos.append([])
            for m in e:
                start = sent_map[m["pos"][0]]
                end = sent_map[m["pos"][1]]
                s_id = m["sent_id"]

                entity_pos[-1].append((start, end, e_id, s_id, men_id, men_id+ent_num))
                men_id += 1

        entity_node = []
        mention_node = []
        for idx in range(len(entity_pos)):
            entity_node += [[idx, idx, idx, idx, idx, idx, 0]]
            for item in entity_pos[idx]:
                mention_node += [list(item) + [1]]

        nodes = entity_node + mention_node + link_node      
        nodes = np.array(nodes)         

        xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')       
        l_type, r_type = nodes[xv, 6], nodes[yv, 6]         
        l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]           
        l_h_lid, r_h_lid = nodes[xv, 3], nodes[yv, 3]       
        l_t_lid, r_t_lid = nodes[xv, 4], nodes[yv, 4]       
        l_sid, r_sid = nodes[xv, 5], nodes[yv, 5]           


        adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0, 'i')
        adjacency = np.full((5, l_type.shape[0], r_type.shape[0]), 0.0)
        adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
        adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])
        adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
        adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
        adjacency[1] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[1])      
        adjacency[1] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[1])
        adj_temp = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
        adj_temp = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
        adjacency[2] = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])      
        adjacency[2] = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])      
        adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
        adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])
        for x, y in zip(xv.ravel(), yv.ravel()):                                                       
            if nodes[x, 5] == 0 and nodes[y, 5] == 2:                                                   
                z = np.where((l_eid == nodes[x, 2]) & (l_type == 1) & (r_type == 2) & np.logical_or(r_h_lid == nodes[y, 3], r_t_lid == nodes[y, 4]))   
                temp_ = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)                
                temp_ = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, temp_)                   
                adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0                                  
                adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0                                  

        adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])      

        relations, hts, dists = [], [], []           
        for h, t in train_triples.keys():
            relation = [0] * len(cdr_rel2id)
            for mention in train_triples[h, t]:
                relation[mention["relation"]] = 1
                if mention["dist"] == "CROSS":
                    dist = 1                            
                elif mention["dist"] == "NON-CROSS":
                    dist = 0                            
            relations.append(relation)
            hts.append([h, t])
            dists.append(dist)                  

    maxlen = max(maxlen, len(sents))
    sents = sents[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(sents)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

    if len(hts) > 0:
        feature = {'input_ids': input_ids,
                    'entity_pos': mention_pos,
                    'labels': relations,
                    'dists': dists,              
                    'hts': hts,
                    'title': pmid,
                    'adjacency': adjacency,      
                    'link_pos': link_pos,       
                    'nodes_info': nodes,         
                }
        features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features