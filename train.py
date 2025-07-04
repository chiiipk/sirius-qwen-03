import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

import warnings
warnings.filterwarnings("ignore")

from sampler import data_sampler_CFRL
from data_loader import get_data_loader
from utils import Moment
from encoder import EncodingModel
from losses import TripletLoss
from transformers import AutoTokenizer

class Manager(object):
    def __init__(self, config, args) -> None:
        super().__init__()
        self.config = config
        self.args = args 
    def _edist(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = [] # B
        for i in range(b):
            dist_i = L2dist(x2, x1[i])
            dist.append(torch.unsqueeze(dist_i, 0)) # (N) --> (1,N)
        dist = torch.cat(dist, 0) # (B, N)
        return dist

    def _cosine_similarity(self, x1, x2):
        x2_aligned = x2.to(device=x1.device, dtype=x1.dtype)
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2_aligned, p=2, dim=1)
        sim = torch.matmul(x1_norm, x2_norm.T)
        return sim

    def get_memory_proto(self, encoder, dataset):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader(config, dataset, shuffle=False, drop_last=False,  batch_size=1)
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.detach().cpu().float()
            features.append(fea)
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)
        return proto, features

    def select_memory(self, encoder, dataset):
        '''
        only for one relation data
        '''
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader(self.config, dataset, shuffle=False, drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.detach().cpu().float()
            features.append(fea)

        features = np.concatenate(features) # tensor-->numpy array; (N, H)

        if N <= M:
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M # memory_size < len(dataset)
        distances = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit_transform(features) # (N, M)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0) # (M, H)
        mem_feas = torch.from_numpy(mem_feas)
        features = torch.from_numpy(features) # (N, H) tensor
        return mem_set, mem_feas


    def get_cluster_and_centroids(self, embeddings):
        embeddings_np = embeddings.cpu().float().numpy()
        clustering_model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=self.args.distance_threshold)
        clusters = clustering_model.fit_predict(embeddings_np)

        centroids = {}
        for cluster_id in np.unique(clusters):
            cluster_embeddings = embeddings[clusters == cluster_id]
            centroid = torch.mean(cluster_embeddings, dim=0)
            centroids[cluster_id] = centroid

        return clusters, centroids

    def train_model(self, encoder, training_data, seen_des, seen_relations, list_seen_des, is_memory=False):
        data_loader = get_data_loader(self.config, training_data, shuffle=True)
        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
        encoder.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch
        triplet = TripletLoss()
        optimizer.zero_grad()

        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    instance[k] = instance[k].to(self.config.device)

                batch_instance = {'ids': [], 'mask': []}
                batch_instance['ids'] = torch.tensor([seen_des[self.id2rel[label.item()]]['ids'] for label in labels]).to(self.config.device)
                batch_instance['mask'] = torch.tensor([seen_des[self.id2rel[label.item()]]['mask'] for label in labels]).to(self.config.device)

                hidden = encoder(instance) # b, dim
                rep_des = encoder(batch_instance, is_des = True) # b, dim
                rep_des_2 = encoder(batch_instance, is_des = True) # b, dim

                with torch.no_grad():
                    rep_seen_des = []
                    for i2 in range(len(list_seen_des)):
                        sample = {
                            'ids' : torch.tensor([list_seen_des[i2]['ids']]).to(self.config.device),
                            'mask' : torch.tensor([list_seen_des[i2]['mask']]).to(self.config.device)
                        }
                        hidden_des = encoder(sample, is_des=True)
                        rep_seen_des.append(hidden_des)
                    rep_seen_des = torch.cat(rep_seen_des, dim=0)
                    # SỬA LỖI: Gọi hàm get_cluster_and_centroids mà không cần truyền args vì nó đã là thuộc tính của class
                    clusters, clusters_centroids = self.get_cluster_and_centroids(rep_seen_des)
                flag = 0
                if len(clusters) == max(clusters) + 1:
                    flag = 1

                relationid2_clustercentroids = {}
                for index, rel in enumerate(seen_relations):
                    relationid2_clustercentroids[self.rel2id[rel]] = clusters_centroids[clusters[index]]

                relation_2_cluster = {}
                for i1 in range(len(seen_relations)):
                    relation_2_cluster[self.rel2id[seen_relations[i1]]] = clusters[i1]

                loss2 = self.moment.mutual_information_loss_cluster(hidden, rep_des, labels, temperature=self.args.temperature,relation_2_cluster=relation_2_cluster)
                loss4 = self.moment.mutual_information_loss_cluster(rep_des, rep_des_2, labels, temperature=self.args.temperature,relation_2_cluster=relation_2_cluster)

                cluster_centroids_list = [relationid2_clustercentroids[label.item()] for label in labels]
                cluster_centroids  = torch.stack(cluster_centroids_list, dim = 0).to(self.config.device)

                nearest_cluster_centroids = []
                for hid in hidden:
                    cos_similarities = torch.nn.functional.cosine_similarity(hid.unsqueeze(0), cluster_centroids.to(hid.dtype), dim=1)

                    try:
                        k_val = min(2, cos_similarities.shape[0])
                        if k_val > 1:
                            top2_similarities, top2_indices = torch.topk(cos_similarities, k=k_val, dim=0)
                            top2_centroids = relationid2_clustercentroids[labels[top2_indices[1].item()].item()]
                        else:
                            top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]
                    except RuntimeError as e:
                        print(f"RuntimeError in top-k selection: {e}")
                        top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                    nearest_cluster_centroids.append(top2_centroids)

                nearest_cluster_centroids = torch.stack(nearest_cluster_centroids, dim = 0).to(self.config.device)
                loss1 = self.moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                if flag == 0:
                    loss3 = triplet(hidden, rep_des,  cluster_centroids) + triplet(hidden, cluster_centroids, nearest_cluster_centroids)
                    loss = self.args.lambda_1*(loss1) + self.args.lambda_2*(loss2) + self.args.lambda_3*(loss3) + self.args.lambda_4*(loss4)
                else:
                    loss = self.args.lambda_1*(loss1) + self.args.lambda_2*(loss2) + self.args.lambda_4*(loss4)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if is_memory:
                    self.moment.update_des(ind, hidden.detach().cpu().float(), rep_des.detach().cpu().float(), is_memory=True)
                else:
                    self.moment.update_des(ind, hidden.detach().cpu().float(), rep_des.detach().cpu().float(), is_memory=False)

                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                sys.stdout.flush()
        print('')

    def eval_encoder_proto_des(self, encoder, seen_proto, seen_relid, test_data, rep_des):
        batch_size = 16
        test_loader = get_data_loader(self.config, test_data, False, False, batch_size)
        corrects, corrects1, corrects2, total = 0.0, 0.0, 0.0, 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            with torch.no_grad():
                hidden = encoder(instance)

            device = hidden.device
            dtype = hidden.dtype

            seen_proto_aligned = seen_proto.to(device=device, dtype=dtype)
            rep_des_aligned = rep_des.to(device=device, dtype=dtype)

            fea = hidden

            logits = self._cosine_similarity(fea, seen_proto_aligned)
            logits_des = self._cosine_similarity(fea, rep_des_aligned)
            logits_rrf = logits + logits_des

            label = label.cpu()
            cur_index = torch.argmax(logits.cpu(), dim=1)
            pred = torch.tensor([seen_relid[int(i)] for i in cur_index])
            corrects += torch.eq(pred, label).sum().item()

            cur_index1 = torch.argmax(logits_des.cpu(),dim=1)
            pred1 = torch.tensor([seen_relid[int(i)] for i in cur_index1])
            corrects1 += torch.eq(pred1, label).sum().item()

            cur_index2 = torch.argmax(logits_rrf.cpu(),dim=1)
            pred2 = torch.tensor([seen_relid[int(i)] for i in cur_index2])
            corrects2 += torch.eq(pred2, label).sum().item()

            total += label.size(0)
            acc = corrects / total if total > 0 else 0
            acc1 = corrects1 / total if total > 0 else 0
            acc2 = corrects2 / total if total > 0 else 0

            sys.stdout.write(f'[EVAL RRF] batch: {batch_num:4} | acc: {100*acc2:3.2f}%, total acc: {100*(corrects2/total):3.2f}%   ' + '\r')
            sys.stdout.flush()
        print('')
        return corrects / total, corrects1 / total, corrects2 / total

    def _read_description(self, r_path):
        rset = {}
        with open(r_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                rset[items[1]] = items[2]
        return rset

    def train(self):
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        if self.config.model == 'qwen':
            print(f"Đang tải tokenizer cho: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            from transformers import BertTokenizer
            print(f"Đang tải tokenizer cho: {self.config.bert_path}")
            self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)

        encoder = EncodingModel(self.config)
        encoder.to(self.config.device)

        cur_acc, total_acc = [], []
        cur_acc1, total_acc1 = [], []
        cur_acc2, total_acc2 = [], []
        cur_acc_num, total_acc_num = [], []
        cur_acc_num1, total_acc_num1 = [], []
        cur_acc_num2, total_acc_num2 = [], []

        memory_samples = {}
        seen_des = {}

        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):

            for rel in current_relations:
                if rel not in seen_des:
                    description = seen_descriptions[rel][0]
                    tokenized_output = self.tokenizer(
                        description,
                        padding='max_length',
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors='pt'
                    )
                    seen_des[rel] = {
                        'ids': tokenized_output['input_ids'].squeeze().tolist(),
                        'mask': tokenized_output['attention_mask'].squeeze().tolist()
                    }

            seen_relid = [self.rel2id[rel] for rel in seen_relations]
            seen_des_by_id = {self.rel2id[rel]: seen_des[rel] for rel in seen_relations}
            list_seen_des = [seen_des_by_id[rel_id] for rel_id in seen_relid]

            self.moment = Moment(self.config)
            training_data_initialize = []

            for rel in current_relations:
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])

            if step > 0:
                relations = list(set(seen_relations) - set(current_relations))
                for rel in relations:
                    training_data_initialize += memory_samples[rel]
            for rel in current_relations:
                training_data_initialize += training_data[rel]

            self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
            # SỬA LỖI: Gọi train_model mà không cần truyền `args` vì nó đã là thuộc tính của class
            self.train_model(encoder, training_data_initialize, seen_des, seen_relations, list_seen_des, is_memory=False)

            seen_proto = []
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
            seen_proto = torch.stack(seen_proto, dim=0)

            test_data_initialize_cur = [item for rel in current_relations for item in test_data[rel]]
            test_data_initialize_seen = [item for rel in seen_relations for item in historic_test_data[rel]]

            with torch.no_grad():
                encoder.eval()
                rep_des_tensors = []
                for i in range(len(list_seen_des)):
                    sample = {
                        'ids' : torch.tensor([list_seen_des[i]['ids']]).to(self.config.device),
                        'mask' : torch.tensor([list_seen_des[i]['mask']]).to(self.config.device)
                    }
                    hidden = encoder(sample, is_des=True)
                    rep_des_tensors.append(hidden)
                rep_des = torch.cat(rep_des_tensors, dim=0)
            encoder.train()

            ac1, ac1_des, ac1_rrf = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur,rep_des)
            ac2, ac2_des, ac2_rrf = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen, rep_des)

            cur_acc_num.append(ac1)
            total_acc_num.append(ac2)
            cur_acc.append(f'{ac1:.4f}')
            total_acc.append(f'{ac2:.4f}')
            print('cur_acc: ', cur_acc)
            print('his_acc: ', total_acc)

            cur_acc_num1.append(ac1_des)
            total_acc_num1.append(ac2_des)
            cur_acc1.append(f'{ac1_des:.4f}')
            total_acc1.append(f'{ac2_des:.4f}')
            print('cur_acc des: ', cur_acc1)
            print('his_acc des: ', total_acc1)

            cur_acc_num2.append(ac1_rrf)
            total_acc_num2.append(ac2_rrf)
            cur_acc2.append(f'{ac1_rrf:.4f}')
            total_acc2.append(f'{ac2_rrf:.4f}')
            print('cur_acc rrf: ', cur_acc2)
            print('his_acc rrf: ', total_acc2)

        torch.cuda.empty_cache()
        return total_acc_num, total_acc_num1, total_acc_num2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    parser.add_argument("--lambda_1", default=1, type=float)
    parser.add_argument("--lambda_2", default=1, type=float)
    parser.add_argument("--lambda_3", default=0.25, type=float)
    parser.add_argument("--lambda_4", default=0.25, type=float)
    parser.add_argument("--temperature", default=0.01, type=float)
    parser.add_argument("--distance_threshold", default=0.1, type=float)
    parser.add_argument("--top_k", default=10, type=int)

    args = parser.parse_args()

    config = Config('config.ini')

    for key, value in vars(args).items():
        if value is not None:
             setattr(config, key, value)

    print('#############params############')
    config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(config.device)

    print(f"DEBUG: Giá trị của config.task_name là: '{config.task_name}' (Loại: {type(config.task_name)})")

    # SỬA LỖI: Sử dụng .strip() để loại bỏ khoảng trắng thừa có thể có từ file config
    task_name_stripped = config.task_name.strip()
    if task_name_stripped == 'FewRel':
        print("INFO: Đang cấu hình đường dẫn cho tác vụ FewRel.")
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = './data/CFRLFewRel/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'

    elif task_name_stripped == 'Tacred':
        print("INFO: Đang cấu hình đường dẫn cho tác vụ Tacred.")
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = './data/CFRLTacred/relation_description.txt'
    else:
        raise ValueError(f"Giá trị của 'task_name' là '{config.task_name}' không được hỗ trợ. Vui lòng kiểm tra file config.ini hoặc tham số dòng lệnh.")

    if config.model == 'qwen':
        print(f'Encoding model: {config.model_name}')
    else:
        print(f'Encoding model: {config.model}')
    print(f'Task={config.task_name}, {config.num_k}-shot')
    print(f'pattern={config.pattern}')
    print('#############params############')

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    base_seed = config.seed

    acc_list, acc_list1, aac_list2 = [], [], []
    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        print('seed: ', config.seed)
        # SỬA LỖI: Truyền `args` vào constructor của Manager
        manager = Manager(config, args)
        acc, acc1, aac2 = manager.train()
        acc_list.append(acc)
        acc_list1.append(acc1)
        aac_list2.append(aac2)
        torch.cuda.empty_cache()

    accs = np.array(acc_list)
    ave = np.mean(accs, axis=0)
    print('----------END')
    print('his_acc mean: ', np.around(ave, 4))
    accs1 = np.array(acc_list1)
    ave1 = np.mean(accs1, axis=0)
    print('his_acc des mean: ', np.around(ave1, 4))
    accs2 = np.array(aac_list2)
    ave2 = np.mean(accs2, axis=0)
    print('his_acc rrf mean: ', np.around(ave2, 4))
