import pickle
import os 
import random
import numpy as np
from transformers import AutoTokenizer

class data_sampler_CFRL(object):
    def __init__(self, config=None, seed=None):
        self.config = config
        self.max_length = self.config.max_length
        self.task_length = self.config.task_length
        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        
        if self.config.model == 'qwen':
            model_path = self.config.model_name
            print(f"Đang tải Qwen tokenizer từ: {model_path}")
            # Thêm các special token ngay khi tải
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                additional_special_tokens=self.unused_tokens,
                trust_remote_code=True
            )
            # Qwen không có pad token mặc định, dùng eos_token thay thế
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Qwen không có mask token như BERT, eos_token 
            if self.tokenizer.mask_token is None:
                self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
                print("Đã thêm token [MASK] vào tokenizer.")
            self.mask_token = self.tokenizer.mask_token
            
        elif self.config.model in ['bert', 'roberta']:
            model_path = self.config.bert_path if self.config.model == 'bert' else self.config.roberta_path
            self.mask_token = '[MASK]' if self.config.model == 'bert' else '<mask>'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                additional_special_tokens=self.unused_tokens
            )
        else:
            raise ValueError(f"Model '{self.config.model}' không được hỗ trợ trong sampler.")

        self.config.vocab_size = len(self.tokenizer)
        self.config.sep_token_ids = self.tokenizer.sep_token_id
        self.config.mask_token_ids = self.tokenizer.mask_token_id
        
        if config.pattern == 'marker':
            # Lấy ID của các unused token đã thêm
            self.config.h_ids = self.tokenizer.convert_tokens_to_ids(self.unused_tokens[0])
            self.config.t_ids = self.tokenizer.convert_tokens_to_ids(self.unused_tokens[2])
        elif config.pattern in ['softprompt', 'hybridprompt']:
            self.unused_token = '[unused0]'
            self.config.prompt_token_ids = self.tokenizer.convert_tokens_to_ids(self.unused_token)

        self.sep_token_ids, self.mask_token_ids =  self.config.sep_token_ids, self.config.mask_token_ids

        # read relations
        self.id2rel, self.rel2id = self._read_relations(self.config.relation_name)
        self.rel2des, self.id2des = self._read_descriptions(self.config.relation_description)
        self.config.num_of_relation = len(self.id2rel)

        self.training_data = self._read_data(self.config.training_data, self._temp_datapath('train'))
        self.valid_data = self._read_data(self.config.valid_data, self._temp_datapath('valid'))
        self.test_data = self._read_data(self.config.test_data, self._temp_datapath('test'))

        rel_index = np.load(self.config.rel_index)
        rel_cluster_label = np.load(self.config.rel_cluster_label)
        self.cluster_to_labels = {}
        for index, i in enumerate(rel_index):
            if rel_cluster_label[index] in self.cluster_to_labels.keys():
                self.cluster_to_labels[rel_cluster_label[index]].append(i-1)
            else:
                self.cluster_to_labels[rel_cluster_label[index]] = [i-1]
        self.seed = seed
        if self.seed is not None:
            self.set_seed(self.seed)
        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)        
        print(f'Task_order: {self.shuffle_index}')
        self.batch = 0
        self.seen_relations = []
        self.history_test_data = {}
        self.seen_descriptions = {}

    def _temp_datapath(self, data_type):
        '''
        Tạo đường dẫn cho file cache dữ liệu đã xử lý.
        '''
        temp_name = [data_type]
        file_name = f'{data_type}.pkl'
        prompt_len = self.config.prompt_len * self.config.prompt_num

        if self.config.model == 'bert':
            model_process_name = '_process_BERT_'
        elif self.config.model == 'roberta':
            model_process_name = '_process_Roberta_'
        elif self.config.model == 'qwen':
            model_process_name = '_process_Qwen_'
        else:
            model_process_name = '_process_Unknown_'
        
        task_name_path = 'CFRLFewRel/CFRLdata_10_100_10_' if self.config.task_name == 'FewRel' else 'CFRLTacred/CFRLdata_6_100_5_'
        
        pattern_path = self.config.pattern
        if self.config.pattern in ['softprompt', 'hybridprompt']:
            pattern_path += f'_{prompt_len}token'
            
        mid_dir = os.path.join(
            'data', 
            task_name_path + str(self.config.num_k), 
            model_process_name + pattern_path
        )
            
        if not os.path.exists(mid_dir):
            os.makedirs(mid_dir, exist_ok=True)
            
        save_data_path = os.path.join(mid_dir, file_name)   
        return save_data_path

    
    def set_seed(self, seed):
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch == self.task_length:
            raise StopIteration()
        
        indexs = self.cluster_to_labels[self.shuffle_index[self.batch]]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:
            relation_name = self.id2rel[index]
            current_relations.append(relation_name)
            self.seen_relations.append(relation_name)
            cur_training_data[relation_name] = self.training_data[index]
            cur_valid_data[relation_name] = self.valid_data[index]
            cur_test_data[relation_name] = self.test_data[index]
            self.history_test_data[relation_name] = self.test_data[index]
            self.seen_descriptions[relation_name] = self.id2des[index]
            
        return cur_training_data, cur_valid_data, cur_test_data, current_relations,\
            self.history_test_data, self.seen_relations, self.seen_descriptions

    def _read_data(self, file, save_data_path):
        if os.path.isfile(save_data_path):
            with open(save_data_path, 'rb') as f:
                datas = pickle.load(f)
                print(f"Đã tải dữ liệu đã xử lý từ: {save_data_path}")
            return datas
        else:
            print(f"Không tìm thấy file cache. Đang xử lý dữ liệu từ: {file}")
            samples = []
            with open(file) as f:
                for i, line in enumerate(f):
                    items = line.strip().split('\t')
                    if len(items) > 8 and items[1] != 'noNegativeAnswer':
                        sample = {
                            'relation': int(items[0]) - 1,
                            'index': i,
                            'tokens': items[2].split(),
                            'description': self.id2des[int(items[0]) - 1],
                            'h': [items[3], items[7], [[int(ix) for ix in items[4].split()]]],
                            't': [items[5], items[8], [[int(ix) for ix in items[6].split()]]]
                        }
                        samples.append(sample)

            read_data = [[] for _ in range(self.config.num_of_relation)]
            for sample in samples:
                tokenized_sample = self.tokenize(sample)
                read_data[tokenized_sample['relation']].append(tokenized_sample)
            
            with open(save_data_path, 'wb') as f:
                pickle.dump(read_data, f)
                print(f"Đã lưu dữ liệu đã xử lý vào: {save_data_path}")
            return read_data

    def tokenize(self, sample):
        tokenized_sample = {
            'relation': sample['relation'],
            'index': sample['index']
        }
        
        if self.config.pattern == 'hardprompt':
            ids, mask = self._tokenize_hardprompt(sample)
        elif self.config.pattern == 'softprompt':
            ids, mask = self._tokenize_softprompt(sample)   
        elif self.config.pattern == 'hybridprompt':
            ids, mask = self._tokenize_hybridprompt(sample)                     
        elif self.config.pattern == 'marker':
            ids, mask = self._tokenize_marker(sample)
        elif self.config.pattern == 'cls':
            ids, mask = self._tokenize_cls(sample)            
        else:
            raise ValueError(f"Pattern '{self.config.pattern}' không hợp lệ.")
            
        tokenized_sample['ids'] = ids
        tokenized_sample['mask'] = mask
        return tokenized_sample

    def _read_relations(self, file):
        id2rel, rel2id = {}, {}
        with open(file) as f:
            for index, line in enumerate(f):
                rel = line.strip()
                id2rel[index] = rel
                rel2id[rel] = index
        return id2rel, rel2id
    
    def _read_descriptions(self, file):
        rel2des, id2des = {}, {}
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            for index, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) > 2:
                    rel_name = parts[1]
                    description = parts[2:]
                    rel2des[rel_name] = description
                    id2des[index] = description
        return rel2des, id2des
    
    def _tokenize_template(self, prompt_text):
        # Hàm chung để tokenize, tránh lặp code
        ids = self.tokenizer.encode(
            prompt_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        mask = np.zeros(self.max_length, dtype=np.int32)
        try:
            end_index = ids.index(self.sep_token_ids)
            mask[:end_index + 1] = 1
        except ValueError:
            # Nếu không có SEP token (do bị cắt), mask toàn bộ những gì không phải PAD
            pad_token_id = self.tokenizer.pad_token_id
            for i, token_id in enumerate(ids):
                if token_id != pad_token_id:
                    mask[i] = 1
        return ids, mask
        
    def _tokenize_softprompt(self, sample):
        prompt_len = self.config.prompt_len
        text = ' '.join(sample['tokens'])
        prompt_tokens = ' '.join([self.unused_token] * (prompt_len * 4))
        prompt_text = f"{text} {prompt_tokens}"
        return self._tokenize_template(prompt_text)

    def _tokenize_hybridprompt(self, sample):
        prompt_len = self.config.prompt_len
        text = ' '.join(sample['tokens'])
        h, t = sample['h'][0], sample['t'][0]
        prompt_seg = ' '.join([self.unused_token] * prompt_len)
        prompt_text = f"{text} {prompt_seg} {h} {prompt_seg} {self.mask_token} {prompt_seg} {t} {prompt_seg}"
        return self._tokenize_template(prompt_text)

    def _tokenize_hardprompt(self, sample):
        text = ' '.join(sample['tokens'])
        h, t = sample['h'][0], sample['t'][0]
        prompt_text = f"{text} {h} {self.mask_token} {t}"
        return self._tokenize_template(prompt_text)

    def _tokenize_marker(self, sample):
        raw_tokens = sample['tokens']
        h_start, h_end = sample['h'][2][0][0], sample['h'][2][0][-1]
        t_start, t_end = sample['t'][2][0][0], sample['t'][2][0][-1]
        
        new_tokens = []
        # Sắp xếp các marker để chèn đúng thứ tự
        markers = sorted([
            (h_start, self.unused_tokens[0]), 
            (h_end + 1, self.unused_tokens[1]), 
            (t_start, self.unused_tokens[2]), 
            (t_end + 1, self.unused_tokens[3])
        ], key=lambda x: x[0], reverse=True)
        
        temp_tokens = list(raw_tokens)
        for index, marker_token in markers:
            temp_tokens.insert(index, marker_token)
        
        return self._tokenize_template(' '.join(temp_tokens))

    def _tokenize_cls(self, sample):
        prompt_text = ' '.join(sample['tokens'])
        return self._tokenize_template(prompt_text)
