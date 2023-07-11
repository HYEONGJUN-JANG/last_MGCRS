import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import deque
from loguru import logger
from copy import deepcopy


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    # input_ids = prefix + input_ids[-truncate_size:] + suffix
    # input_ids = input_ids + [0] * (max_length - len(input_ids))
    # return input_ids
    if truncate_size <= len(input_ids):
        input_ids = prefix + input_ids[len(input_ids) - truncate_size:] + suffix
    else:
        input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))


class GenerationDataset(Dataset):
    """
    goal, topic, 용 dataset
    """

    def __init__(self, args, data_sample, knowledgeDB, tokenizer, mode='train', subtask='response'):
        super(Dataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.augmented_raw_sample = data_sample
        self.mode = mode  # train , test
        self.subtask = subtask  # goal , topic
        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))
        self.idxList = deque(maxlen=len(self.augmented_raw_sample))
        self.TopicTask_Train_Prompt_usePredGoal = args.TopicTask_Train_Prompt_usePredGoal
        self.TopicTask_Test_Prompt_usePredGoal = args.TopicTask_Test_Prompt_usePredGoal
        # TopicTask_Train_Prompt_usePredGoal TopicTask_Test_Prompt_usePredGoal

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        self.idxList.append(idx)
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, goal, topic, situation, target_knowledge_idx, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id

        context_batch = defaultdict()
        context_batch['goal_type'] = self.tokenizer(goal, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids
        resp_batch = []
        context_len_batch = []

        ## Prompt 관련
        prefix = ''
        if self.subtask == 'topic':
            if self.TopicTask_Test_Prompt_usePredGoal:
                predicted_goal = data['predicted_goal'][0]
            else:
                predicted_goal = goal

            prefix = self.tokenizer.encode('<goal>%s. <profile>%s.' % (predicted_goal, user_profile))[:int(self.args.gt_max_length * 2 / 3)]
            prompt = self.tokenizer.encode('. predict the next topic: ')
        elif self.subtask == 'goal':
            prefix = self.tokenizer.encode('<profile>%s.' % user_profile)[:int(self.args.gt_max_length * 2 / 3)]
            prompt = self.tokenizer.encode('predict the next goal: ')
        else:
            prefix, prompt = [], []

        ## Dialog input 관련
        # dialog = self.tokenizer('<dialog>' + dialog, max_length=self.args.gt_max_length - len(prompt), truncation=True).input_ids
        dialog = self.tokenizer('<dialog>' + dialog).input_ids[-(self.args.gt_max_length - len(prefix) - len(prompt)):]
        dialog = prefix + dialog + prompt

        if self.subtask == 'goal':
            label = self.tokenizer(goal, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids
        elif self.subtask == 'topic':
            label = self.tokenizer(topic, max_length=self.args.max_gen_length, truncation=True, padding='max_length').input_ids

        context_ids = dialog
        context_ids = context_ids[-self.args.gt_max_length:]
        context_ids = context_ids + [pad_token_id] * (self.args.gt_max_length - len(context_ids))
        label = label + [pad_token_id] * (self.args.max_gen_length - len(label))
        resp_batch = [token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in label]
        # resp_batch = label
        context_batch['input_ids'] = torch.LongTensor(context_ids)
        context_batch['attention_mask'] = torch.ne(context_batch['input_ids'], pad_token_id)
        context_batch['response'] = torch.LongTensor(resp_batch)

        context_batch['goal_idx'] = self.args.goalDic['str'][goal]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index로 바꿈
        # context_batch['predicted_goal_idx']

        if 'predicted_goal' in data:
            context_batch['predicted_goal_idx'] = self.args.goalDic['str'][data['predicted_goal'][0]]
        if 'predicted_topic' in data:
            context_batch['predicted_topic_idx'] = self.args.topicDic['str'][data['predicted_topic'][0]]

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                # context_batch[k] = torch.as_tensor(v)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


def convert_idx_to_docid(idx): return f"{idx}"


class KnowledgeDataset(Dataset):
    """ Knowledge Passage encoding --> Context
    """

    def __init__(self, args, knowledgeDB, tokenizer):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.know_max_length = args.know_max_length
        self.knowledgeDB = knowledgeDB
        self.data_samples = []

    def __getitem__(self, item):
        data = self.knowledgeDB[item]
        tokenized_data = self.tokenizer(data,
                                        max_length=self.know_max_length,
                                        padding='max_length',
                                        truncation=True,
                                        add_special_tokens=True)
        tokens = torch.LongTensor(tokenized_data.input_ids)
        mask = torch.LongTensor(tokenized_data.attention_mask)
        docid = self.tokenizer.encode(convert_idx_to_docid(item), truncation=True, padding='max_length', max_length=10)[1:-1]  # 이미 Tensor로 받음
        docid = torch.LongTensor(docid)
        return tokens, mask, docid

    def __len__(self):
        return len(self.knowledgeDB)


class DialogDataset(Dataset):
    """ Retrieve task를 학습하기위한 dialog dataset --> Question (query)
    """

    def __init__(self, args, data_sample, knowledgeDB, train_knowledgeDB, tokenizer, task, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.task = task
        self.tokenizer = tokenizer
        self.knowledgeDB = knowledgeDB
        self.train_knowledgeDB = train_knowledgeDB  # new knowledge 체크용
        self.augmented_raw_sample = data_sample
        self.mode = mode

    def negative_sampler(self, target_knowledge, candidate_knowledges):
        # candidate_entity = self.knowledgeDB[target_knowledge][0]
        # candiate_all_list = self.knowledgeDB_entity_values[candidate_entity]
        # negative_indice = random.choices(candiate_all_list, k=self.args.negative_num if len(candiate_all_list) > self.args.negative_num else len(candiate_all_list))
        # total_knowledge_num = self.args.knowledge_num

        negative_indice = []
        if len(candidate_knowledges) < self.args.negative_num:
            for idx in range(self.args.negative_num - len(candidate_knowledges)):
                negative_indice.append(0)

        while len(negative_indice) < self.args.negative_num:
            # negative_idx = random.randint(0, total_knowledge_num - 1)
            negative_idx = random.choice(candidate_knowledges)
            if (negative_idx not in negative_indice) and (negative_idx not in target_knowledge):
                negative_indice.append(negative_idx)
        return negative_indice

    def all_negative(self, candidate_knowledges):
        all_negative = [i for i in range(len(self.knowledgeDB))]
        for candidate in candidate_knowledges:
            all_negative.remove(candidate)
        return all_negative

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
        dialog, user_profile, response, goal, topic, situation, target_knowledge, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
        candidate_knowledges = [self.knowledgeDB.index(passage) for passage in candidate_knowledges]
        # candidate_confidences = min_max_norm(candidate_confidences)
        candidate_confidences = softmax(candidate_confidences)

        target_knowledge_idx = self.knowledgeDB.index(target_knowledge)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        context_batch = defaultdict()
        predicted_topic_list = deepcopy(data['predicted_topic'][:self.args.topk_topic])

        if self.mode == 'train':
            # predicted_goal, predicted_topic = goal, topic
            random.shuffle(predicted_topic_list)
            predicted_goal, predicted_topic = data['predicted_goal'][0], '|'.join(predicted_topic_list)
        else:
            predicted_goal = data['predicted_goal'][0]
            if data['predicted_topic_confidence'][0] > self.args.topic_conf:
                predicted_topic = data['predicted_topic'][0]
            else:
                predicted_topic = '|'.join(predicted_topic_list)

        if self.args.input_prompt == 'dialog':
            prefix = ''
        elif self.args.input_prompt == 'dialog_goal':
            prefix = '<goal>' + predicted_goal + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_topic':
            prefix = '<topic>' + predicted_topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_goal_topic':
            prefix = '<goal>' + predicted_goal + '<topic>' + predicted_topic + self.tokenizer.sep_token

        elif self.args.input_prompt == 'dialog_topic_profile':
            prefix = '<profile>' + user_profile + '<topic>' + predicted_topic + self.tokenizer.sep_token
        elif self.args.input_prompt == 'dialog_goal_profile':
            prefix = '<profile>' + user_profile + '<goal>' + predicted_goal + self.tokenizer.sep_token
        else:
            assert Exception

        prefix_encoding = self.tokenizer.encode(prefix)[1:-1][:30]
        input_sentence = self.tokenizer('<dialog>' + dialog, add_special_tokens=False).input_ids

        input_sentence = [self.tokenizer.cls_token_id] + prefix_encoding + input_sentence[-(self.args.max_length - len(prefix_encoding) - 1):]
        input_sentence = input_sentence + [pad_token_id] * (self.args.max_length - len(input_sentence))

        context_batch['input_ids'] = torch.LongTensor(input_sentence).to(self.args.device)
        attention_mask = context_batch['input_ids'].ne(pad_token_id)
        context_batch['attention_mask'] = attention_mask
        context_batch['response'] = self.tokenizer(response,
                                                   add_special_tokens=True,
                                                   max_length=self.args.max_length,
                                                   padding='max_length',
                                                   truncation=True).input_ids

        context_batch['goal_idx'] = self.args.goalDic['str'][goal]  # index로 바꿈
        context_batch['topic_idx'] = self.args.topicDic['str'][topic]  # index로 바꿈
        context_batch['topic'] = self.tokenizer(topic, truncation=True, padding='max_length', max_length=32).input_ids

        # List-wise
        # if target_knowledge_idx in candidate_knowledges:
        #     index = candidate_knowledges.index(target_knowledge_idx)
        #     candidate_knowledges = candidate_knowledges[:index] + candidate_knowledges[index+1:]
        #     candidate_confidences = candidate_confidences[:index] + candidate_confidences[index + 1:]
        # candidate_knowledges = [target_knowledge_idx] + candidate_knowledges
        # candidate_confidences = [100] + candidate_confidences

        # if target_knowledge_idx in candidate_knowledges:
        #     idxinlist = candidate_knowledges.index(target_knowledge_idx)
        #     candidate_knowledges.pop(idxinlist)
        #     candidate_knowledges.insert(0, target_knowledge_idx)
        #     target_confidence = candidate_confidences.pop(idxinlist)
        #     candidate_confidences.insert(0, target_confidence)
        candidate_confidences_pos = candidate_confidences[:self.args.pseudo_pos_num]
        candidate_knowledges_pos = candidate_knowledges[:self.args.pseudo_pos_num]
        # sampling_results = []
        # for i, data in enumerate(candidate_knowledges_pos):
        #     if i == 0:
        #         sampling_results.append(1)
        #     else:
        #         sampling_result = np.random.binomial(1, candidate_confidences_pos[i])
        #         sampling_results.append(sampling_result)

        # random_idx = random.randrange(min(self.args.pseudo_pos_num, len(candidate_knowledges)))

        ### Positive sampling (Top-K)
        # candidate_knowledges = candidate_knowledges[:self.args.pseudo_pos_num]
        # candidate_confidences = candidate_confidences[:self.args.pseudo_pos_num]
        # random_idx = random.randrange(self.args.pseudo_pos_num)
        # candidate_knowledges = [candidate_knowledges[random_idx]]
        # candidate_confidences = [candidate_confidences[random_idx]]

        ### Positive sampling (pseudo)
        # pseudo_candidate_knowledges = [know for know, conf in zip(candidate_knowledges, candidate_confidences) if conf > 0.1]
        # if len(pseudo_candidate_knowledges) == 0:
        #     candidate_knowledges = [candidate_knowledges[0]]
        #     candidate_confidences = [candidate_confidences[0]]
        # else:
        #     random_idx = random.randrange(len(pseudo_candidate_knowledges))
        #     candidate_knowledges = [pseudo_candidate_knowledges[random_idx]]
        #     candidate_confidences = [pseudo_candidate_knowledges[random_idx]]

        # # Only Top-K
        # candidate_knowledges = [candidate_knowledges[self.args.pseudo_pos_num]]
        # candidate_confidences = [candidate_confidences[self.args.pseudo_pos_num]]

        # candidate_knowledges = candidate_knowledges + [0] * (self.args.pseudo_pos_num - len(candidate_knowledges))
        # candidate_confidences = candidate_confidences + [0] * (self.args.pseudo_pos_num - len(candidate_confidences))

        pseudo_negative = self.negative_sampler(candidate_knowledges_pos, candidate_knowledges)
        # pseudo_negative = [] # todo: remove for rarank!!!!!!!!!!

        ### Grouping
        # group_num = min(self.args.pseudo_pos_rank, len(candidate_knowledges)) - 1
        # random_idx = sorted(random.sample(list(range(1, len(candidate_knowledges))), k=group_num))
        # candidate_knowledges_pos = [candidate_knowledges[0]] + [candidate_knowledges[idx] for idx in random_idx]
        # # candidate_confidences = [candidate_confidences[0]] + [candidate_confidences[idx] for idx in random_idx]
        # # candidate_confidences = (np.array(candidate_confidences) - min(candidate_confidences)) / (max(candidate_confidences) - min(candidate_confidences) + 1e-10)
        # candidate_confidences_pos = list(np.cumprod([0.8] * len(candidate_knowledges_pos)) / 0.8)

        # sampled_pair = sorted(random.sample(list(range(len(candidate_positives_idx))), k=2))
        # pseudo_positive = candidate_positives_idx[sampled_pair[0]]
        # pseudo_negative = candidate_positives_idx[sampled_pair[1]]

        # pseudo_positive = random.choice(candidate_positives_idx)
        # pseudo_positive = candidate_positives_idx[self.args.pseudo_pos_rank - 1]

        candidate_indice = candidate_knowledges_pos + pseudo_negative  # [candidate_positives_idx[self.args.pseudo_pos_rank]]

        candidate_knowledge_text = [self.knowledgeDB[idx] for idx in candidate_indice]
        candidate_knowledge = self.tokenizer(candidate_knowledge_text, truncation=True, padding='max_length', max_length=self.args.max_length)
        candidate_knowledge_token = candidate_knowledge.input_ids
        candidate_knowledge_mask = candidate_knowledge.attention_mask
        #
        context_batch['candidate_indice'] = candidate_indice
        context_batch['candidate_knowledge_token'] = candidate_knowledge_token
        context_batch['candidate_knowledge_mask'] = candidate_knowledge_mask

        context_batch['pseudo_targets'] = candidate_knowledges_pos  # [candidate_knowledges[0]]
        context_batch['pseudo_confidences'] = (candidate_confidences_pos > self.args.know_conf)  # + [-1e10] * (self.args.knowledge_num - len(candidate_confidences_pos))

        context_batch['target_knowledge'] = [target_knowledge_idx]  # candidate_knowledges[:3]  # target_knowledge_idx
        context_batch['all_negative'] = candidate_knowledges + self.all_negative(candidate_knowledges)
        context_batch['bm25_top20'] = candidate_knowledges
        context_batch['new_knowledge'] = self.knowledgeDB[target_knowledge_idx] not in self.train_knowledgeDB
        context_batch['isFood'] = (goal == 'Food recommendation')

        # context_batch['sampling_results'] = sampling_results

        context_batch['indices'] = idx
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
                # context_batch[k] = torch.as_tensor(v)
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)