import json
import os

import torch
from collections import defaultdict
import random
from datetime import datetime

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords
# import logging; logger = logging.getLogger(__name__)
from loguru import logger
stop_words = set(stopwords.words('english'))


def readDic(filename, out=None):
    output_idx_str = dict()
    output_idx_int = dict()
    with open(filename, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                k, idx = line.strip().split('\t')
            except:
                print(line)
                k, idx = line.strip().split()
            output_idx_str[k] = int(idx)
            output_idx_int[int(idx)] = k
        # output_idx_str[len(output_idx_str)] = '<PAD>'
        # output_idx_int[len(output_idx_int)] = '<PAD>'
    if out == 'str': return output_idx_str
    elif out == 'idx': return output_idx_int
    else: return {'str': output_idx_str, 'int': output_idx_int}


# def v2conv(conv, istopic=True, isgoal=True, iskg=True):
#     """
#     Args:
#         conv: v2lines[i]
#         istopic: text Topic 출력여부
#         isgoal: text type(goal)출력여부
#         iskg: text kg 출력여부
#     Returns: {'uttrs':uttrs, 'roles':roles, 'topics':topics, 'goals':goals, 'kgs':kgs, 'situation':situation, 'user_profile':usr_profile}
#     """
#     usr_profile = conv.get('user_profile')
#     situation = conv.get('situation')
#     topics = conv.get('goal_topic_list') if istopic else ["" for _ in range(len(conv.get('goal_type_list')))]
#     goals = conv.get('goal_type_list') if isgoal else ["" for _ in range(len(conv.get('goal_type_list')))]
#     kgs = conv.get('knowledge')  # if iskg else ["" for _ in range(len(conv.get('goal_type_list')))]
#     uttrs = [i if i[0] != '[' else i[4:] for i in conv.get('conversation')]  # utterance 내 [1] 과 같은 형태 삭제
#     roles = ["system", 'user'] if goals[0] == 'Greetings' else ['user', 'system']
#     for i in range(len(kgs) - 2): roles.append(roles[i % 2])
#     return {'uttrs': uttrs, 'roles': roles, 'topics': topics, 'goals': goals, 'kgs': kgs, 'situation': situation, 'user_profile': usr_profile}


def truncationPadding(input_ids, max_length, prefix=[], suffix=[]):
    truncate_size = max_length - len(prefix) - len(suffix)
    if truncate_size <= len(input_ids): input_ids = prefix + input_ids[len(input_ids) - truncate_size:] + suffix
    else: input_ids = prefix + input_ids + suffix
    return input_ids + [0] * (max_length - len(input_ids))


def user_profile_setting(ufDic: dict) -> str:
    uf = ''
    for k,v in ufDic.items():
        if isinstance(v,list): uf+=f" {k}: {', '.join(v)}|"
        elif isinstance(v, str):uf+=f" {k}: {v.replace(' ','')}|"
    return uf

def user_profile_setting_th(ufDic: dict) -> str: # OLD
    uf = ''
    for i, key in enumerate(ufDic.keys()):
        one = ufDic[key]
        if i == 0 or key[0].lower() != "a": pass
        else: uf += ' | '
        if type(one) == list: uf += f"{key}: {', '.join(one[:-5])}"
        else: uf += f"{key}: {one}"
    return uf

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def convert_know(know):
    if len(know) == 0: return ''
    if know[1] == 'Sings': know = ' '.join([know[0], 'singer', know[2]])
    elif know[1] == 'Stars': know = ' '.join([know[0], 'star', know[2]])
    elif know[1] == 'Intro': know = ' '.join([know[0], 'is', know[2]])
    elif know[1] == 'Comments': know = ' '.join([know[0], 'is known', know[2]])
    elif know[1] == 'Birthday': know = ' '.join([know[0], know[1], datetime.strptime(know[2].replace(' ', ''), '%Y-%m-%d').strftime('%Y %B %dth')])
    else: know = ' '.join(know)
    know = know.replace('℃', ' degrees Celsius')
    return know


def bm_tokenizer(text, tokenizer):
    # # 특정 구문을 임시 토큰으로 대체
    # for phrase in phrase_list:
    #     text = text.replace(phrase, phrase.replace(' ', '_'))
    #
    # # 기본 NLTK tokenizer를 사용하여 텍스트를 토큰화
    # tokens = nltk.word_tokenize(text)
    #
    # # 임시 토큰을 원래 구문으로 되돌리기
    # for i, token in enumerate(tokens):
    #     tokens[i] = token.replace('_', ' ')
    text = " ".join([word for word in text.split() if word not in stop_words])
    tokens = tokenizer.encode(text)[1:-1]

    return tokens


def process_augment_all_sample(raw_data, tokenizer, knowledgeDB):
    train_sample = []
    if tokenizer.eos_token is not None: eos_token = tokenizer.eos_token
    else: eos_token = tokenizer.sep_token
    for ij in range(len(raw_data)):
        conversation = raw_data[ij]
        augmented_dialog = []
        for i in range(len(conversation['dialog'])):
            role = conversation['role_seq'][i]
            utterance = conversation['dialog'][i] + eos_token
            goal = conversation['goal'][i]
            if role == 'System' and len(augmented_dialog) > 0:
                flatten_dialog = ''.join(augmented_dialog)
                train_sample.append({'dialog': flatten_dialog,
                                     'user_profile': conversation['user_profile'],
                                     'response': utterance,
                                     'goal': conversation['goal'][i],
                                     'topic': conversation['topic'][i],
                                     'situation': conversation['situation'],
                                     'target_knowledge': conversation['knowledge_seq'][i],
                                     'candidate_knowledges': conversation['pseudo_knowledge_seq'][i],
                                     'candidate_confidences': conversation['pseudo_confidence_seq'][i],  # prob
                                     })
            augmented_dialog.append(utterance)
    logger.info(f'All sample 들어감, Sample count: {len(train_sample)}')
    return train_sample


def process_augment_sample(raw_data, tokenizer=None, knowledgeDB=None, goal_list=['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'Food recommendation']):
    train_sample = []
    if tokenizer:
        if tokenizer.eos_token is not None: eos_token = tokenizer.eos_token
        else: eos_token = tokenizer.sep_token
    else: eos_token = '[SEP]'
    for ij in range(len(raw_data)):
        conversation = raw_data[ij]
        augmented_dialog = []
        for i in range(len(conversation['dialog'])):
            role = conversation['role_seq'][i]
            utterance = conversation['dialog'][i] + eos_token
            goal = conversation['goal'][i]
            if goal in goal_list:
                if role == 'System' and len(augmented_dialog) > 0 and len(conversation['pseudo_knowledge_seq'][i]) != 0: # Test 3360 Setting
                    flatten_dialog = ''.join(augmented_dialog)
                    train_sample.append({'dialog': flatten_dialog,
                                         'user_profile': conversation['user_profile'],
                                         'response': utterance,
                                         'goal': conversation['goal'][i],
                                         'last_goal': conversation['goal'][i-1],
                                         'topic': conversation['topic'][i],
                                         'situation': conversation['situation'],
                                         'target_knowledge': conversation['knowledge_seq'][i],
                                         'candidate_knowledges': conversation['pseudo_knowledge_seq'][i],
                                         'candidate_confidences': conversation['pseudo_confidence_seq'][i]  # prob
                                         })
            augmented_dialog.append(utterance)
    logger.info(f"Pseudo_knowledge_seq, goal_list있는 sample만 들어감, Sample count: {len(train_sample)}, Goal list: {goal_list}")
    return train_sample


def dataset_reader(args, data_name='train'):
    all_knowledge = set()
    all_knowledge_topic = []
    conversation_sample = []
    data_path = os.path.join(args.data_dir, f"en_{data_name}_know_cand_score20_new.txt")
    with open(data_path, 'r', encoding='UTF-8') as f:
        line_idx=0
        for line in tqdm(f, desc="Dataset Read", bar_format='{l_bar} | {bar:23} {r_bar}'):
            line_idx+=1
            if args.debug and line_idx>30: break
            dialog = json.loads(line)
            conversation = dialog['conversation']
            role_seq = ["User", "System"] if dialog['goal_type_list'][0] != 'Greetings' else ["System", "User"]

            for i in range(2, len(conversation)):
                role_seq.append(role_seq[i % 2])

            knowledge_seq = dialog['knowledge']
            know_candidates = dialog['know_candidates']
            pseudo_knowledge_seq = []
            pseudo_confidence_seq = []
            for idx, know_conf_list in enumerate(know_candidates):
                positive_candidates = [know[0] for know in know_conf_list]
                knowledge_topic = [args.topicDic['str'][candidate[0]] if candidate[0] in args.topicDic else 0 for candidate in positive_candidates]
                positive_candidates = [convert_know(candidate) for candidate in positive_candidates]

                conf_list = [know[1] for know in know_conf_list]
                pseudo_knowledge_seq.append(positive_candidates)
                pseudo_confidence_seq.append(conf_list)
                # if len(positive_candidates) > 0:
                #     positive_candidates_list[idx] = [' '.join(candidate) for candidate in positive_candidates]
                #     # positive_candidates_list[idx] = [args.knowledgeDB.index(candidate) for candidate in positive_candidates]

            knowledge_seq = [convert_know(know) for know in knowledge_seq]
            all_knowledge.update(knowledge_seq)
            # pseudo_knowledge_seq = [' '.join(know) for know in pseudo_knowledge_seq]
            # for topic, know in zip(dialog['goal_topic_list'], knowledge_seq):
            #     if (topic, know) not in all_knowledge_topic and know != '':
            #         all_knowledge_topic.append((topic, know))

            user_profile = user_profile_setting(dialog['user_profile'])
            situation = dialog['situation']

            for i in range(len(conversation)):  # HJ: [1],[2] 같은 text 제거, conversation 추가해넣는코드
                conversation[i] = conversation[i] if conversation[i][0] != '[' else conversation[i][4:]
                conversation[i] = role_seq[i] + ": " + conversation[i]
            conversation_sample.append({
                'dialog': conversation,
                'role_seq': role_seq,
                'goal': dialog['goal_type_list'],
                'topic': dialog['goal_topic_list'],
                'situation': situation,
                'user_profile': user_profile,
                'knowledge_seq': knowledge_seq,
                'pseudo_knowledge_seq': pseudo_knowledge_seq,
                'pseudo_confidence_seq': pseudo_confidence_seq
            })

    return conversation_sample, list(all_knowledge), all_knowledge_topic
