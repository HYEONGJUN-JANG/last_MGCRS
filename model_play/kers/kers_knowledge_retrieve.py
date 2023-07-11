import sys
import os
import torch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset  # , RandomSampler, SequentialSampler
from tqdm import tqdm
from collections import defaultdict
from loguru import logger


def train_test_pseudo_knowledge_bart(args, model, tokenizer, train_dataset_aug, test_dataset_aug, train_knowledge_seq_set, test_knowledge_seq_set):
    """
    Train시 Input: Dialog + goal + (knowledge) || label: Pseudo label
    Test시 Input: 동일 || label: gold label
    """
    args.data_mode = 'train'
    logger.info("Train Bart")
    logger.info(f'Train aug dataset len: {len(train_dataset_aug)}, Train knowledge sequence len: {len(train_knowledge_seq_set)}')  # TH 230601
    logger.info(f'Test aug dataset len: {len(test_dataset_aug)}, Test knowledge sequence len: {len(test_knowledge_seq_set)}')  # TH 230601

    train_datamodel_resp = GenerationDataset(args, train_dataset_aug, train_knowledge_seq_set, tokenizer, mode='train')
    test_datamodel_resp = GenerationDataset(args, test_dataset_aug, train_knowledge_seq_set, tokenizer, mode='test')
    train_data_loader = DataLoader(train_datamodel_resp, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_datamodel_resp, batch_size=args.batch_size, shuffle=False)

    # train_data_loader = DataLoader(train_dataset_aug, batch_size=args.batch_size, shuffle=True)
    # test_data_loader = DataLoader(test_dataset_aug, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4, eps=5e-9)
    beam_temp = args.num_beams
    args.num_beams = 1
    logger.info(f"Train with Pseudo knowledge label: {args.usePseudoTrain}, Test with Pseudo knowledge label: {args.usePseudoTest}")
    # Fine-tune
    for epoch in range(args.num_epochs):
        if epoch == args.num_epochs - 1: args.num_beams = beam_temp
        train_loss, dev_loss, test_loss = 0, 0, 0
        args.data_mode = 'train'
        context_words, pred_words, label_words = [], [], []
        new_knows = []
        model.train()
        torch.cuda.empty_cache()
        for batch in tqdm(train_data_loader, desc=f"Epoch {epoch}__{args.data_mode}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            dialog = torch.as_tensor(batch['knowledge_task_input'], device=args.device)
            knowledge = torch.as_tensor(batch['knowledge_task_label'], device=args.device)  # Golden label
            new_knows.extend([i for i in batch['is_new_knowledge']])
            ### For Pseudo Knowledge 학습 --> Target label로 Scoring
            # if args.usePseudoLabel: # Train시 Pseudo label 사용하도록 전면 수정
            # else: outputs = model(input_ids=dialog, labels = knowledge, output_hidden_states=True)
            if args.usePseudoTrain: knowledge = torch.as_tensor(batch['knowledge_task_pseudo_label'], device=args.device)  # Pseudo label
            outputs = model(input_ids=dialog, labels=knowledge, output_hidden_states=True)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            loss.detach()
            # context_word, label_word, pred_word = decode_withBeam(args, model=model, tokenizer=tokenizer, input=dialog, label=pseudo_knowledge, num_beams=args.num_beams)

            # context_words.extend(context_word)
            # pred_words.extend(pred_word)
            # label_words.extend(label_word)

        # print(train_loss, f", New Knowledge Count in Train: {sum(new_knows)}")
        # p, r, f = know_f1_score(args, pred_words, label_words)
        # print(f"{args.data_mode} Knowledge P/R/F1", p, r, f)

        # hit1, hit3, hit5 = know_hit_ratio(args, pred_words, label_words)
        # print(f"{args.data_mode} Knowledge hit / hit_hr: {hit1}, {hit3}, {hit5}")

        # # logger.info(f'Epoch_{epoch} Train loss: {train_loss}, Samples: {len(context_words)}')
        # # logger.info(f"Epoch_{epoch}_{args.data_mode} Knowledge P/R/F1/Hit@1/Hit@{args.num_beams}: {p}, {r}, {f}, {hit1}, {hit3}, {hit5} \t Train loss: {train_loss}, Samples: {len(context_words)}")
        # logger.info(f"Epoch_{epoch}_{args.data_mode} Knowledge Hit@1/Hit@3/Hit@5: {hit1}, {hit3}, {hit5} \t Train loss: {train_loss:.3f}, Samples: {len(context_words)}")
        # save_preds(args, context_words, pred_words, label_words, epoch)

        args.data_mode = 'test'
        model.eval()
        torch.cuda.empty_cache()
        new_knows = []
        context_words, pred_words, label_words = [], [], []
        with torch.no_grad():
            for batch in tqdm(test_data_loader, desc=f"Epoch {epoch}__{args.data_mode}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
                dialog = torch.as_tensor(batch['knowledge_task_input'], device=args.device)
                knowledge = torch.as_tensor(batch['knowledge_task_label'], device=args.device)  ##
                new_knows.extend([int(i) for i in batch['is_new_knowledge']])
                ### For Pseudo Knowledge 학습 --> Target label로 Scoring
                if args.usePseudoTest: knowledge = torch.as_tensor(batch['knowledge_task_pseudo_label'], device=args.device)
                # Goal label로 Test하도록 (230704 이후)
                outputs = model(input_ids=dialog, labels=knowledge, output_hidden_states=True)
                loss = outputs.loss
                test_loss += loss.item()
                # HJ: Beam 사이즈 조절하여 decode할 수 있도록 코드 수정 (train, test) model, tokenizer, input, label, num_beams
                context_word, label_word, pred_word = decode_withBeam(args, model=model, tokenizer=tokenizer, input=dialog, label=knowledge, num_beams=args.num_beams)

                context_words.extend(context_word)
                pred_words.extend(pred_word)
                label_words.extend(label_word)

        # print(test_loss, f", New Knowledge Count in Test: {sum(new_knows)}")
        # p, r, f = utils.know_f1_score(args, pred_words, label_words)
        # print(f"{args.data_mode} Knowledge P/R/F1: ", p, r, f)

        hit1, hit3, hit5, hit1_new, hit3_new, hit5_new = know_hit_ratio(args, pred_words, label_words, new_knows)
        print(f"{args.data_mode} Knowledge hit / hit_k: {hit1}, {hit3}, {hit5}")
        print(f"{args.data_mode} New_Knowledge hit / hit_k: {hit1_new}, {hit3_new}, {hit5_new}")

        logger.info(f'Epoch_{epoch} Test loss: {test_loss}, Samples: {len(context_words)}')
        logger.info(f"Epoch_{epoch}_{args.data_mode}  Knowledge     Hit@1/Hit@3/Hit@5: {hit1}, {hit3}, {hit5} \t ")
        logger.info(f"Epoch_{epoch}_{args.data_mode}  New Knowledge Hit@1/Hit@3/Hit@5: {hit1_new}, {hit3_new}, {hit5_new} \t New Knowledge Count: {sum(new_knows)}")
        save_preds(args, context_words, pred_words, label_words, epoch, new_knows)


def decode_withBeam(args, model, tokenizer, input, label, num_beams=1):
    # TODO : beam 2 이상에서 debug 필요
    context_word = tokenizer.batch_decode(input)
    label_word = tokenizer.batch_decode(label)
    summary_ids = model.generate(input, num_return_sequences=num_beams, num_beams=num_beams, min_length=0, max_length=args.max_gen_length, early_stopping=True)
    pred_words = tokenizer.batch_decode(summary_ids)
    if num_beams > 1:
        output = [pred_words[i * num_beams:(i + 1) * num_beams] for i in range(len(input))]
    else:
        output = pred_words
    return context_word, label_word, output


class GenerationDataset(Dataset):  # knowledge용 데이터셋
    def __init__(self, args, data_sample, train_knowledge_seq_set, tokenizer, mode='train'):
        super(Dataset, self).__init__()
        self.args = args
        self.train_knowledge_seq_set = train_knowledge_seq_set
        self.tokenizer = tokenizer
        self.augmented_raw_sample = data_sample
        self.mode = mode
        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))
        self.n_related_knowledge = 20
        self.kers_retrieve_input_length = args.kers_retrieve_input_length if 'kers_retrieve_input_length' in args else 768  # KERS Default: 768

    def __getitem__(self, idx):  # TODO 구현 전
        data = self.augmented_raw_sample[idx]
        # cbdicKeys = ['dialog', 'user_profile', 'situation', 'response', 'goal', 'last_goal', 'topic', 'related_knowledges', 'augmented_knowledges', 'target_knowledge', 'candidate_knowledges']
        # dialog, user_profile, situation, response, type, last_type, topic, related_knowledges, augmented_knowledges, target_knowledge, candidate_knowledges = [data[i] for i in cbdicKeys]
        cbdicKeys = [ \
            'dialog', 'user_profile', 'situation', 'response', 'goal', 'last_goal', 'topic', 'target_knowledge', 'candidate_knowledges', 'predicted_goal', 'predicted_topic']
        dialog, user_profile, situation, response, type, last_type, topic, target_knowledge, candidate_knowledges, predicted_goal, predicted_topic = [data[i] for i in cbdicKeys]
        pad_token_id = self.tokenizer.pad_token_id
        type = data['predicted_goal'][0]
        context_batch = defaultdict()

        # ## Related knowledge 범위 관련 세팅##
        # related_knowledge_text = ""
        # if self.args.n_candidate_knowledges > 0:  ## Pseudo related knowledge 줄 때 (약 60%로 정답포함)
        #     related_knowledge_text = " | ".join(list(filter(lambda x: x, candidate_knowledges[:self.args.n_candidate_knowledges])))
        # else:  ## candidate knowledge에서 사용할 때,
        related_knowledge_text = " | ".join(list(filter(lambda x: x, candidate_knowledges)))

        ## Related knowledge 범위 관련 세팅## -- 1. 해당대화전체knowledge, 2. 해당응답전까지knowledge 비교
        # related_knowledge를 candidate_knowledges에서 25개 무작위로 뽑아서 넣어줘서 돌리기

        related_knowledge_text += situation  # situation
        related_knowledge_text += user_profile  # user_profile

        max_knowledge_length = self.kers_retrieve_input_length * 5 // 10  # 768의 50%까지 knowledge데이터 넣어주기
        related_knowledge_tokens = self.tokenizer('<knowledge>' + related_knowledge_text, max_length=max_knowledge_length, truncation=True).input_ids

        type_token = self.tokenizer('<type>' + type, max_length=max_knowledge_length // 20, truncation=True).input_ids
        last_type_token = self.tokenizer('<last_type>' + last_type, max_length=max_knowledge_length // 20, truncation=True).input_ids
        topic_token = self.tokenizer('<topic>' + topic, max_length=max_knowledge_length // 20, truncation=True).input_ids

        if self.args.inputWithKnowledge:
            if self.args.inputWithTopic:
                input = self.tokenizer('<dialog>' + dialog, max_length=self.kers_retrieve_input_length - len(related_knowledge_tokens) - len(type_token) - len(topic_token), padding='max_length', truncation=True).input_ids
                input = related_knowledge_tokens + input + type_token + topic_token  # {TH} knowledge 를 input에서 빼보는 ablation 적용을 위해 주석 (윗줄포함)
            else:  # Original KERS (With knowledge) Setting (Related knowledges 가 존재할 때 --> 우리 방식으로 RelKnow만들어주고 2stage결과 보려고할때)
                input = self.tokenizer('<dialog>' + dialog, max_length=self.kers_retrieve_input_length - len(related_knowledge_tokens) - len(type_token) - len(last_type_token), padding='max_length', truncation=True).input_ids
                input = related_knowledge_tokens + input + type_token + last_type_token  # {TH} knowledge 를 input에서 빼보는 ablation 적용을 위해 주석 (윗줄포함)

        else:  # KEMGCRS세팅과 같은 input (dialog + type + topic으로 knowledge예측)
            if self.args.inputWithTopic:  ## KEMGCRS와 같은 input구성일때
                input = self.tokenizer('<dialog>' + dialog, max_length=self.kers_retrieve_input_length - len(type_token) - len(last_type_token) - len(topic_token), padding='max_length', truncation=True).input_ids
                input = input + type_token + last_type_token + topic_token
            else:  # Original KERS (Without knowledges) Setting (Related knowledges 가 없을 때 --> E4의 KERS(w/o RelKnow))
                input = self.tokenizer('<dialog>' + dialog, max_length=self.kers_retrieve_input_length - len(type_token) - len(last_type_token), padding='max_length', truncation=True).input_ids
                input = input + type_token + last_type_token

        # Train시 label: pseudo knowledge top 1
        pseudo_label = self.tokenizer('<knowledge>' + candidate_knowledges[0], max_length=self.args.max_gen_length, padding='max_length', truncation=True).input_ids
        # Test시 label: gold knowledge
        label = self.tokenizer('<knowledge>' + target_knowledge, max_length=self.args.max_gen_length, padding='max_length', truncation=True).input_ids

        # <s> knowledge text~~ </s> 로 EOS가 제대로 들어가짐
        context_batch['knowledge_task_input'] = torch.LongTensor(input)
        context_batch['attention_mask'] = torch.ne(context_batch['knowledge_task_input'], pad_token_id)
        context_batch['knowledge_task_label'] = torch.LongTensor(label)
        context_batch['knowledge_task_pseudo_label'] = torch.LongTensor(pseudo_label)
        context_batch['is_new_knowledge'] = 1 if target_knowledge not in self.train_knowledge_seq_set else 0
        return context_batch

    def __len__(self):
        return len(self.augmented_raw_sample)


def know_hit_ratio(args, pred_pt, gold_pt, new_knows=None):
    # TODO: Beam처리
    hit1, hit3, hit5 = 0, 0, 0
    hit1_new, hit3_new, hit5_new = 0, 0, 0

    for idx in range(len(gold_pt)):
        pred, gold = pred_pt[idx], gold_pt[idx]
        if args.num_beams > 1:
            if gold == pred[0]: hit1 += 1
            if gold in pred[:3]: hit3 += 1
            if gold in pred: hit5 += 1
        else:
            if gold == pred: hit1 += 1

        if new_knows:
            new = new_knows[idx]
            if args.num_beams > 1:
                if new and gold == pred[0]: hit1_new += 1
                if new and gold in pred[:3]: hit3_new += 1
                if new and gold in pred: hit5_new += 1
            else:
                if new and gold == pred: hit1_new += 1
    if new_knows:
        return round(hit1 / len(gold_pt), 4), round(hit3 / len(gold_pt), 4), round(hit5 / len(gold_pt), 4), round(hit1_new / len(gold_pt), 4), round(hit3_new / len(gold_pt), 4), round(hit5_new / len(gold_pt), 4)
    else:
        return round(hit1 / len(gold_pt), 4), round(hit3 / len(gold_pt), 4), round(hit5 / len(gold_pt), 4)


def save_preds(args, context, pred_words, label_words, epoch=None, new_knows=None):
    # HJ: 동일 파일 덮어쓰면서 맨 윗줄에 몇번째 에폭인지만 쓰도록 수정
    mode = args.data_mode
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    # log_file_path = os.path.join(args.home, 'output',str(args.version), args.method, args.time+'_'+ args.log_name)
    # checkPath(log_file_path)
    log_file_name = f"{str(epoch)}_{mode}_output.txt"
    path = os.path.join(args.output_dir, log_file_name)
    print(f"Save {args.task}, Epoch: {str(epoch)}, Mode: {mode} generated results in {path}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{args.task}, Epoch: {str(epoch)} Input and Output results {args.time}\n")
        f.write(f"Log File Name: {args.log_name} \n")
        for i, (ctx, pred, label) in enumerate(zip(context, pred_words, label_words)):
            if i == 500: break
            f.write(f"Source: {ctx}\n")
            if new_knows: f.write(f"Is_New_Knows: {new_knows[i]}\n")
            f.write(f"Pred : {pred}\n")
            f.write(f"Label: {label}\n")
            f.write(f"\n")
    return
