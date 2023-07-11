from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import write_pkl, save_json
import numpy as np
import pickle
from loguru import logger


def knowledge_reindexing(args, knowledge_data, retriever, stage):
    # 모든 know_index를 버트에 태움
    logger.info('...knowledge indexing...(%s)' % stage)
    retriever.eval()
    knowledgeDataLoader = DataLoader(
        knowledge_data,
        batch_size=args.batch_size
    )
    knowledge_index = []

    for batch in tqdm(knowledgeDataLoader, bar_format=' {l_bar} | {bar:23} {r_bar}'):
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        if stage == 'retrieve':
            knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        elif stage == 'rerank':
            knowledge_emb = retriever.rerank_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]

        # knowledge_emb = retriever.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, d]
        # knowledge_emb = torch.sum(knowledge_emb * attention_mask.unsqueeze(-1), dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-20)  # [B, d]

        knowledge_index.extend(knowledge_emb.cpu().detach())
    knowledge_index = torch.stack(knowledge_index, 0)
    return knowledge_index


def eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer, write=None, retrieve=None, data_type='test'):
    logger.info(args.stage)
    retriever.eval()
    # Read knowledge DB
    # knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
    # knowledge_index = knowledge_index.to(args.device)
    jsonlineSave = []
    # bert_model = bert_model.to(args.device)
    new_cnt = 0
    logger.info('Knowledge indexing for test')
    knowledge_index = knowledge_reindexing(args, knowledge_data, retriever, stage='retrieve')
    knowledge_index = knowledge_index.to(args.device)

    if args.stage == 'rerank':
        knowledge_index_rerank = knowledge_reindexing(args, knowledge_data, retriever, stage='rerank')
        knowledge_index_rerank = knowledge_index_rerank.to(args.device)

    goal_list = ['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'Food recommendation', 'Chat about stars']
    hit1_goal, hit3_goal, hit5_goal, hit10_goal, hit20_goal = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    hit1, hit5, hit3, hit10, hit20 = [], [], [], [], []
    hit1_new, hit5_new, hit3_new, hit10_new, hit20_new = [], [], [], [], []

    hit20_p1, hit20_p2, hit20_p3, hit20_p23 = [], [], [], []

    cnt = 0

    pred = []
    targets = []
    current = 0
    for batch in tqdm(test_dataloader, desc="Knowledge_Test", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):  # TODO: Knowledge task 분리중
        batch_size = batch['attention_mask'].size(0)
        dialog_token = batch['input_ids']
        dialog_mask = batch['attention_mask']
        response = batch['response']
        new_knowledge = batch['new_knowledge']

        # candidate_indice = batch['candidate_indice']
        # candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,5,256]
        # candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]

        goal_idx = [args.goalDic['int'][int(idx)] for idx in batch['goal_idx']]
        topic_idx = [args.topicDic['int'][int(idx)] for idx in batch['topic_idx']]

        # candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,5,256]
        target_knowledge_idx = batch['target_knowledge']

        # if args.stage == 'retrieve':
        dot_score = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, batch['goal_idx'])

        if args.stage == 'rerank':
            # candidate_indice = torch.topk(dot_score, k=args.know_topk, dim=1).indices
            # dot_score = retriever.compute_know_score_candidate(dialog_token, dialog_mask, knowledge_index_rerank[candidate_indice]) # todo: 2-stage
            dot_score = retriever.compute_know_score_candidate(dialog_token, dialog_mask, knowledge_index_rerank)  # todo: DPR용 (1/2)

            # dot_score = torch.gather(dot_score, 1, candidate_indice)

            # candidate_indice = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]
            # candidate_knowledge_text = [args.knowledgeDB[idx] for candidates in candidate_indice for idx in candidates]
            # candidate_knowledge = tokenizer(candidate_knowledge_text, truncation=True, padding='max_length', max_length=args.max_length)
            # candidate_knowledge_token = candidate_knowledge.input_ids
            # candidate_knowledge_mask = candidate_knowledge.attention_mask
            # candidate_knowledge_token = torch.LongTensor(candidate_knowledge_token).to(args.device).view(-1, args.know_topk, args.max_length)
            # candidate_knowledge_mask = torch.LongTensor(candidate_knowledge_mask).to(args.device).view(-1, args.know_topk, args.max_length)
            # dot_score = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]

        # if retrieve:
        #     top_candidate = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]
        #     for idx in range(batch_size):
        #         # query_token = tokenizer.encode(tokenizer.decode(response[idx], skip_special_tokens=True) + "|" + topic_idx[idx])[1:-1]
        #         # bm_scores = args.bm25.get_scores(query_token)
        #         # candidate_list = top_candidate.cpu().numpy().tolist()[idx]
        #         # candidate_list_score = [bm_scores[candi] for candi in candidate_list]
        #         # sorted_list_idx = np.argsort(candidate_list_score)[::-1]
        #         # sorted_candidates = [candidate_list[idx] for idx in sorted_list_idx]
        #         # test_dataloader.dataset.augmented_raw_sample[current + idx]['candidate_knowledges'] = sorted_candidates
        #         test_dataloader.dataset.augmented_raw_sample[current + idx]['target_knowledge'] = knowledgeDB[target_knowledge_idx[idx]]
        #         test_dataloader.dataset.augmented_raw_sample[current + idx]['candidate_knowledges'] = [knowledgeDB[rank] for rank in top_candidate[idx].cpu().tolist()]

        #     current += batch_size

        if write:
            top_candidate = torch.topk(dot_score, k=args.know_topk, dim=1).indices  # [B, K]
            input_text = '||'.join(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))
            target_knowledge_text = knowledgeDB[target_knowledge_idx]
            retrieved_knowledge_text = [knowledgeDB[idx].lower() for idx in top_candidate[0]]  # list
            correct = target_knowledge_idx in top_candidate

            response = '||'.join(tokenizer.batch_decode(response, skip_special_tokens=True))
            query = topic_idx[0] + "|" + response
            bm_scores = args.bm25.get_scores(bm_tokenizer(query, tokenizer))
            retrieved_knowledge_score = bm_scores[top_candidate[0].cpu().numpy()]
            jsonlineSave.append({'goal_type': goal_idx[0], 'topic': topic_idx[0], 'tf': correct, 'dialog': input_text, 'target': target_knowledge_text, 'response': response, "predict5": retrieved_knowledge_text, "score5": retrieved_knowledge_score})
            # save_json(args, f"{args.time}_{args.model_name}_inout", jsonlineSave)

        for idx, (score, target, pseudo_targets, goal, new) in enumerate(zip(dot_score, target_knowledge_idx, batch['pseudo_targets'], goal_idx, new_knowledge)):
            # goal = args.goalDic['int'][int(goal_idx)]
            # top_candidate = torch.topk(score, k=args.know_topk, dim=0).indices  # [K]
            # candidate_knowledge_text = [args.knowledgeDB[int(idx)] for idx in top_candidate]  # [K, .]
            # candidate_knowledge = tokenizer(candidate_knowledge_text, truncation=True, padding='max_length', max_length=args.max_length, return_tensors='pt')
            # candidate_knowledge_token = candidate_knowledge.input_ids.to(args.device)  # [K, L]
            # candidate_knowledge_mask = candidate_knowledge.attention_mask.to(args.device)  # [K, L]
            # re_rank_score = retriever.knowledge_retrieve(dialog_token[idx].unsqueeze(0), dialog_mask[idx].unsqueeze(0), candidate_knowledge_token.unsqueeze(0), candidate_knowledge_mask.unsqueeze(0)).squeeze(0)  # [K]
            # pseudo_targets1 = pseudo_targets[0]
            # pseudo_targets2 = pseudo_targets[1]
            # pseudo_targets3 = pseudo_targets[2]

            if new:
                new_cnt += 1
            for k in [1, 3, 5, 10]:

                top_candidate = torch.topk(score, k=k).indices
                # if args.stage == 'rerank': # todo: DPR시 주석처리 / 2-stage 시 roll-back(2/2)
                #     top_candidate = torch.gather(candidate_indice[idx], 0, top_candidate)
                correct_k = False
                for t in target:
                    correct_k |= (t in top_candidate)
                if k == 1:
                    hit1.append(correct_k)
                    hit1_goal[goal].append(correct_k)
                    if new:
                        hit1_new.append(correct_k)
                elif k == 3:
                    hit3.append(correct_k)
                    hit3_goal[goal].append(correct_k)
                    if new:
                        hit3_new.append(correct_k)
                elif k == 5:
                    hit5.append(correct_k)
                    hit5_goal[goal].append(correct_k)
                    if new:
                        hit5_new.append(correct_k)
                elif k == 10:
                    hit10.append(correct_k)
                    hit10_goal[goal].append(correct_k)
                    if new:
                        hit10_new.append(correct_k)
                elif k == 20:
                    hit20.append(correct_k)
                    hit20_goal[goal].append(correct_k)
                    if new:
                        hit20_new.append(correct_k)

                # correct_k = pseudo_targets1 in top_candidate
                # if k == 20:
                #     hit20_p1.append(correct_k)
                #
                # correct_k = pseudo_targets2 in top_candidate
                # if k == 20:
                #     hit20_p2.append(correct_k)
                #
                # correct_k = pseudo_targets3 in top_candidate
                # if k == 20:
                #     hit20_p3.append(correct_k)
                #
                # correct_k = (pseudo_targets2 in top_candidate) or (pseudo_targets3 in top_candidate)
                # if k == 20:
                #     hit20_p23.append(correct_k)
    # for i in range(10):
    #     logger.info("T:%s\tP:%s" %(targets[i], pred[i]))

    # topic_eval(targets, pred)
    hit1 = np.average(hit1)
    hit3 = np.average(hit3)
    hit5 = np.average(hit5)
    hit10 = np.average(hit10)
    hit20 = np.average(hit20)

    hit1_new = np.average(hit1_new)
    hit3_new = np.average(hit3_new)
    hit5_new = np.average(hit5_new)
    hit10_new = np.average(hit10_new)
    hit20_new = np.average(hit20_new)

    # hit20_p1 = np.average(hit20_p1)
    # hit20_p2 = np.average(hit20_p2)

    hit_movie_result = [np.average(hit1_goal["Movie recommendation"]), np.average(hit3_goal["Movie recommendation"]), np.average(hit5_goal["Movie recommendation"]), np.average(hit10_goal["Movie recommendation"]), np.average(hit20_goal["Movie recommendation"])]
    hit_music_result = [np.average(hit1_goal["Music recommendation"]), np.average(hit3_goal["Music recommendation"]), np.average(hit5_goal["Music recommendation"]), np.average(hit10_goal["Music recommendation"]), np.average(hit20_goal["Music recommendation"])]
    hit_qa_result = [np.average(hit1_goal["Q&A"]), np.average(hit3_goal["Q&A"]), np.average(hit5_goal["Q&A"]), np.average(hit10_goal["Q&A"]), np.average(hit20_goal["Q&A"])]
    hit_poi_result = [np.average(hit1_goal["POI recommendation"]), np.average(hit3_goal["POI recommendation"]), np.average(hit5_goal["POI recommendation"]), np.average(hit10_goal["POI recommendation"]), np.average(hit20_goal["POI recommendation"])]
    hit_food_result = [np.average(hit1_goal["Food recommendation"]), np.average(hit3_goal["Food recommendation"]), np.average(hit5_goal["Food recommendation"]), np.average(hit10_goal["Food recommendation"]), np.average(hit20_goal["Food recommendation"])]
    hit_chat_result = [np.average(hit1_goal["Chat about stars"]), np.average(hit3_goal["Chat about stars"]), np.average(hit5_goal["Chat about stars"]), np.average(hit10_goal["Chat about stars"]), np.average(hit20_goal["Chat about stars"])]

    hit_movie_result = ["%.4f" % hit for hit in hit_movie_result]
    hit_music_result = ["%.4f" % hit for hit in hit_music_result]
    hit_qa_result = ["%.4f" % hit for hit in hit_qa_result]
    hit_poi_result = ["%.4f" % hit for hit in hit_poi_result]
    hit_food_result = ["%.4f" % hit for hit in hit_food_result]
    hit_chat_result = ["%.4f" % hit for hit in hit_chat_result]

    if retrieve:
        with open(f'augmented_dataset_{data_type}.txt', 'wb') as f:
            pickle.dump(test_dataloader.dataset.augmented_raw_sample, f)

    if write:
        # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
        write_pkl(obj=jsonlineSave, filename='jsonline.pkl')  # 입출력 저장
        save_json(args, f"{args.time}_{args.model_name}_inout", jsonlineSave)
    else:
        logger.info(f"Test Hit@1: %.4f" % np.average(hit1))
        logger.info(f"Test Hit@3: %.4f" % np.average(hit3))
        logger.info(f"Test Hit@5: %.4f" % np.average(hit5))
        logger.info(f"Test Hit@10: %.4f" % np.average(hit10))
        logger.info(f"Test Hit@20: %.4f" % np.average(hit20))

        logger.info(f"Test New Hit@1: %.4f" % np.average(hit1_new))
        logger.info(f"Test New Hit@3: %.4f" % np.average(hit3_new))
        logger.info(f"Test New Hit@5: %.4f" % np.average(hit5_new))
        logger.info(f"Test New Hit@10: %.4f" % np.average(hit10_new))
        logger.info(f"Test New Hit@20: %.4f" % np.average(hit20_new))

        # logger.info(f"Test Hit@20_P1: %.4f" % np.average(hit20_p1))
        # logger.info(f"Test Hit@20_P2: %.4f" % np.average(hit20_p2))

        logger.info("Movie recommendation\t" + "\t".join(hit_movie_result))
        logger.info("Music recommendation\t" + "\t".join(hit_music_result))
        logger.info("Q&A\t" + "\t".join(hit_qa_result))
        logger.info("POI recommendation\t" + "\t".join(hit_poi_result))
        logger.info("Food recommendation\t" + "\t".join(hit_food_result))
        logger.info("Chat about stars\t" + "\t".join(hit_chat_result))

    logger.info("new knowledge %d" % new_cnt)
    return [hit1, hit3, hit5, hit10, hit20, hit_movie_result, hit_music_result, hit_qa_result, hit_poi_result, hit_food_result, hit_chat_result, hit1_new, hit3_new, hit5_new, hit10_new, hit20_new]


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