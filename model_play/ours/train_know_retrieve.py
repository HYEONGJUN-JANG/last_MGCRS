import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
# from data_temp import DialogDataset_TEMP
from model_play.ours.eval_know_retrieve import knowledge_reindexing, eval_know #### Check
from metric import EarlyStopping
from utils import *
# from models import *
import logging
import numpy as np
from loguru import logger


def update_key_bert(key_bert, query_bert):
    logger.info('update moving average')
    decay = 0  # If 0 then change whole parameter
    for current_params, ma_params in zip(query_bert.parameters(), key_bert.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = decay * old_weight + (1 - decay) * up_weight


def train_know(args, train_dataloader, test_dataloader, retriever, knowledge_data, knowledgeDB, all_knowledge_data, all_knowledgeDB, tokenizer):
    criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
    optimizer = optim.AdamW(retriever.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    # eval_know(args, test_dataloader, retriever, knowledge_data, knowledgeDB, tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

    best_hit = [[], [], [], [], []]
    best_hit_new = [[], [], [], [], []]

    best_hit_movie = [[], [], [], [], []]
    best_hit_poi = [[], [], [], [], []]
    best_hit_music = [[], [], [], [], []]
    best_hit_qa = [[], [], [], [], []]
    best_hit_chat = [[], [], [], [], []]
    best_hit_food = [[], [], [], [], []]

    eval_metric = [-1]
    result_path = f"{args.time}_{args.model_name}_result"
    # with open(os.path.join(args.output_dir, result_path), 'a', encoding='utf-8') as result_f:
    #     result_f.write(
    #         '\n=================================================\n')
    #     result_f.write(get_time_kst())
    #     result_f.write('\n')
    #     result_f.write('Argument List:' + str(sys.argv) + '\n')
    #     # for i, v in vars(args).items():
    #     #     result_f.write(f'{i}:{v} || ')
    #     result_f.write('\n')
    #     result_f.write('[EPOCH]\tHit@1\tHit@5\tHit@10\tHit@20\n')

    if args.stage == 'retrieve':
        knowledge_index = knowledge_reindexing(args, knowledge_data, retriever, args.stage)
        knowledge_index = knowledge_index.to(args.device)

    for epoch in range(args.num_epochs):
        if args.update_freq == -1:
            update_freq = len(train_dataloader)
        else:
            update_freq = min(len(train_dataloader), args.update_freq)

        # if epoch < 4:
        #     args.pseudo_pos_num = 1
        #     args.pseudo_pos_rank = 1
        # else:
        #     args.pseudo_pos_num = 2
        #     args.pseudo_pos_rank = 2
        train_epoch_loss = 0
        num_update = 0
        for batch in tqdm(train_dataloader, desc="Knowledge_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            retriever.train()
            dialog_token = batch['input_ids']
            dialog_mask = batch['attention_mask']
            goal_idx = batch['goal_idx']
            # response = batch['response']
            candidate_indice = batch['candidate_indice']
            candidate_knowledge_token = batch['candidate_knowledge_token']  # [B,2,256]
            candidate_knowledge_mask = batch['candidate_knowledge_mask']  # [B,2,256]
            # sampling_results = batch['sampling_results']

            # pseudo_positive_idx = torch.stack([idx[0] for idx in batch['candidate_indice']])
            # pseudo_positive = batch['pseudo_positive']
            # pseudo_negative = batch['pseudo_negative']
            # target_knowledge = candidate_knowledge_token[:, 0, :]

            target_knowledge_idx = batch['target_knowledge']  # [B,5,256]

            # # for every update
            # if args.stage == 'retrieve':
            #     knowledge_index = knowledge_reindexing(args, knowledge_data, retriever, args.stage)
            #     knowledge_index = knowledge_index.to(args.device)

            if args.know_ablation == 'target':
                logit = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, goal_idx)
                loss = torch.mean(criterion(logit, target_knowledge_idx[:, 0]))  # For MLP predict

            if args.know_ablation == 'pseudo':
                # dialog_token = dialog_token.unsqueeze(1).repeat(1, batch['pseudo_target'].size(1), 1).view(-1, dialog_mask.size(1))  # [B, K, L] -> [B * K, L]
                # dialog_mask = dialog_mask.unsqueeze(1).repeat(1, batch['pseudo_target'].size(1), 1).view(-1, dialog_mask.size(1))  # [B, K, L] -> [B * K, L]

                if args.stage == 'retrieve':
                    # logit = retriever.compute_know_score_candidate(dialog_token, dialog_mask, knowledge_index[batch['candidate_indice']])
                    # logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]
                    # cumsum_logit = torch.cumsum(logit, dim=1)  # [B, K]
                    # loss = 0
                    # for idx in range(args.pseudo_pos_rank):
                    #     g_logit = cumsum_logit[:, idx]
                    #
                    #     pseudo_mask = torch.zeros_like(logit)
                    #     pseudo_mask[:, :idx + 1] = -1e10
                    #     pseudo_mask = torch.cat([torch.zeros(pseudo_mask.size(0)).unsqueeze(1).to(args.device), pseudo_mask], dim=1)
                    #
                    #     g_logit = torch.cat([g_logit.unsqueeze(1), logit], dim=1)
                    #     loss += (-torch.log_softmax(g_logit + pseudo_mask, dim=1).select(dim=1, index=0)).mean()

                    # cumsum_logit = torch.cumsum(gathered_logit, dim=1)  # [B, K]
                    # for idx in range(args.pseudo_pos_rank):
                    #     g_logit = cumsum_logit[:, idx]
                    #     pseudo_mask = torch.zeros_like(logit)
                    #     pseudo_mask[:, :idx + 1] = -1e10
                    #     pseudo_mask = torch.cat([torch.zeros(pseudo_mask.size(0)).unsqueeze(1).to(args.device), pseudo_mask], dim=1)
                    #     g_logit = torch.cat([g_logit.unsqueeze(1), logit], dim=1)
                    #     loss += (-torch.log_softmax(g_logit + pseudo_mask, dim=1).select(dim=1, index=0)).mean()

                    ### ListMLE (final)
                    if args.train_ablation == 'R':
                        logit = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, goal_idx)
                        logit_exp = torch.exp(logit - torch.max(logit, dim=1, keepdim=True)[0])  # [B, K]
                        all_sum = torch.sum(logit_exp, dim=1, keepdim=True)  # [B, 1]
                        pseudo_logit = torch.gather(logit_exp, 1, batch['pseudo_targets'])

                        cumsum_logit = torch.cumsum(pseudo_logit, dim=1)  # [B, K]
                        denominator = all_sum - (cumsum_logit - pseudo_logit) + 1e-10
                        loss = torch.mean(torch.sum(-torch.log(pseudo_logit / denominator), dim=1))

                    ### Sampling
                    if args.train_ablation == 'S':
                        logit = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, goal_idx)

                        loss = 0
                        exclude = torch.zeros_like(logit)
                        exclude[:, 0] = -1e10
                        for idx in range(args.pseudo_pos_rank):
                            exclude[torch.arange(logit.size(0)), batch['pseudo_targets'][:, idx]] = -1e10
                        pseudo_mask = torch.cat([torch.zeros(exclude.size(0)).unsqueeze(1).to(args.device), exclude], dim=1)

                        for idx in range(args.pseudo_pos_rank):
                            g_logit = logit[torch.arange(logit.size(0)), batch['pseudo_targets'][:, idx]]
                            g_logit = torch.cat([g_logit.unsqueeze(1), logit], dim=1)
                            loss += (-torch.log_softmax(g_logit + pseudo_mask, dim=1).select(dim=1, index=0)).mean()

                    ### Sampling
                    if args.train_ablation == 'O':
                        logit = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, goal_idx)

                        loss = 0
                        exclude = torch.zeros_like(logit)
                        exclude[:, 0] = -1e10
                        for idx in range(args.pseudo_pos_rank):
                            exclude[torch.arange(logit.size(0)), batch['pseudo_targets'][:, idx]] = -1e10
                        pseudo_mask = torch.cat([torch.zeros(exclude.size(0)).unsqueeze(1).to(args.device), exclude], dim=1)

                        g_logit = logit[torch.arange(logit.size(0)), batch['pseudo_targets'][:, args.pseudo_pos_rank - 1]]
                        g_logit = torch.cat([g_logit.unsqueeze(1), logit], dim=1)
                        loss += (-torch.log_softmax(g_logit + pseudo_mask, dim=1).select(dim=1, index=0)).mean()

                    ### List_Group
                    if args.train_ablation == 'G':
                        logit = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, goal_idx)
                        logit_pseudo = torch.gather(logit, 1, batch['pseudo_targets'])  # [B, K]
                        g_logit = torch.mean(logit_pseudo, dim=1)
                        loss = 0
                        exclude = torch.zeros_like(logit)
                        exclude[:, 0] = -1e10
                        for idx in range(args.pseudo_pos_rank):
                            exclude[torch.arange(logit.size(0)), batch['pseudo_targets'][:, idx]] = -1e10

                        pseudo_mask = torch.cat([torch.zeros(exclude.size(0)).unsqueeze(1).to(args.device), exclude], dim=1)

                        g_logit = torch.cat([g_logit.unsqueeze(1), logit], dim=1)
                        loss += (-torch.log_softmax(g_logit + pseudo_mask, dim=1).select(dim=1, index=0)).mean()

                    ### List_Group
                    if args.train_ablation == 'RG':
                        logit = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, goal_idx)
                        logit_pseudo = torch.gather(logit, 1, batch['pseudo_targets'])  # [B, K]
                        cumsum_logit = torch.cumsum(logit_pseudo, dim=1)
                        loss = 0
                        exclude = torch.zeros_like(logit)
                        exclude[:, 0] = -1e10
                        for idx in range(args.pseudo_pos_rank):
                            g_logit = cumsum_logit[:, idx] / (idx + 1)
                            exclude[torch.arange(logit.size(0)), batch['pseudo_targets'][:, idx]] = -1e10

                            pseudo_mask = torch.cat([torch.zeros(exclude.size(0)).unsqueeze(1).to(args.device), exclude], dim=1)

                            g_logit = torch.cat([g_logit.unsqueeze(1), logit], dim=1)
                            loss += (-torch.log_softmax(g_logit + pseudo_mask, dim=1).select(dim=1, index=0)).mean()

                    if args.train_ablation == 'LG':
                        logit = retriever.compute_know_score(dialog_token, dialog_mask, knowledge_index, goal_idx)
                        logit_pseudo = torch.gather(logit, 1, batch['all_negative'])  # [B, K]
                        cumsum_logit = torch.cumsum(logit_pseudo, dim=1)  # [B, K]
                        loss = 0
                        for i in range(args.pseudo_pos_rank):  #
                            # widx = args.pseudo_pos_rank
                            widx = i + 1
                            a = cumsum_logit[:, widx - 1:]
                            b = torch.cat([torch.zeros(cumsum_logit.size(0)).unsqueeze(1).to(args.device), cumsum_logit[:, :-widx]], dim=1)
                            c = (a - b) / widx  # [B, K-1]
                            c = torch.cat([c[:, :1], c[:, widx:]], dim=1)
                            loss += (-torch.log_softmax(c, dim=1).select(dim=1, index=0)).mean()

                        # loss = torch.mean(criterion(logit, batch['pseudo_targets'][:, 0]))
                        # for idx in range(1, args.pseudo_pos_rank):
                        #     pseudo_targets = batch['pseudo_targets'][:, :idx + 1]
                        #     exclude = batch['pseudo_targets'][:, :idx + 1]
                        #     # pseudo_targets = batch['pseudo_targets'][:, :idx+1]
                        #     # know_mask = (pseudo_targets != 0)
                        #     # num_know = torch.sum(know_mask, dim=1)
                        #     # g_logit = torch.gather(logit, 1, pseudo_targets)
                        #     # g_logit = torch.sum(g_logit, dim=1) / (num_know + 1e-10)
                        #     g_logit = torch.mean(torch.gather(logit, 1, pseudo_targets), dim=1)
                        #     pseudo_mask = torch.zeros_like(logit)
                        #     pseudo_mask[:, 0] = -1e10
                        #     for j in range(exclude.size(1)):
                        #         pseudo_target = exclude[:, j]  # [B]
                        #         pseudo_mask[torch.arange(logit.size(0)), pseudo_target] = -1e10
                        #     pseudo_mask = torch.cat([torch.zeros(pseudo_mask.size(0)).unsqueeze(1).to(args.device), pseudo_mask], dim=1)
                        #     g_logit = torch.cat([g_logit.unsqueeze(1), logit], dim=1)
                        #     # loss += torch.mean(criterion(logit + pseudo_mask, pseudo_target))
                        #     loss += (-torch.log_softmax(g_logit + pseudo_mask, dim=1).select(dim=1, index=0)).mean()

                ## Group-wise + list-wise (sampling)
                # logit = retriever.compute_know_score_candidate(dialog_token, dialog_mask, knowledge_index[batch['candidate_indice']])

                if args.stage == 'rerank':
                    # loss = retriever.dpr_retrieve_train(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)

                    # G-ListMLE
                    # logit_pos, logit_neg = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_indice, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]
                    # logit = torch.cat([logit_pos, logit_neg], dim=-1)
                    # cumsum_logit = torch.cumsum(logit_pos, dim=1)  # [B, K]
                    # loss = 0
                    # for idx in range(args.pseudo_pos_rank):
                    #     g_logit = cumsum_logit[:, idx] / (idx + 1)
                    #     g_logit = torch.cat([g_logit.unsqueeze(1), logit], dim=1)
                    #     pseudo_mask = torch.zeros_like(logit)
                    #     pseudo_mask[:, :idx + 1] = -1e10
                    #     pseudo_mask = torch.cat([torch.zeros(pseudo_mask.size(0)).unsqueeze(1).to(args.device), pseudo_mask], dim=1)
                    #     loss += (-torch.log_softmax(g_logit + pseudo_mask, dim=1).select(dim=1, index=0)).mean()

                    # GCL version (+food의 경우에는 idea X)

                    logit_pos, logit_neg = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_indice, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]
                    cumsum_logit = torch.cumsum(logit_pos, dim=1)  # [B, K]  # Grouping

                    loss = 0
                    pseudo_confidences_mask = batch['pseudo_confidences']  # [B, K]
                    for idx in range(args.pseudo_pos_rank):
                        # confidence = torch.softmax(pseudo_confidences[:, :idx + 1], dim=-1)
                        g_logit = torch.sum(logit_pos[:, :idx + 1] * pseudo_confidences_mask[:, :idx + 1], dim=-1) / (torch.sum(pseudo_confidences_mask[:, :idx + 1], dim=-1) + 1e-20)
                        # g_logit = cumsum_logit[:, idx] / (idx + 1)
                        # g_logit = cumsum_logit[:, idx] / batch_denominator[:, idx]

                        # g_logit = cumsum_logit[:, idx] / num_samples[:, idx]
                        # g_logit = cumsum_logit[:, idx]  # For Sampling
                        g_logit = torch.cat([g_logit.unsqueeze(1), logit_neg], dim=1)
                        loss += (-torch.log_softmax(g_logit, dim=1).select(dim=1, index=0)).mean()

                ### Group-wise + Seq (original)
                # loss = torch.mean(criterion(logit, batch['pseudo_targets'][:, 0]))
                # for idx in range(1, batch['pseudo_targets'].size(1)):
                #     pseudo_targets = batch['pseudo_targets'][:, :idx + 1]
                #     exclude = batch['pseudo_targets'][:, :idx + 1]
                #     # pseudo_targets = batch['pseudo_targets'][:, :idx+1]
                #
                #     # know_mask = (pseudo_targets != 0)
                #     # num_know = torch.sum(know_mask, dim=1)
                #     # g_logit = torch.gather(logit, 1, pseudo_targets)
                #     # g_logit = torch.sum(g_logit, dim=1) / (num_know + 1e-10)
                #     g_logit = torch.mean(torch.gather(logit, 1, pseudo_targets), dim=1)
                #     pseudo_mask = torch.zeros_like(logit)
                #     pseudo_mask[:, 0] = -1e10
                #     for j in range(exclude.size(1)):
                #         pseudo_target = exclude[:, j]  # [B]
                #         pseudo_mask[torch.arange(logit.size(0)), pseudo_target] = -1e10
                #     pseudo_mask = torch.cat([torch.zeros(pseudo_mask.size(0)).unsqueeze(1).to(args.device), pseudo_mask], dim=1)
                #     g_logit = torch.cat([g_logit.unsqueeze(1), logit], dim=1)
                #     # loss += torch.mean(criterion(logit + pseudo_mask, pseudo_target))
                #     loss += (-torch.log_softmax(g_logit + pseudo_mask, dim=1).select(dim=1, index=0)).mean()

                ### ListNet (final)
                # pseudo_mask = torch.zeros_like(logit)
                # pseudo_mask[:, 0] = -1e10
                # Pd = torch.softmax(logit + pseudo_mask, dim=1)
                # pseudo_soft_label = torch.zeros_like(logit) - 1e10
                # for j in range(batch['pseudo_targets'].size(1)):
                #     pseudo_soft_label[torch.arange(logit.size(0)), batch['pseudo_targets'][:, j]] = batch['pseudo_confidences'][:, j]
                #     pseudo_mask[torch.arange(logit.size(0)), batch['pseudo_targets'][:, j]] = 1
                # Qd = torch.softmax(pseudo_soft_label / args.tau, dim=1)
                # loss = torch.mean(-torch.sum(Qd * torch.log(Pd + 1e-10), dim=1))

                ### ListNet2.0
                # if args.stage == 'rerank':
                #     # logit = retriever.compute_know_score_candidate(dialog_token, dialog_mask, knowledge_index[batch['candidate_indice']])
                #     logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]
                #     Pd = torch.softmax(logit, dim=1)
                #     Qd = torch.softmax(batch['pseudo_confidences'] / args.tau, dim=1)
                #     loss = torch.mean(-torch.sum(Qd * torch.log(Pd + 1e-10), dim=1))

                ### ListMLE (prev)
                # pseudo_soft_label = torch.zeros_like(logit)
                # for j in range(batch['pseudo_targets'].size(1)):
                #     pseudo_soft_label[torch.arange(logit.size(0)), batch['pseudo_targets'][:, j]] = batch['pseudo_confidences'][:, j]
                #
                # pseudo_soft_label = pseudo_soft_label
                # v_min, v_max = pseudo_soft_label.min(dim=1).values, pseudo_soft_label.max(dim=1).values
                # pseudo_confidence = (pseudo_soft_label - v_min.unsqueeze(-1)) / (v_max-v_min + 1e-10).unsqueeze(-1)
                # # pseudo_confidence = torch.softmax(pseudo_soft_label / args.tau, dim=1)
                # pseudo_confidence = torch.gather(pseudo_confidence, 1, batch['pseudo_targets'])  # [B, K]
                # pseudo_mask = torch.zeros_like(logit)
                # pseudo_mask[:, 0] = -1e10
                # logit = logit + pseudo_mask

                ### ListMLE (sampling)
                # logit = retriever.compute_know_score_candidate(dialog_token, dialog_mask, knowledge_index[batch['candidate_indice']])
                # logit_exp = torch.exp(logit - torch.max(logit, dim=1, keepdim=True)[0])  # [B, K]
                # all_sum = torch.sum(logit_exp, dim=1, keepdim=True)  # [B, 1]
                # pseudo_logit = logit_exp[:, :args.pseudo_pos_rank]
                # cumsum_logit = torch.cumsum(pseudo_logit, dim=1)  # [B, K]
                # denominator = all_sum - (cumsum_logit - pseudo_logit) + 1e-10
                # loss = torch.mean(torch.sum(-torch.log(pseudo_logit / denominator), dim=1))

                ### ListMLE2
                # loss_list = []
                # pseudo_soft_label = torch.zeros_like(logit) - 1e10
                # for j in range(batch['pseudo_targets'].size(1)):
                #     pseudo_soft_label[torch.arange(logit.size(0)), batch['pseudo_targets'][:, j]] = batch['pseudo_confidences'][:, j]
                # pseudo_confidence = torch.softmax(pseudo_soft_label / args.tau, dim=1)
                # pseudo_confidence = torch.gather(pseudo_confidence, 1, batch['pseudo_targets'])  # [B, K]
                # for i in range(args.pseudo_pos_rank):
                #     pseudo_mask = torch.zeros_like(logit)
                #     pseudo_mask[:, 0] = -1e10
                #     pseudo_target = batch['pseudo_targets'][:, i]  # [B]
                #     # pseudo_confidence = batch['pseudo_confidences'][:, i]
                #     for j in range(batch['pseudo_targets'].size(1)):
                #         if j < i:
                #             exclude = batch['pseudo_targets'][:, j]
                #             pseudo_mask[torch.arange(logit.size(0)), exclude] = -1e10
                #         # if 2 * i >= j > i:
                #         #     exclude = batch['pseudo_targets'][:, j]
                #         #     pseudo_mask[torch.arange(logit.size(0)), exclude] = -1e10
                #         # if j != i:
                #         #     exclude = batch['pseudo_targets'][:, j]
                #         #     pseudo_mask[torch.arange(logit.size(0)), exclude] = -1e10
                #     # loss_list.append(torch.mean(criterion(logit + pseudo_mask, pseudo_target)))
                #     loss += torch.mean(criterion(logit + pseudo_mask, pseudo_target))  # For MLP predict

                # loss = torch.mean(criterion(logit + pseudo_mask, target_knowledge_idx))
                # loss = torch.mean(loss_list)
                # if args.pseudo_confidence:
                #     loss += torch.mean(criterion(logit + pseudo_mask, pseudo_target) * pseudo_confidence)
                # else:
                # loss += torch.mean(criterion(logit + pseudo_mask, pseudo_target))

                # ranking loss with decay (1>2,3,4,5) (2>3,4,5)
                # pseudo_target = batch['pseudo_target'][:, 0]  # [B * K]
                # loss = criterion(logit, pseudo_target)  # For MLP predict
                # select_mask = torch.zeros_like(logit)
                # for i in range(batch['pseudo_target'].size(1)):
                #     exclude = batch['pseudo_target'][:, i]
                #     select_mask[torch.arange(logit.size(0)), exclude] = -1e10
                # lmb = 0.5
                # for i in range(batch['pseudo_target'].size(1) - 1):
                #     pseudo_target = batch['pseudo_target'][:, i + 1]  # [B * K]
                #     exclude = batch['pseudo_target'][:, i]
                #     select_mask[torch.arange(logit.size(0)), exclude] = -1e10
                #     select_logit = logit + select_mask
                #     loss += lmb * criterion(select_logit, pseudo_target)  # For MLP predict
                #     lmb *= lmb

                # Reranking Loss
                # logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]
                # binary_target = torch.zeros_like(logit)
                # binary_target[:, 0] = 1
                # loss_rerank = nn.BCELoss()(torch.sigmoid(logit), binary_target)
                # loss = loss + loss_rerank

                ## Negative sampling
                # logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]
                # loss = (-torch.log_softmax(logit/2, dim=1).select(dim=1, index=0)).mean()

                ## BPR
                # logit = retriever.knowledge_retrieve(dialog_token, dialog_mask, candidate_knowledge_token, candidate_knowledge_mask)  # [B, 2]
                # predicted_positive = logit[:, 0]
                # predicted_negative = logit[:, 1]
                # relative_preference = predicted_positive-predicted_negative
                # loss = -relative_preference.sigmoid().log().mean()

            train_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_update += 1

        scheduler.step()
        # if num_update > update_freq:
        #     update_key_bert(retriever.key_bert, retriever.query_bert)
        #     num_update = 0
        #     knowledge_index = knowledge_reindexing(args, knowledge_data, retriever)
        #     knowledge_index = knowledge_index.to(args.device)

        knowledge_index = knowledge_reindexing(args, knowledge_data, retriever, args.stage)
        knowledge_index = knowledge_index.to(args.device)
        # retriever.init_know_proj(knowledge_index)

        logger.info(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")

        hit1, hit3, hit5, hit10, hit20, hit_movie_result, hit_music_result, hit_qa_result, hit_poi_result, hit_food_result, hit_chat_result, hit1_new, hit3_new, hit5_new, hit10_new, hit20_new = eval_know(args, test_dataloader, retriever, all_knowledge_data, all_knowledgeDB,
                                                                                                                                                                                                            tokenizer)  # HJ: Knowledge text top-k 뽑아서 output만들어 체크하던 코드 분리

        logger.info("Results")
        logger.info("EPOCH:\t%d" % epoch)
        logger.info("Overall\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (hit1, hit3, hit5, hit10, hit20))
        logger.info("Overall new knowledge\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (hit1_new, hit3_new, hit5_new, hit10_new, hit20_new))

        logger.info("Movie recommendation\t" + "\t".join(hit_movie_result))
        logger.info("Music recommendation\t" + "\t".join(hit_music_result))
        logger.info("Q&A\t" + "\t".join(hit_qa_result))
        logger.info("POI recommendation\t" + "\t".join(hit_poi_result))
        logger.info("Food recommendation\t" + "\t".join(hit_food_result))
        logger.info("Chat about stars\t" + "\t".join(hit_chat_result))
        # with open(os.path.join('results', result_path), 'a', encoding='utf-8') as f:
        #     f.write("EPOCH:\t%d\n" % epoch)
        #     f.write("Overall\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (hit1, hit3, hit5, hit10, hit20))
        #     f.write("Overall new knowledge\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (hit1_new, hit3_new, hit5_new, hit10_new, hit20_new))

        #     f.write("Movie recommendation\t" + "\t".join(hit_movie_result) + "\n")
        #     f.write("Music recommendation\t" + "\t".join(hit_music_result) + "\n")
        #     f.write("Q&A\t" + "\t".join(hit_qa_result) + "\n")
        #     f.write("POI recommendation\t" + "\t".join(hit_poi_result) + "\n")
        #     f.write("Food recommendation\t" + "\t".join(hit_food_result) + "\n")
        #     f.write("Chat about stars\t" + "\t".join(hit_chat_result) + "\n\n")

        if hit1 > eval_metric[0]:
            eval_metric[0] = hit1
            best_hit[0] = hit1
            best_hit[1] = hit3
            best_hit[2] = hit5
            best_hit[3] = hit10
            best_hit[4] = hit20
            best_hit_new[0] = hit1_new
            best_hit_new[1] = hit3_new
            best_hit_new[2] = hit5_new
            best_hit_new[3] = hit10_new
            best_hit_new[4] = hit20_new
            best_hit_movie = hit_movie_result
            best_hit_poi = hit_poi_result
            best_hit_music = hit_music_result
            best_hit_qa = hit_qa_result
            best_hit_chat = hit_chat_result
            best_hit_food = hit_food_result

            if args.stage == 'retrieve':
                torch.save(retriever.state_dict(), os.path.join(args.model_dir, f"{args.model_name}_retriever.pt"))  # TIME_MODELNAME 형식

            # best_hit_chat = hit_chat_result

    logger.info(f'BEST RESULT')
    logger.info(f"BEST Test Hit@1: {best_hit[0]}")
    logger.info(f"BEST Test Hit@3: {best_hit[1]}")
    logger.info(f"BEST Test Hit@5: {best_hit[2]}")
    logger.info(f"BEST Test Hit@10: {best_hit[3]}")
    logger.info(f"BEST Test Hit@20: {best_hit[4]}")

    logger.info("[BEST]")
    logger.info("Overall\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (best_hit[0], best_hit[1], best_hit[2], best_hit[3], best_hit[4]))
    logger.info("New\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (best_hit_new[0], best_hit_new[1], best_hit_new[2], best_hit_new[3], best_hit_new[4]))

    logger.info("Movie recommendation\t" + "\t".join(best_hit_movie))
    logger.info("Music recommendation\t" + "\t".join(best_hit_music))
    logger.info("QA\t" + "\t".join(best_hit_qa))
    logger.info("POI recommendation\t" + "\t".join(best_hit_poi))
    logger.info("Food recommendation\t" + "\t".join(best_hit_food))
    logger.info("Chat about stars\t" + "\t".join(best_hit_chat))
    # with open(os.path.join('results', result_path), 'a', encoding='utf-8') as f:
    #     f.write("[BEST]\n")
    #     f.write("Overall\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (best_hit[0], best_hit[1], best_hit[2], best_hit[3], best_hit[4]))
    #     f.write("New\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (best_hit_new[0], best_hit_new[1], best_hit_new[2], best_hit_new[3], best_hit_new[4]))

    #     f.write("Movie recommendation\t" + "\t".join(best_hit_movie) + "\n")
    #     f.write("Music recommendation\t" + "\t".join(best_hit_music) + "\n")
    #     f.write("QA\t" + "\t".join(best_hit_qa) + "\n")
    #     f.write("POI recommendation\t" + "\t".join(best_hit_poi) + "\n")
    #     f.write("Food recommendation\t" + "\t".join(best_hit_food) + "\n")
    #     f.write("Chat about stars\t" + "\t".join(best_hit_chat) + "\n")
