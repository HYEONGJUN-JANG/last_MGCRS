import torch
from torch import nn
from tqdm import tqdm
import os
from loguru import logger
from torch.utils.data import DataLoader
"""
Goal은 pred정보가 없기에 train,test에 label이 모두 golden인 상황
Topic은 Goal에 대한 pred 정보가 있기에, prompt에서 현재 goal로 무엇을 쓸지 결정해줘야하는 상황
Topic-Train시 prompt의 goal은 
"""
def pred_goal_topic_aug(args, retriever, tokenizer, Auged_Dataset, task):
    Auged_Dataset.args.task = task
    optimizer = torch.optim.Adam(retriever.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(Auged_Dataset), eta_min=args.lr * 0.1)
    data_loader = DataLoader(Auged_Dataset, batch_size=args.batch_size*20, shuffle=False)
    with torch.no_grad():
        task_preds, _ = inEpoch_BatchPlay(args, retriever, tokenizer, data_loader, optimizer, scheduler, epoch=0, task=task, mode='test')
    for i, dataset in enumerate(Auged_Dataset.augmented_raw_sample):
        dataset[f"predicted_{task}"] = [args.taskDic[task]['int'][task_preds[i][j]] for j in range(5)]

def train_goal_topic_bert(args, retriever, tokenizer, train_data_loader, test_data_loader, task):
    optimizer = torch.optim.Adam(retriever.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_data_loader), eta_min=args.lr * 0.1)
    epoch_loss, topk, max_hit_train, max_hit_test = 0, 1 if task == 'goal' else 5, 0, 0
    logger.info(f"{task:^6} train start, Train-Test samples: {len(train_data_loader.dataset)}, {len(test_data_loader.dataset)}")
    for epoch in range(args.num_epochs):
        retriever.train()
        train_task_preds, _ = inEpoch_BatchPlay(args, retriever, tokenizer, train_data_loader, optimizer, scheduler, epoch, task=task, mode='train')

        with torch.no_grad():
            test_task_preds, test_hit1 = inEpoch_BatchPlay(args, retriever, tokenizer, test_data_loader, optimizer, scheduler, epoch, task=task, mode='test')
            if test_hit1 > max_hit_test:
                for i, idx in enumerate(train_data_loader.dataset.idxList):
                    train_data_loader.dataset.augmented_raw_sample[idx][f"predicted_{task}"] = [args.taskDic[task]['int'][train_task_preds[i][j]] for j in range(5)]
                for i, idx in enumerate(test_data_loader.dataset.idxList):
                    test_data_loader.dataset.augmented_raw_sample[idx][f"predicted_{task}"] = [args.taskDic[task]['int'][test_task_preds[i][j]] for j in range(5)]  # args.taskDic[task]['int'][test_task_preds[i][0]]
                ## BEST MODEL SAVE
                model_path = os.path.join(args.saved_model_path, f"{task}_best_model_{args.device}_{args.log_name}.pt") # 동시실행 all flow 진행시 겹치지 않도록 --> 가장좋던모델 저장해줄필요
                logger.info(f"{task} Best model saved: {model_path}")
                torch.save(retriever.state_dict(), model_path)


def inEpoch_BatchPlay(args, retriever, tokenizer, data_loader, optimizer, scheduler, epoch, task, mode='train'):
    if task.lower() not in ['goal', 'topic']: raise Exception("Task should be 'goal' or 'topic'")
    criterion = nn.CrossEntropyLoss().to(args.device)
    data_loader.dataset.args.task = task
    data_loader.dataset.subtask = task

    if task=='topic':         # TopicTask_Train_Prompt_usePredGoal TopicTask_Test_Prompt_usePredGoal
        if data_loader.dataset.TopicTask_Train_Prompt_usePredGoal:
            logger.info(f"Topic {mode}_prompt input predicted goal hit@1: {sum([aug['predicted_goal'][0]==aug['goal'] for aug in data_loader.dataset.augmented_raw_sample])/len(data_loader.dataset.augmented_raw_sample):.3f}")
        elif data_loader.dataset.TopicTask_Test_Prompt_usePredGoal:
            logger.info(f"Topic {mode}_prompt input predicted goal hit@1: {sum([aug['predicted_goal'][0]==aug['goal'] for aug in data_loader.dataset.augmented_raw_sample])/len(data_loader.dataset.augmented_raw_sample):.3f}")

    gradient_accumulation_steps = 500
    epoch_loss, steps = 0, 0

    torch.cuda.empty_cache()
    contexts, resps, task_labels, gen_resps, task_preds, gold_goal, gold_topic, types = [], [], [], [], [], [], [], []
    test_hit1, test_hit3, test_hit5 = [], [], []
    predicted_goal_True_cnt = []
    for batch in tqdm(data_loader, desc=f"Epoch_{epoch}_{task:^5}_{mode:^5}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        input_ids, attention_mask, response, goal_idx, topic_idx = [batch[i].to(args.device) for i in ["input_ids", "attention_mask", "response", 'goal_idx', 'topic_idx']]

        target = goal_idx if task == 'goal' else topic_idx

        # Model Forwarding
        dialog_emb = retriever(input_ids=input_ids, attention_mask=attention_mask)  # [B, d]
        if task == 'goal': dialog_emb = retriever.goal_proj(dialog_emb)  # [B, goal_num]
        elif task == 'topic': dialog_emb = retriever.topic_proj(dialog_emb) # [B, topic_num]
        loss = criterion(dialog_emb, target)
        epoch_loss += loss
        if 'train' == mode:
            optimizer.zero_grad()
            loss.backward()
            if (steps + 1) % gradient_accumulation_steps == 0: torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1)
            optimizer.step()
            loss.detach()
            retriever.zero_grad()

        topk_pred = [list(i) for i in torch.topk(dialog_emb, k=5, dim=-1).indices.detach().cpu().numpy()]
        ## For Scoring and Print
        contexts.extend(tokenizer.batch_decode(input_ids))
        task_preds.extend(topk_pred)
        task_labels.extend([int(i) for i in target.detach()])
        gold_goal.extend([int(i) for i in goal_idx])
        gold_topic.extend([int(i) for i in topic_idx])

        # if task=='topic' and mode=='test': predicted_goal_True_cnt.extend([real_goal==pred_goal for real_goal, pred_goal  in zip(goal_idx, batch['predicted_goal_idx'])])

    hit1_ratio = sum([label == preds[0] for preds, label in zip(task_preds, task_labels)]) / len(task_preds)

    Hitdic, Hitdic_ratio, output_str = HitbyType(args, task_preds, task_labels, gold_goal)
    assert Hitdic['Total']['total'] == len(data_loader.dataset)
    if mode == 'test':
        for i in output_str:
            logger.info(f"{mode}_{epoch}_{task} {i}")
    if 'train' == mode: scheduler.step()
    savePrint(args, contexts, task_preds, task_labels, gold_goal, gold_topic, epoch, task, mode)
    torch.cuda.empty_cache()
    return task_preds, hit1_ratio


def savePrint(args, contexts, task_preds, task_labels, gold_goal, gold_topic, epoch, task, mode):
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)
    path = os.path.join(args.output_dir, f"{args.log_name}_{epoch}_{task}_{mode}.txt")
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(contexts)):
            if i > 400: break
            f.write(f"Input: {contexts[i]}\n")
            f.write(f"Pred : {', '.join([args.taskDic[task]['int'][i] for i in task_preds[i]])}\n")
            f.write(f"Label: {args.taskDic[task]['int'][task_labels[i]]}\n")
            f.write(f"Real_Goal : {args.taskDic['goal']['int'][gold_goal[i]]}\n")
            f.write(f"Real_Topic: {args.taskDic['topic']['int'][gold_topic[i]]}\n\n")


def HitbyType(args, task_preds, task_labels, gold_goal, goal_types = ['Q&A', 'Movie recommendation', 'Music recommendation', 'POI recommendation', 'Food recommendation']):
    if len(task_preds[0]) != 2: Exception("Task preds sould be list of tok-k(5)")
    # Hitdit=defaultdict({'hit1':0,'hit3':0,'hit5':0})
    Hitdic = {goal_type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'total': 0} for goal_type in goal_types + ["Others", 'Total']}
    for goal, preds, label in zip(gold_goal, task_preds, task_labels):
        goal_type = args.taskDic['goal']['int'][goal]
        if goal_type in Hitdic: tmp_goal_type = goal_type
        else: tmp_goal_type = 'Others'
        Hitdic[tmp_goal_type]['total'] += 1
        Hitdic['Total']['total'] += 1
        if label in preds:
            Hitdic[tmp_goal_type]['hit5'] += 1
            Hitdic['Total']['hit5'] += 1
            if label in preds[:3]:
                Hitdic[tmp_goal_type]['hit3'] += 1
                Hitdic['Total']['hit3'] += 1
                if label == preds[0]:
                    Hitdic[tmp_goal_type]['hit1'] += 1
                    Hitdic['Total']['hit1'] += 1
    assert Hitdic['Total']['hit1'] == sum([label == preds[0] for preds, label in zip(task_preds, task_labels)]) and Hitdic['Total']['total'] == len(task_preds)
    Hitdic_ratio = {goal_type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'total': 0} for goal_type in goal_types + ["Others", 'Total']}
    output_str = [f"                         hit1,  hit3,  hit5, total_cnt"]
    for k in Hitdic_ratio.keys():
        Hitdic_ratio[k]['total'] = Hitdic[k]['total']
        for hit in ['hit1', 'hit3', 'hit5']:
            if Hitdic[k]['total'] > 0:
                Hitdic_ratio[k][hit] = Hitdic[k][hit] / Hitdic[k]['total']
        output_str.append(f"{k:^22}: {Hitdic_ratio[k]['hit1']:.3f}, {Hitdic_ratio[k]['hit3']:.3f}, {Hitdic_ratio[k]['hit5']:.3f}, {Hitdic_ratio[k]['total']}")
    return Hitdic, Hitdic_ratio, output_str





# {'input':input_text[i], 'pred': pred_topic_text[i], 'target':target_topic_text[i], 'correct':correct[i], 'response': real_resp[i], 'goal_type': goal_type[i]}


if __name__ == "__main__":
    import main

    main.main()