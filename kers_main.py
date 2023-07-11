import argparse
import utils
from main import initLogging, log_args
from loguru import logger
from copy import deepcopy
import os
import data_utils
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import data_model as data_model
from models.ours.retriever import Retriever
from model_play.ours import train_bert_goal_topic
from model_play.kers import kers_knowledge_retrieve
from random import shuffle

"""
1. ours의 goal, topic model을 통해 data에 predicted goal, topic 레이블 붙이기
2. knowledge retrieve task 수행 및 평가, output저장
3. KERS의 decoder 수행
"""


def add_kers_specific_args(parser):
    parser.add_argument("--method", type=str, default="kers", help=" Method ")
    parser.add_argument("--kers_retrieve_input_length", type=int, default=768, help=" Method ")
    parser.add_argument("--TopicTask_Train_Prompt_usePredGoal", action='store_true', help="Topic prediction시 Predicted goal 사용여부 (Train)")
    parser.add_argument("--TopicTask_Test_Prompt_usePredGoal", action='store_true', help="Topic prediction시 Predicted goal 사용여부 (Test)")
    parser.add_argument("--gtpred", action='store_true', help="Goal-Topic prediction 해서 label로 추가 할지 여부")
    parser.add_argument("--usePseudoTrain", action='store_true', help="Pseudo label을 label로 사용할지 여부 (Train)")
    parser.add_argument("--usePseudoTest", action='store_true', help="Pseudo label을 label로 사용할지 여부 (Test)")

    parser.add_argument("--inputWithKnowledge", action='store_true', help="Input으로 Dialog 외의 정보들도 줄지 여부")
    parser.add_argument("--inputWithTopic", action='store_true', help="Input에 Topic도 넣어줄지 여부")
    parser.add_argument("--shuffle_pseudo", action='store_true', help="Pseudo knowledge에 대해 shuffle할지 여부")

    return parser


def main():
    parser = argparse.ArgumentParser(description="kers_main.py")
    parser = utils.default_parser(parser)
    parser = add_kers_specific_args(parser)
    # default_args.debug=True

    args = parser.parse_args()
    args.method = 'kers'
    args.model_name = 'kers'
    args.gtpred = True
    args.max_length = 256  # BERT
    args.max_gen_length = 256  # knowledge comment들어간경우 무진장 긺
    args.shuffle_pseudo = True
    # args.usePseudoTrain, args.usePseudoTest = True, False # 230711 TH: Train은 Pseudo_label, Test는 Gold_label이 우리 상황과 맞는것

    args = utils.dir_init(args)
    initLogging(args)
    log_args(args)

    # args.batch_size = 512
    topicDic = data_utils.readDic(os.path.join(args.data_dir, "topic2id.txt"))
    goalDic = data_utils.readDic(os.path.join(args.data_dir, "goal2id.txt"))
    args.topicDic = topicDic
    args.goalDic = goalDic
    args.topic_num = len(topicDic['int'])
    args.goal_num = len(goalDic['int'])
    args.taskDic = {'goal': goalDic, 'topic': topicDic}
    train_dataset_raw, train_knowledge_base, _ = data_utils.dataset_reader(args, 'train')
    test_dataset_raw, test_knowledge_base, _ = data_utils.dataset_reader(args, 'test')
    dev_dataset_raw, valid_knowledge_base, _ = data_utils.dataset_reader(args, 'dev')  # TH: 이거 dev_dataset_raw 가 아니라 train_dataset_raw 로 되어 있던데?? 230601

    # train_dataset_resp = data_utils.process_augment_sample(train_dataset_raw)
    # test_dataset_resp = data_utils.process_augment_sample(test_dataset_raw)

    logger.info("Knowledge DB 구축")
    train_knowledgeDB, all_knowledgeDB = set(), set()
    train_knowledgeDB.update(train_knowledge_base)

    all_knowledgeDB.update(train_knowledge_base)
    all_knowledgeDB.update(valid_knowledge_base)
    all_knowledgeDB.update(test_knowledge_base)

    train_knowledgeDB = list(train_knowledgeDB)
    all_knowledgeDB = list(all_knowledgeDB)

    train_dataset_aug, test_dataset_aug = None, None
    # -- For Knowledge Retrieve Task --#
    if args.gtpred or not os.path.exists(os.path.join(args.data_dir, 'pred_aug', 'kers_train_gt_pred_auged_dataset.pkl')):
        logger.info(f"Create Predicted augmented dataset in {os.path.join(args.data_dir, 'pred_aug', 'kers_train_gt_pred_auged_dataset.pkl')}")
        # train_dataset_resp = data_utils.process_augment_all_sample(train_dataset_raw)
        # test_dataset_resp = data_utils.process_augment_all_sample(test_dataset_raw)
        train_dataset_resp = data_utils.process_augment_sample(train_dataset_raw, goal_list=['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'POI recommendation', 'Food recommendation'])
        test_dataset_resp = data_utils.process_augment_sample(test_dataset_raw, goal_list=['Movie recommendation', 'POI recommendation', 'Music recommendation', 'Q&A', 'POI recommendation', 'Food recommendation'])
        train_dataset_aug, test_dataset_aug = mk_goal_topic_pred(args=args, aug_train_dataset=train_dataset_resp, aug_test_dataset=test_dataset_resp)

    kers_knowledge_retrieve_task = True
    if kers_knowledge_retrieve_task:
        from transformers import BertTokenizer, BartForConditionalGeneration, BartTokenizer
        model = 'facebook/bart-base' if args.version == '2' else 'fnlp/bart-base-chinese'
        assert args.version == '2'
        args.num_beams = 5
        args.inputWithKnowledge = True
        # args.inputWithTopic=False

        model_cache_dir = os.path.join(args.home, 'model_cache', model)
        if args.version == 1:
            tokenizer = BertTokenizer.from_pretrained(model, cache_dir=model_cache_dir)
        else:
            tokenizer = BartTokenizer.from_pretrained(model, cache_dir=model_cache_dir)
        print("Use Pretrained Model")
        bert_special_tokens_dict = {'additional_special_tokens': ['<dialog>', '<topic>', '<type>', '<user_profile>', '<situation>', '<last_type>', '<knowledge>']}
        tokenizer.add_special_tokens(bert_special_tokens_dict)
        model = BartForConditionalGeneration.from_pretrained(model, cache_dir=model_cache_dir)
        model.resize_token_embeddings(len(tokenizer))
        model.to(args.device)
        logger.info(model.config)

        if train_dataset_aug or test_dataset_aug:
            pass
        else:
            logger.info(f"Read Dataset pkl : {os.path.join(args.data_dir, 'pred_aug', 'kers_train_gt_pred_auged_dataset.pkl')}")
            # train_dataset_aug = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', 'gt_train_pred_aug_dataset.pkl'))
            # test_dataset_aug  = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', 'gt_test_pred_aug_dataset.pkl'))
            train_dataset_aug = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', 'kers_train_gt_pred_auged_dataset.pkl'))
            test_dataset_aug = utils.read_pkl(os.path.join(args.data_dir, 'pred_aug', 'kers_test_gt_pred_auged_dataset.pkl'))
        args.task = 'knowledge'

        log_args(args)
        if args.shuffle_pseudo:
            logger.info("Shuffle Pseudo knowledge order")
            train_dataset_aug = pseudo_knowledge_shuffle(train_dataset_aug)
            test_dataset_aug = pseudo_knowledge_shuffle(test_dataset_aug)
        logger.info(f'Input with knowledges: {args.inputWithKnowledge}, Input with topic: {args.inputWithTopic}')
        kers_knowledge_retrieve.train_test_pseudo_knowledge_bart(args, model, tokenizer, train_dataset_aug, test_dataset_aug, train_knowledgeDB, all_knowledgeDB)

        # train_datamodel_resp = data_pseudo.GenerationDataset(args, train_dataset_resp, train_knowledge_seq_set, tokenizer, mode='train')
        # test_datamodel_resp = data_pseudo.GenerationDataset(args, test_dataset_resp, train_knowledge_seq_set, tokenizer, mode='test')
        #
        # train_data_loader = DataLoader(train_datamodel_resp, batch_size=args.batch_size, shuffle=True)
        # test_data_loader = DataLoader(test_datamodel_resp, batch_size=args.batch_size, shuffle=False)
        #
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4, eps=5e-9)
        # beam_temp = args.num_beams
        # args.num_beams = 1


def pseudo_knowledge_shuffle(dataset_aug):
    shuffled_dataset = deepcopy(dataset_aug)
    for data in shuffled_dataset:
        tmp = [[k, c] for k, c in zip(data['candidate_knowledges'], data['candidate_confidences'])]
        shuffle(tmp)
        data['candidate_knowledges'] = [i[0] for i in tmp]
        data['candidate_confidences'] = [i[1] for i in tmp]
    return shuffled_dataset


def mk_goal_topic_pred(args, aug_train_dataset, aug_test_dataset):
    args = deepcopy(args)
    args.bert_name = 'bert-base-uncased'
    bert_special_tokens_dict = {'additional_special_tokens': ['<dialog>', '<topic>', '<type>', '<user_profile>', '<situation>'], }
    logger.info("Load Model")
    bert_model = AutoModel.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name, cache_dir=os.path.join(args.home, "model_cache", args.bert_name))
    tokenizer.add_special_tokens(bert_special_tokens_dict)  # [TH] add bert special token (<dialog>, <topic> , <type>)
    bert_model.resize_token_embeddings(len(tokenizer))
    args.hidden_size = bert_model.config.hidden_size  # BERT large 쓸 때 대비

    logger.info(f"Dataset Length: {len(aug_train_dataset)}, {len(aug_test_dataset)}")
    retriever = Retriever(args, bert_model)
    model_path = os.path.join(args.saved_model_path, f"goal_best_model.pt")
    logger.info(f"Load Goal Model: {model_path}")
    retriever.load_state_dict(torch.load(model_path, map_location=args.device))
    retriever.to(args.device)

    train_datamodel_topic = data_model.GenerationDataset(args, data_sample=aug_train_dataset, knowledgeDB=None, tokenizer=tokenizer, mode='train', subtask='goal')
    test_datamodel_topic = data_model.GenerationDataset(args, data_sample=aug_test_dataset, knowledgeDB=None, tokenizer=tokenizer, mode='test', subtask='goal')
    pred_goal_topic_aug(args, retriever, tokenizer, train_datamodel_topic, task='goal')
    pred_goal_topic_aug(args, retriever, tokenizer, test_datamodel_topic, task='goal')

    model_path = os.path.join(args.saved_model_path, f"topic_best_model_GP.pt")
    logger.info(f"Load Topic Model: {model_path}")
    retriever.load_state_dict(torch.load(model_path, map_location=args.device))
    retriever.to(args.device)
    # pred_goal_topic_aug(args, retriever, tokenizer, train_datamodel_topic, task='topic')
    pred_auged_train_dataset = pred_goal_topic_aug(args, retriever, tokenizer, train_datamodel_topic, task='topic')
    pred_auged_test_dataset = pred_goal_topic_aug(args, retriever, tokenizer, test_datamodel_topic, task='topic')
    utils.write_pkl(pred_auged_train_dataset, os.path.join(args.data_dir, 'pred_aug', 'kers_train_gt_pred_auged_dataset.pkl'))
    utils.write_pkl(pred_auged_test_dataset, os.path.join(args.data_dir, 'pred_aug', 'kers_test_gt_pred_auged_dataset.pkl'))
    return aug_train_dataset, aug_test_dataset


def pred_goal_topic_aug(args, retriever, tokenizer, Auged_Dataset, task):
    Auged_Dataset.args.task = task
    optimizer = None  # torch.optim.Adam(retriever.parameters(), lr=args.lr)
    scheduler = None  # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(Auged_Dataset), eta_min=args.lr * 0.1)
    data_loader = DataLoader(Auged_Dataset, batch_size=args.batch_size * 20, shuffle=False)
    with torch.no_grad():
        task_preds, _ = inEpoch_BatchPlay(args, retriever, tokenizer, data_loader, optimizer, scheduler, epoch=0, task=task, mode='test')
    for i, dataset in enumerate(Auged_Dataset.augmented_raw_sample):
        dataset[f"predicted_{task}"] = [args.taskDic[task]['int'][task_preds[i][j]] for j in range(5)]
    return Auged_Dataset.augmented_raw_sample


def inEpoch_BatchPlay(args, retriever, tokenizer, data_loader, optimizer, scheduler, epoch, task, mode='train'):
    if task.lower() not in ['goal', 'topic']: raise Exception("Task should be 'goal' or 'topic'")
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    data_loader.dataset.args.task = task
    data_loader.dataset.subtask = task

    if task == 'topic':  # TopicTask_Train_Prompt_usePredGoal TopicTask_Test_Prompt_usePredGoal
        if data_loader.dataset.TopicTask_Train_Prompt_usePredGoal:
            logger.info(f"Topic {mode}에 사용된_prompt input predicted goal hit@1: {sum([aug['predicted_goal'][0] == aug['goal'] for aug in data_loader.dataset.augmented_raw_sample]) / len(data_loader.dataset.augmented_raw_sample):.3f}")
        elif data_loader.dataset.TopicTask_Test_Prompt_usePredGoal:
            logger.info(f"Topic {mode}에 사용된_prompt input predicted goal hit@1: {sum([aug['predicted_goal'][0] == aug['goal'] for aug in data_loader.dataset.augmented_raw_sample]) / len(data_loader.dataset.augmented_raw_sample):.3f}")

    gradient_accumulation_steps = 500
    epoch_loss, steps = 0, 0

    torch.cuda.empty_cache()
    contexts, resps, task_labels, gen_resps, task_preds, gold_goal, gold_topic, types = [], [], [], [], [], [], [], []
    test_hit1, test_hit3, test_hit5 = [], [], []
    predicted_goal_True_cnt = []
    for batch in tqdm(data_loader, desc=f"Epoch_{epoch}_{task:^5}_{mode:^5}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        # "predicted_goal_idx", "predicted_topic_idx"
        input_ids, attention_mask, response, goal_idx, topic_idx = [batch[i].to(args.device) for i in ["input_ids", "attention_mask", "response", 'goal_idx', 'topic_idx']]

        target = goal_idx if task == 'goal' else topic_idx
        # Model Forwarding
        dialog_emb = retriever(input_ids=input_ids, attention_mask=attention_mask)  # [B, d]
        if task == 'goal':
            dialog_emb = retriever.goal_proj(dialog_emb)
        elif task == 'topic':
            dialog_emb = retriever.topic_proj(dialog_emb)
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


def HitbyType(args, task_preds, task_labels, gold_goal):
    if len(task_preds[0]) != 2: Exception("Task preds sould be list of tok-k(5)")
    goal_types = ['Q&A', 'Movie recommendation', 'Music recommendation', 'POI recommendation', 'Food recommendation']
    # Hitdit=defaultdict({'hit1':0,'hit3':0,'hit5':0})
    Hitdic = {goal_type: {'hit1': 0, 'hit3': 0, 'hit5': 0, 'total': 0} for goal_type in goal_types + ["Others", 'Total']}
    for goal, preds, label in zip(gold_goal, task_preds, task_labels):
        goal_type = args.taskDic['goal']['int'][goal]
        if goal_type in Hitdic:
            tmp_goal_type = goal_type
        else:
            tmp_goal_type = 'Others'
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


if __name__ == '__main__':
    main()