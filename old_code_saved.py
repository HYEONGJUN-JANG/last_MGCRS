
if False:
    ## 이전에 쓰던 logging 세팅 보존하던것
    def initLogging(args):
        filename = f'{args.time}_{"DEBUG" if args.debug else args.log_name}_{args.model_name.replace("/", "_")}_log.txt'
        filename = os.path.join(args.log_dir, filename)
        # global logger
        # if logger == None: logger = logging.getLogger()
        # else:  # wish there was a logger.close()
        #     for handler in logger.handlers[:]:  # make a copy of the list
        #         logger.removeHandler(handler)
        #
        # logger.setLevel(logging.DEBUG)
        # formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%Y/%m/%d_%p_%I:%M:%S ')
        # if args.debug : pass
        # else:
        #     fh = logging.FileHandler(filename)
        #     fh.setFormatter(formatter)
        #     logger.addHandler(fh)
        #
        # sh = logging.StreamHandler(sys.stdout)
        # sh.setFormatter(formatter)
        # logger.addHandler(sh)
        # if not args.debug :
        logger.remove()
        fmt="<green>{time:YYMMDD_HH:mm:ss}</green> | {message}"
        logger.add(filename, encoding='utf-8')
        logger.add(sys.stdout, format=fmt, level="INFO", colorize=True)
        logger.info(f"FILENAME: {filename}")
        logger.info('Commend: {}'.format(', '.join(map(str, sys.argv))))
        return logger


if False:
    ## train_goal_topic.py에서 쓰이던 코드
    class EarlyStopping:
    #주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지

        def __init__(self, args, patience=7, verbose=False, delta=0.004, path='checkpoint.pt'):
            """
            Args:
                patience (int): validation score 가 개선된 후 기다리는 기간 | Default: 7
                verbose (bool): True일 경우 각 Score 의 개선 사항 메세지 출력 | Default: False
                delta (float): score가 delta % 만큼 개선되었을때, 인정 | Default: 0.05
                path (str): checkpoint저장 경로 | Default: 'checkpoint.pt'
            """
            self.args = args
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = Inf
            self.delta = delta
            self.path = path

        def __call__(self, score, model):
            score = score
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(score, model)
            elif score < self.best_score * (1 + self.delta):
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience and self.args.earlystop: self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0


    def train_goal(args, train_dataloader, test_dataloader, retriever, tokenizer):
        assert args.task == 'type'
        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer = optim.Adam(retriever.parameters(), lr=args.lr)
        jsonlineSave = []
        TotalLoss = 0
        checkf1 = 0
        save_output_mode = False  # True일 경우 해당 epoch에서의 batch들 모아서 output으로 save
        # modelpath = os.path.join(args.model_dir, f"{args.task}_best_model.pt")

        modelpath = os.path.join(args.model_dir, f"{args.task}_best_bart_model.pt") if args.usebart else os.path.join(args.model_dir, f"{args.task}_best_model.pt")

        early_stopping = EarlyStopping(args, patience=7, path=modelpath, verbose=True)
        logger.info("Train_Goal")
        gpucheck = True
        cnt = 0
        for epoch in range(args.num_epochs):
            train_epoch_loss = 0
            if args.num_epochs > 1:
                torch.cuda.empty_cache()
                cnt = 0
                if epoch >= args.num_epochs - 1: save_output_mode = True
                # TRAIN
                print("Train")
                #### return {'dialog_token': dialog_token, 'dialog_mask': dialog_mask, 'target_knowledge': target_knowledge, 'goal_type': goal_type, 'response': response, 'topic': topic, 'user_profile':user_profile, 'situation':situation}
                for batch in tqdm(train_dataloader, desc="Type_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
                    retriever.train()
                    cbdicKeys = ['dialog_token', 'dialog_mask', 'response', 'type', 'topic']
                    if args.task == 'know': cbdicKeys += ['candidate_indice']
                    context_batch = batchify(args, batch, tokenizer, task=args.task)
                    dialog_token, dialog_mask, response, type, topic = [context_batch[i] for i in cbdicKeys]
                    targets = type
                    if args.usebart:
                        gen_labels = context_batch['label']
                        gen_loss, dot_score = retriever.goal_selection(dialog_token, dialog_mask, gen_labels)
                        loss = gen_loss
                    else:
                        dot_score = retriever.goal_selection(dialog_token, dialog_mask)
                        loss = criterion(dot_score, targets)
                    # if args.usebart: loss = gen_loss
                    train_epoch_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss.detach()
                    if gpucheck: gpucheck = checkGPU(args, logger)
                    cnt += len(batch['dialog'])

            # TEST
            test_labels = []
            test_preds = []

            # test_inputs = []
            target_goal_texts = []
            test_gens = []

            test_loss = 0
            print("TEST")
            torch.cuda.empty_cache()
            retriever.eval()
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Type_Test", bar_format=' {l_bar} | {bar:23} {r_bar}'):
                    cbdicKeys = ['dialog_token', 'dialog_mask', 'response', 'type', 'topic']
                    if args.task == 'know': cbdicKeys += ['candidate_indice']
                    context_batch = batchify(args, batch, tokenizer, task=args.task)
                    dialog_token, dialog_mask, response, type, topic = [context_batch[i] for i in cbdicKeys]
                    batch_size = dialog_token.size(0)
                    targets = type
                    if args.usebart:
                        gen_labels = context_batch['label']
                        gen_loss, dot_score = retriever.goal_selection(dialog_token, dialog_mask, gen_labels)
                        loss = gen_loss
                    else:
                        dot_score = retriever.goal_selection(dialog_token, dialog_mask)
                        loss = criterion(dot_score, targets)

                    # if args.usebart: loss= gen_loss
                    test_loss += loss
                    test_pred, test_label = [], []
                    if args.usebart:
                        generated_ids = retriever.query_bert.generate(input_ids=dialog_token, attention_mask=dialog_mask, num_beams=1, max_length=32,  # repetition_penalty=2.5,# length_penalty=1.5,
                                                                    early_stopping=True, )
                        # test_inputs.extend([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in dialog_token])
                        test_gen = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                        test_gens.extend(test_gen)
                        pass
                    else:
                        test_pred.extend(list(map(int, dot_score.argmax(1))))

                    test_label.extend(list(map(int, type)))
                    test_labels.extend(test_label)
                    test_preds.extend(test_pred)

                    target_goal_text = [args.goalDic['int'][idx] for idx in test_label]
                    target_goal_texts.extend(target_goal_text)
                    correct = [p == l for p, l in zip(test_pred, test_label)]
                    if save_output_mode:
                        input_text = tokenizer.batch_decode(dialog_token, skip_special_tokens=True)
                        # target_goal_text = [args.goalDic['int'][idx] for idx in test_label]  # target goal
                        pred_goal_text = [args.goalDic['int'][idx] for idx in test_pred]
                        for i in range(batch_size):
                            if args.usebart:
                                jsonlineSave.append({'input': input_text[i], 'pred_goal': pred_goal_text[i], 'gen_goal': test_gen[i], 'target_goal': target_goal_text[i], 'correct': correct[i]})
                            else:
                                jsonlineSave.append({'input': input_text[i], 'pred_goal': pred_goal_text[i], 'target_goal': target_goal_text[i], 'correct': correct[i]})
            p, r, f = round(precision_score(test_labels, test_preds, average='weighted', zero_division=0), 3), round(recall_score(test_labels, test_preds, average='weighted', zero_division=0), 3), round(f1_score(test_labels, test_preds, average='weighted', zero_division=0), 3)

            print(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")
            print(f"Train samples: {cnt}, Test samples: {len(test_labels)}")
            print(f"Test Loss: {test_loss}")
            print(f"P/R/F1: {p} / {r} / {f}")
            # sum([1 if g in k else 0 for g, k in zip(test_gens , target_goal_texts)])
            print(f"Test Hit@1: {sum(correct) / len(correct)}")
            if args.usebart:
                print(f"{args.task} Generation Test: {sum([1 if target in gen else 0 for gen, target in zip(test_gens, target_goal_texts)]) / len(test_gens)}")
            logger.info("{} Epoch: {}, Train Loss: {}, Test Loss: {}, P/R/F: {}/{}/{}".format(args.task, epoch, train_epoch_loss, test_loss, p, r, f))
            logger.info(f"Train samples: {cnt}, Test samples: {len(test_labels)}")
            TotalLoss += train_epoch_loss / len(train_dataloader)
            early_stopping(f, retriever)
            if early_stopping.early_stop and args.earlystop:
                print("Early stopping")
                logger.info("Early Stopping on Epoch {}, Path: {}".format(epoch, modelpath))
                break

        # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
        write_pkl(obj=jsonlineSave, filename=os.path.join(args.data_dir, 'print', 'goal_jsonline_test_output.pkl'))  # 입출력 저장
        save_json_hj(args, f"{args.time}_inout", jsonlineSave, "goal")
        del optimizer
        torch.cuda.empty_cache()
        print('done')


    def train_topic(args, train_dataloader, test_dataloader, retriever, tokenizer):
        assert args.task == 'topic'
        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer = optim.Adam(retriever.parameters(), lr=args.lr)
        jsonlineSave = []
        TotalLoss = 0
        save_output_mode = False  # True일 경우 해당 epoch에서의 batch들 모아서 output으로 save
        # modelpath = os.path.join(args.model_dir, f"{args.task}_best_model.pt")
        modelpath = os.path.join(args.model_dir, f"{args.task}_best_bart_model.pt") if args.usebart else os.path.join(args.model_dir, f"{args.task}_best_model.pt")
        early_stopping = EarlyStopping(args, patience=7, path=modelpath, verbose=True)
        gpucheck = True
        cnt = 0
        for epoch in range(args.num_epochs):
            logger.info("train epoch: {}".format(epoch))
            torch.cuda.empty_cache()
            cnt = 0
            train_epoch_loss = 0
            checkf1 = 0
            if epoch >= args.num_epochs - 1: save_output_mode = True
            # TRAIN
            print("Train")
            #### return {'dialog_token': dialog_token, 'dialog_mask': dialog_mask, 'target_knowledge': target_knowledge, 'goal_type': goal_type, 'response': response, 'topic': topic, 'user_profile':user_profile, 'situation':situation}
            if args.num_epochs > 1:
                retriever.train()
                for batch in tqdm(train_dataloader, desc="Topic_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
                    cbdicKeys = ['dialog_token', 'dialog_mask', 'response', 'type', 'topic']
                    if args.task == 'know': cbdicKeys += ['candidate_indice']
                    context_batch = batchify(args, batch, tokenizer, task=args.task)
                    dialog_token, dialog_mask, response, type, topic = [context_batch[i] for i in cbdicKeys]
                    batch_size = dialog_token.size(0)
                    targets = topic

                    # dot_score = retriever.topic_selection(dialog_token, dialog_mask)
                    if args.usebart:
                        gen_labels = context_batch['label']
                        gen_loss, dot_score = retriever.topic_selection(dialog_token, dialog_mask, gen_labels)
                        # retriever.query_bert(input_ids=dialog_mask, attention_mask=dialog_mask, labels=gen_labels)
                        loss = gen_loss
                    else:
                        dot_score = retriever.topic_selection(dialog_token, dialog_mask)
                        loss = criterion(dot_score, targets)

                    # if args.usebart: loss = gen_loss
                    train_epoch_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss.detach()
                    if gpucheck: gpucheck = checkGPU(args, logger)
                    cnt += len(batch['dialog'])

            # TEST
            test_labels = []
            test_preds = []
            test_pred_at5s = []
            test_pred_at1_tfs, test_pred_at5_tfs = [], []
            target_topic_texts = []
            test_loss = 0
            test_gens = []
            print("TEST")
            # torch.cuda.empty_cache()
            retriever.eval()
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Topic_Test", bar_format=' {l_bar} | {bar:23} {r_bar}'):
                    cbdicKeys = ['dialog_token', 'dialog_mask', 'response', 'type', 'topic']
                    context_batch = batchify(args, batch, tokenizer, task=args.task)
                    if args.task == 'know':
                        cbdicKeys += ['candidate_indice']
                        dialog_token, dialog_mask, response, type, topic, candidate_indice = [context_batch[i] for i in cbdicKeys]
                    else:
                        dialog_token, dialog_mask, response, type, topic = [context_batch[i] for i in cbdicKeys]
                    batch_size = dialog_token.size(0)
                    goal_type = [args.goalDic['int'][int(i)] for i in type]
                    targets = topic

                    test_label = list(map(int, targets))
                    test_labels.extend(test_label)
                    # user_profile = batch['user_profile']
                    if args.usebart:
                        gen_labels = context_batch['label']
                        gen_loss, dot_score = retriever.topic_selection(dialog_token, dialog_mask, gen_labels)
                    else:
                        dot_score = retriever.topic_selection(dialog_token, dialog_mask)

                    # dot_score = retriever.topic_selection(dialog_token, dialog_mask)
                    loss = criterion(dot_score, targets)
                    test_loss += loss
                    # test_preds.extend(list(map(int, dot_score.argmax(1))))
                    test_pred = [int(i) for i in torch.topk(dot_score, k=1, dim=1).indices]
                    test_pred_at5 = [list(map(int, i)) for i in torch.topk(dot_score, k=5, dim=1).indices]
                    test_preds.extend(test_pred)
                    test_pred_at5s.extend(test_pred_at5)
                    correct = [p == l for p, l in zip(test_pred, test_label)]
                    correct_at5 = [l in p for p, l in zip(test_pred_at5, test_label)]
                    test_pred_at1_tfs.extend(correct)
                    test_pred_at5_tfs.extend(correct_at5)

                    if args.usebart:
                        generated_ids = retriever.query_bert.generate(
                            input_ids=dialog_token,
                            attention_mask=dialog_mask,
                            num_beams=1,
                            max_length=32,
                            # repetition_penalty=2.5,
                            # length_penalty=1.5,
                            early_stopping=True,
                        )
                        # test_inputs.extend([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in dialog_token])
                        test_gen = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                        test_gens.extend(test_gen)
                    target_topic_text = [args.topicDic['int'][idx] for idx in test_labels]  # target goal
                    target_topic_texts.extend(target_topic_text)
                    if save_output_mode:
                        input_text = tokenizer.batch_decode(dialog_token)
                        pred_topic_text = [args.topicDic['int'][idx] for idx in test_preds]
                        pred_top5_texts = [[args.topicDic['int'][idx] for idx in top5_idxs] for top5_idxs in test_pred_at5]
                        real_resp = tokenizer.batch_decode(response, skip_special_tokens=True)
                        for i in range(batch_size):
                            if args.usebart:
                                jsonlineSave.append({'input': input_text[i], 'pred': pred_topic_text[i], 'gen_pred': test_gen, 'pred5': pred_top5_texts[i], 'target': target_topic_text[i], 'correct': correct[i], 'response': real_resp[i], 'goal_type': goal_type[i]})
                            else:
                                jsonlineSave.append({'input': input_text[i], 'pred': pred_topic_text[i], 'pred5': pred_top5_texts[i], 'target': target_topic_text[i], 'correct': correct[i], 'response': real_resp[i], 'goal_type': goal_type[i]})
            p, r, f = round(precision_score(test_labels, test_preds, average='weighted', zero_division=0), 3), round(recall_score(test_labels, test_preds, average='weighted', zero_division=0), 3), round(f1_score(test_labels, test_preds, average='weighted', zero_division=0), 3)
            test_hit1 = round(test_pred_at1_tfs.count(True) / len(test_pred_at1_tfs), 3)
            test_hit5 = round(test_pred_at5_tfs.count(True) / len(test_pred_at5_tfs), 3)
            print(f"Epoch: {epoch}\nTrain Loss: {train_epoch_loss}")
            print(f"Train sampels: {cnt} , Test samples: {len(test_labels)}")
            print(f"Test Loss: {test_loss}")
            print(f"Test P/R/F1: {p} / {r} / {f}")

            print(f"Test Hit@1 / Hit@5: {test_hit1} / {test_hit5}")
            if args.usebart:
                print(f"{args.task} Generation Test: {sum([1 if target in gen else 0 for gen, target in zip(test_gens, target_topic_texts)]) / len(test_gens)}")
            logger.info("{} Epoch: {}, Training Loss: {}, Test Loss: {}".format(args.task, epoch, train_epoch_loss, test_loss))
            logger.info("Test P/R/F1:\t {} / {} / {}".format(p, r, f))
            logger.info("Test Hit@5: {}".format(test_hit5))
            logger.info(f"Train sampels: {cnt} , Test samples: {len(test_labels)}")
            TotalLoss += train_epoch_loss / len(train_dataloader)
            early_stopping(test_hit5, retriever)
            if early_stopping.early_stop and args.earlystop:
                print("Early stopping")
                logger.info("Early Stopping on Epoch {}, Path: {}".format(epoch, modelpath))
                break

        # TODO HJ: 입출력 저장 args처리 필요시 args.save_know_output 에 store_true 옵션으로 만들 필요
        write_pkl(obj=jsonlineSave, filename=os.path.join(args.data_dir, 'print', 'topic_jsonline_test_output.pkl'))  # 입출력 저장
        save_json_hj(args, f"{args.time}_inout", jsonlineSave, 'topic')
        del optimizer
        torch.cuda.empty_cache()
        print('done')


    def json2txt_goal(saved_jsonlines: list, usebart) -> list:
        txtlines = []
        for js in saved_jsonlines:  # TODO: Movie recommendation, Food recommendation, POI recommendation, Music recommendation, Q&A, Chat about stars
            # {'input':input_text[i], 'pred_goal': pred_goal_text[i], 'target_goal':target_goal_text[i], 'correct':correct[i]}
            dialog, pred_goal, target_goal, tf = js['input'], js['pred_goal'], js['target_goal'], js['correct']
            if usebart:
                txt = f"\n---------------------------\n[Target Goal]: {target_goal}\t[Pred Goal]: {pred_goal}\t[Generated Goal]: {js['gen_goal']}\t[TF]: {tf}\n[Dialog]"
            else:
                txt = f"\n---------------------------\n[Target Goal]: {target_goal}\t[Pred Goal]: {pred_goal}\t[TF]: {tf}\n[Dialog]"
            for i in dialog.replace("user :", '|user :').replace("system :", "|system : ").split('|'):
                txt += f"{i}\n"
            txtlines.append(txt)
        return txtlines


    def json2txt_topic(saved_jsonlines: list, usebart) -> list:
        txtlines = []
        for js in saved_jsonlines:  # TODO: Movie recommendation, Food recommendation, POI recommendation, Music recommendation, Q&A, Chat about stars
            # {'input':input_text[i], 'pred': pred_topic_text[i], 'target':target_topic_text[i], 'correct':correct[i], 'response': real_resp[i], 'goal_type': goal_type[i]}
            dialog, pred, pred5, target, tf, response, goal_type = js['input'], js['pred'], js['pred5'], js['target'], js['correct'], js['response'], js['goal_type']
            if usebart:
                txt = f"\n---------------------------\n[Goal]: {goal_type}\t[Target Topic]: {target}\t[Pred Topic]: {pred}\t[Generated Topic]: {js['gen_pred']}\t [TF]: {tf}\n[pred_top5]\n"
            else:
                txt = f"\n---------------------------\n[Goal]: {goal_type}\t[Target Topic]: {target}\t[Pred Topic]: {pred}\t[TF]: {tf}\n[pred_top5]\n"
            for i in pred5:
                txt += f"{i}\n"
            txt += '[Dialog]\n'
            for i in dialog.replace("user :", '||user :').replace("system :", "||system : ").split('||'):
                txt += f"{i}\n"
            txtlines.append(txt)
        return txtlines


    def save_json_hj(args, filename, saved_jsonlines, task):
        '''
        Args:
            args: args
            filename: file name (path포함)
            saved_jsonlines: Key-value dictionary ( goal_type(str), topic(str), tf(str), dialog(str), target(str), response(str) predict5(list)
        Returns: None
        '''
        if task == 'goal':
            txts = json2txt_goal(saved_jsonlines, args.usebart)
        elif task == 'topic':
            txts = json2txt_topic(saved_jsonlines, args.usebart)
        else:
            return
        path = os.path.join(args.data_dir, 'print')
        if not os.path.exists(path): os.makedirs(path)
        file = f'{path}/{args.log_name}_{task}_{filename}.txt'
        with open(file, 'w', encoding='utf-8') as f:
            for i in range(len(txts)):
                f.write(txts[i])

if False: ##-------------------------------------------------------------------------------## 아래 코드 안씀
    """
    data_model.py 에 있던 코드
    """
    class KnowledgeDataset(Dataset):
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


    class KnowledgeTopicDataset(Dataset):
        def __init__(self, args, knowledgeTopicDB, tokenizer):
            super(Dataset, self).__init__()
            self.args = args
            self.tokenizer = tokenizer
            self.know_max_length = args.know_max_length
            self.knowledgeTopicDB = knowledgeTopicDB
            self.data_samples = []

        def __getitem__(self, item):
            topic, data = self.knowledgeTopicDB[item]
            topic_idx = self.args.topicDic['str'][topic]
            tokenized_data = self.tokenizer(data,
                                            max_length=self.know_max_length,
                                            padding='max_length',
                                            truncation=True,
                                            add_special_tokens=True)
            tokens = torch.LongTensor(tokenized_data.input_ids)
            mask = torch.LongTensor(tokenized_data.attention_mask)
            # docid = self.tokenizer.encode(convert_idx_to_docid(item), truncation=True, padding='max_length', max_length=10)[1:-1]  # 이미 Tensor로 받음
            # docid = torch.LongTensor(docid)
            return tokens, mask, topic_idx

        def __len__(self):
            return len(self.knowledgeTopicDB)


    class TopicDataset(Dataset):  # knowledge용 데이터셋
        def __init__(self, args, data_sample, knowledgeDB, train_knowledgeDB, tokenizer, task, mode='train'):
            super(Dataset, self).__init__()
            self.args = args
            self.task = task
            self.tokenizer = tokenizer
            self.knowledgeDB = knowledgeDB
            self.train_knowledgeDB = train_knowledgeDB
            self.augmented_raw_sample = data_sample
            self.mode = mode
            self.idxList = deque(maxlen=len(self.augmented_raw_sample))

        def __getitem__(self, idx):  # TODO 구현 전
            data = self.augmented_raw_sample[idx]
            self.idxList.append(idx)
            cbdicKeys = ['dialog', 'user_profile', 'response', 'goal', 'topic', 'situation', 'target_knowledge', 'candidate_knowledges', 'candidate_confidences']
            dialog, user_profile, response, goal, topic, situation, target_knowledge_idx, candidate_knowledges, candidate_confidences = [data[i] for i in cbdicKeys]
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

            context_batch = defaultdict()
            prefix = '<profile>' + user_profile  # + '<goal>' + goal + self.tokenizer.sep_token

            prefix_encoding = self.tokenizer.encode(prefix)[1:-1][:self.args.max_prefix_length]
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

            if "predicted_goal" in data:
                context_batch['predicted_goal']


            for k, v in context_batch.items():
                if not isinstance(v, torch.Tensor):
                    context_batch[k] = torch.as_tensor(v, device=self.args.device)
                    # context_batch[k] = torch.as_tensor(v)
            return context_batch

        def __len__(self): return len(self.augmented_raw_sample)
