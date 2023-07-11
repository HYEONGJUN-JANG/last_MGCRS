import os
from copy import deepcopy
from torch import optim
from tqdm import tqdm
import torch
import numpy as np
from loguru import logger

## OLD FILE
def train_goal_topic(args, generator, tokenizer, train_dataloader, test_dataloader, subtask):
    logger.info(f"BART Generation Train goal-topic")
    train_dataloader.dataset.subtask = subtask
    test_dataloader.dataset.subtask = subtask
    optimizer = optim.AdamW(generator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_dataloader), eta_min=args.lr * 0.1)
    best_hit = 0
    for epoch in range(args.num_epochs):
        num_beams = 5 if epoch==args.num_epochs-1 else 1
        train_epoch_loss = 0
        for batch in tqdm(train_dataloader, desc="Generate_Train", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            generator.train()
            dialog_token = batch['input_ids'].to(args.device)
            dialog_mask = batch['attention_mask'].to(args.device)
            response = batch['response'].to(args.device)

            loss = generator.generation(dialog_token, dialog_mask, response)
            # loss = criterion(dot_score, targets)
            train_epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        logger.info(f"Epoch: {epoch}_Train Loss: {train_epoch_loss}")

        # test generation task
        all_dialog = []
        all_response = []
        all_generated = []
        goal_types = []
        for batch in tqdm(test_dataloader, desc="Generate Test", bar_format=' {l_bar} | {bar:23} {r_bar}'):
            generator.eval()
            dialog_token = batch['input_ids'].to(args.device)
            dialog_mask = batch['attention_mask'].to(args.device)
            response = batch['response']

            batch_size = dialog_token.shape[0]
            generated = generator.gpt_model.generate(input_ids=dialog_token,
                                                     attention_mask=dialog_mask,
                                                     pad_token_id=tokenizer.pad_token_id,
                                                     max_length=args.max_gen_length,
                                                     early_stopping=True,
                                                     num_return_sequences=num_beams, num_beams=num_beams,
                                                     )
            # decoded_generated = tokenizer.batch_decode(generated)

            gen_resp_ids = []
            for gen_seq, length in zip(generated, batch['context_len']):
                gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                # gen_resp_ids.append(gen_seq[length:]) # for GPT
                gen_resp_ids.append(gen_seq)

            all_generated.extend(tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=True))
            all_response.extend(tokenizer.batch_decode(response, skip_special_tokens=True))
            all_dialog.extend(tokenizer.batch_decode(dialog_token, skip_special_tokens=True))
            goal_types.extend(tokenizer.batch_decode(batch['goal_type'], skip_special_tokens=True))

        result_save_path = os.path.join(args.home, 'output', f"response_write_{args.time}_{args.model_name}_{args.gpt_name}_{args.lr}_{epoch}.txt")
        with open(result_save_path, 'w', encoding='UTF-8') as f:
            for (a, b, c) in zip(all_dialog, all_response, all_generated):
                f.write('[DIALOG]\t%s\n[RESPONSE]\t%s\n[GENERATED]\t%s\n' % (a, b, c))
                f.write('-------------------------------------------\n')

        typelist = ['Q&A', 'POI recommendation', 'Movie recommendation', 'Music recommendation']
        # hitDic = {type: {'hit1': [], 'hit3': [], 'hit5': []} for type in typelist}
        hitAll = {'hit1': [], 'hit3': [], 'hit5': []}
        hit_list = [1]
        for idx in range(len(all_generated)):
            gold = all_response[idx]
            pred = all_generated[idx]
            goal_type = goal_types[idx]

            correct = (gold == pred)
            hitAll["hit1"].append(correct)

        print("[hit1]\t[%s]\t%.4f" % (subtask, np.average(hitAll[f"hit1"])))
        if best_hit < np.average(hitAll[f"hit1"]):
            best_hit = np.average(hitAll[f"hit1"])
            torch.save(generator.state_dict(), os.path.join(args.model_dir, f"{args.model_name}_{args.task}_{subtask}_{args.num_epochs}.pt"))  # TIME_MODELNAME 형식
    print("[BEST][hit1]\t[%s]\t%.4f" % (subtask, best_hit))


def write_goal_topic_result(args, generator, tokenizer, test_dataloader, subtask):
    test_dataloader.dataset.subtask = subtask
    print('[write_goal_topic_result] subtask is %s' % subtask)
    current = 0
    all_response = []
    all_generated = []

    for batch in tqdm(test_dataloader, desc=f"Generate_Predicted_{subtask}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        generator.eval()
        dialog_token = batch['input_ids'].to(args.device)
        dialog_mask = batch['attention_mask'].to(args.device)
        response = batch['response'].to(args.device)

        generated_goal = generator.gpt_model.generate(input_ids=dialog_token,
                                                      attention_mask=dialog_mask,
                                                      pad_token_id=tokenizer.pad_token_id,
                                                      max_length=args.max_gen_length)
        decoded_generated_goal = tokenizer.batch_decode(generated_goal, skip_special_tokens=True)
        all_generated.extend(decoded_generated_goal)
        all_response.extend(tokenizer.batch_decode(response, skip_special_tokens=True))

        for idx in range(len(decoded_generated_goal)):
            test_dataloader.dataset.augmented_raw_sample[current + idx][f"predicted_{subtask}"] = decoded_generated_goal[idx]
        current += dialog_token.size(0)

    hitAll = {'hit1': [], 'hit3': [], 'hit5': []}
    for idx in range(len(all_generated)):
        gold = all_response[idx]
        pred = all_generated[idx]
        correct = (gold == pred)
        hitAll["hit1"].append(correct)
    print("[hit1]\t[%s]\t%.4f" % (subtask, np.average(hitAll[f"hit1"])))

    return deepcopy(test_dataloader.dataset.augmented_raw_sample)
    # test_dataloader.dataset.subtask = 'topic'
