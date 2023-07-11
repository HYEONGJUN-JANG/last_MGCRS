import copy
import torch
from torch import nn

class Retriever(nn.Module):
    def __init__(self, args, query_bert=None, gpt_model=None):
        super(Retriever, self).__init__()
        self.args = args
        self.query_bert = query_bert  # Knowledge text 처리를 위한 BERT
        self.rerank_bert = copy.deepcopy(self.query_bert)

        # if args.know_ablation == 'negative_sampling':
        #     self.key_bert = query_bert
        # else:
        #     self.key_bert = copy.deepcopy(self.query_bert)
        #     self.key_bert.requires_grad = False

        self.gpt_model = gpt_model
        self.hidden_size = args.hidden_size
        self.goal_proj  = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Linear(self.hidden_size//2, args.goal_num)) #nn.Linear(self.hidden_size, args.goal_num)
        self.topic_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Linear(self.hidden_size//2, args.topic_num))# nn.Linear(self.hidden_size, args.topic_num)

        self.linear_proj = nn.Linear(self.hidden_size, 1)
        # self.know_proj = nn.Linear(self.hidden_size, self.args.knowledge_num, bias=False)
        self.goal_embedding = nn.Embedding(self.args.goal_num, self.args.hidden_size)
        nn.init.normal_(self.goal_embedding.weight, 0, self.args.hidden_size ** -0.5)

    def init_reranker(self):
        self.rerank_bert = copy.deepcopy(self.query_bert)

    # def init_know_proj(self, weights):
    #     self.know_proj.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, input_ids, attention_mask):
        dialog_emb = self.query_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        return dialog_emb

    def generation(self, input_ids, attention_mask, labels):
        outputs = self.gpt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)  # , output_hidden_states=True)
        # outputs = self.gpt_model(input_ids=token_seq, labels=labels)
        # logit = self.topic_proj(outputs.decoder_hidden_states[-1][:, 0, :])
        # loss = torch.nn.CrossEntropyLoss()(logit, label_idx)

        return outputs[0]

    def compute_know_score(self, token_seq, mask, knowledge_index, type_idx):
        """
        eval_know.computing_score에서
        모든 key vector에서 올라온 벡터를 통해 계산처리
        """
        dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        # dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state  # [B, L, d]
        # dialog_emb = torch.sum(dialog_emb * mask.unsqueeze(-1), dim=1) / (torch.sum(mask, dim=1, keepdim=True) + 1e-20)  # [B, d]

        dot_score = torch.matmul(dialog_emb, knowledge_index.transpose(1, 0))  # [B, N]
        # if self.args.type_aware:
        #     type_emb = self.goal_embedding(type_idx)  # [B, d]
        #     dot_score = self.know_proj(dialog_emb+type_emb)
        # else:
        #     dot_score = self.know_proj(dialog_emb)
        return dot_score

    def compute_know_score_candidate(self, token_seq, mask, knowledge_index):
        """
        eval_know.computing_score에서
        모든 key vector에서 올라온 벡터를 통해 계산처리
        """
        if self.args.siamese:
            dialog_emb = self.rerank_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        else:
            dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        # dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state  # [B, L, d]
        # dialog_emb = torch.sum(dialog_emb * mask.unsqueeze(-1), dim=1) / (torch.sum(mask, dim=1, keepdim=True) + 1e-20)  # [B, d]

        dot_score = torch.sum(knowledge_index * dialog_emb.unsqueeze(1), dim=-1)  # [B, K, d] x [B, 1, d]
        return dot_score

    def dpr_retrieve_train(self, token_seq, mask, candidate_knowledge_token, candidate_knowledge_mask):
        batch_size = mask.size(0)

        dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]

        candidate_knowledge_token = candidate_knowledge_token.view(-1, self.args.max_length)  # [B*K, L]
        candidate_knowledge_mask = candidate_knowledge_mask.view(-1, self.args.max_length)  # [B*K, L]
        knowledge_index = self.rerank_bert(input_ids=candidate_knowledge_token, attention_mask=candidate_knowledge_mask).last_hidden_state[:, 0, :]  # [B*K, L]
        knowledge_index = knowledge_index.view(batch_size, -1, dialog_emb.size(-1))  # [B, K, d]

        knowledge_index_pos = knowledge_index[:, :self.args.pseudo_pos_rank, :].squeeze(1)  # [B, d]
        knowledge_index_neg = knowledge_index[:, self.args.pseudo_pos_rank:, :].squeeze(1)  # [B, d]

        logit = torch.matmul(dialog_emb, knowledge_index_pos.transpose(1, 0))  # [B, B]
        logit_hn = torch.sum(dialog_emb * knowledge_index_neg, dim=-1, keepdim=True)  # [B, 1]
        logit = torch.cat([logit, logit_hn], dim=-1)  # [B, B+1]
        loss = -torch.log_softmax(logit, dim=-1).diagonal().mean()  # [B]
        return loss

    def knowledge_retrieve(self, token_seq, mask, candidate_indice, candidate_knowledge_token, candidate_knowledge_mask, ablation=None, labels=None):
        """
        Args: 뽑아준 negative에 대해서만 dot-product
            token_seq: [B, L]
            mask: [B, L]
            candidate_knowledge_token: [B, K+1, L]
            candidate_knowledge_mask: [B, K+1, L]
        Returns:
        """
        batch_size = mask.size(0)
        if ablation is None:
            ablation = self.args.know_ablation
        # dot-product
        # if self.args.usebart:
        #     dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask, output_hidden_states=True).decoder_hidden_states[-1][:, 0, :].squeeze(1)
        # else:
        #     dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        # if self.args.siamese:
        #     dialog_emb = self.rerank_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        # else:
        #     dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]
        dialog_emb = self.query_bert(input_ids=token_seq, attention_mask=mask).last_hidden_state[:, 0, :]  # [B, d]

        candidate_knowledge_token = candidate_knowledge_token.view(-1, self.args.max_length)  # [B*K, L]
        candidate_knowledge_mask = candidate_knowledge_mask.view(-1, self.args.max_length)  # [B*K, L]
        knowledge_index = self.rerank_bert(input_ids=candidate_knowledge_token, attention_mask=candidate_knowledge_mask).last_hidden_state[:, 0, :]  # [B*K, L]
        knowledge_index = knowledge_index.view(batch_size, -1, dialog_emb.size(-1))  # [B, K, d]

        knowledge_index_pos = knowledge_index[:, :self.args.pseudo_pos_rank, :]  # [B, 1, d]
        knowledge_index_neg = knowledge_index[:, self.args.pseudo_pos_rank:, :]  # [B, N, d]
        knowledge_index_neg = knowledge_index_neg.reshape(-1, self.hidden_size)
        # inbatch_index = torch.repeat(knowledge_index_pos.squeeze(1).repeat(batch_size, 1))  # [B * B, d]
        # inbatch_index = inbatch_index.view(batch_size, batch_size, -1)  # [B, B, d]
        #
        # logit_inbatch = torch.sum(dialog_emb.unsqueeze(1) * knowledge_index_pos, dim=2)  # [B, B]

        logit_pos = torch.sum(dialog_emb.unsqueeze(1) * knowledge_index_pos, dim=2)  # [B, 1, d] * [B, K, d] = [B, K]
        logit_neg = torch.matmul(dialog_emb, knowledge_index_neg.transpose(1, 0))  # [B, N*B]
        # logit_mask = torch.zeros_like(logit_neg).fill_diagonal_(-1e10)
        # logit_neg = logit_neg + logit_mask  # [B, B]

        # logit_neg = torch.matmul(dialog_emb, knowledge_index_neg.squeeze(1).transpose(1, 0))  # [B, d] x [d, B] = [B, B]
        # logit = torch.cat([logit_pos, logit_inbatch_neg], dim=-1)  # [B, K+B]
        return logit_pos, logit_neg

    def earlyStopping(self):pass ## TODO

class EarlyStopping: ## TODO
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, args,patience=7, verbose=False, delta=0.004, path='checkpoint.pt'):
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
        self.val_loss_min = 1000000
        self.delta = delta
        self.path = path

    def __call__(self, score, model):
        score = score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score * (1+self.delta):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and self.args.earlystop: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose: print(f'Use EarlyStop: {self.args.earlystop} || Validation score increased ({self.val_loss_min:.6f} --> {score:.6f}).  Saving model ... in {self.path}')
        saveTorch(model.state_dict(), self.path)
        self.val_loss_min = score