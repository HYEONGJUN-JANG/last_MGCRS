from numpy import Inf
from torch import save as saveTorch

class EarlyStopping:
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
        self.val_loss_min = Inf
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

def scroing(test_preds, test_labels):
    from sklearn.metrics import precision_score, recall_score, f1_score
    if isinstance(test_preds[0], list): # Top5의 경우
        correct=[1 if label in preds else 0 for preds, label in zip(test_preds, test_labels)]
        return round(sum(correct)/len(correct), 3)
    else:
        p,r,f = round(precision_score(test_labels, test_preds, average='macro', zero_division=0), 3), round(recall_score(test_labels, test_preds, average='macro', zero_division=0), 3), round(f1_score(test_labels, test_preds, average='macro', zero_division=0), 3)
        return p,r,f