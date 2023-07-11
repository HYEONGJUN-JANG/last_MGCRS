# 프로젝트

BigDas Lab - CRS Team project

---

# 1. 구현 명세

파일목록

## main.py

- Main starting point
- training, evalutation, test 진행

utils로부터 parsed argument 받아오기

data로부터 각 dataset 받아오기

model로 부터 model 받아오기

training, eval, test 진행 (metric 들 활용)

## utils.py

- main을 수행하는데 있어 도움되는 유틸함수들
    - parser, save_output 등
    - read_pkl, write_pkl

## data.py

- data preprocessing 및 dataset 관리

data_util로 부터 필요 함수들 호출하여 사용

## data_util.py

- data.py에서 특수하게 수행이 필요할경우 수행에 도움되는 유틸함수들

## models.py

- 모델 관리

## metric.py

- 각 평가함수들 관리

# 2. Directory 관리

```bash
HOME
|-logs
|---1	
|---2
|-model_cache
|	|--bert-base-uncased
|-epoch_output
|	|--1	
|	|--2
|		|--ours
|-model_save
|	|--1	
|	|--2	
|		|--ours
|			|--[Saved Retriever BERT.pt]
|-data
|	|--1	
|	|--2
|		|--cached
|			|--[Preprocessed file cache]
|		|--pred_aug
|			|- # Goal, Topic predicted labeled pkl file
|		|--en_dev.txt
|		|--en_train.txt
|		|--en_test.txt
|-model_play
|	|--1	
|	|--2
|		|--ours
|			|--goal_topic.py
|			|--knowledge_retrieve.py
|			...
|-main.py # ours
|-data.py
|-data_utils.py
|-utils.py
|-models.py
|-metric.py
```

---

# main.py

## main()

- Main starting point
- training, evalutation, test 진행 → 향후 코드 길어질경우 분리예정

## train()

`def train(args, train_dataloader,  knowledge_index, bert_model ):`

## evaluation()

## test()

# dataModel.py

## class KnowledgeDataset(Dataset)

- `def __init__(self, knowledgeDB, max_length, tokenizer):`
    - asd
- `def __getitem__(self, item):`
    - `return tokens, mask`

## class DialogDataset(Dataset)

- `def __init__(self, train_sample):`
- `def __getitem__(self, idx):`
    - `return dialog_token, dialog_mask, target_knowledge, goal_type, response, topic`

# data_utils.py

# utils.py

- `get_time_kst()`
    - 현재 한국 시간을 지정된 form으로 return
    - `return: (str)'%Y-%m-%d_%H%M%S'`
- `write_pkl(obj, filename)`
    - filename의 해당 file이름으로 obj 를 pickle 파일로 저장
    - `return: None`
- `read_pkl(obj, filename)`
    - 해당 file이름의 pickle 파일을 load하여 return
    - `return: (obj) object`
- `parseargs()`
    - CMD의 입력과 기본세팅을 args로 return
    - `return: args`
- print_json(args, filename, saved_jsonlines)
    - 지정된 format에 맞게 해당 `args.data_dir/print/filename.txt` 의 위치에 saved_jsonlines 저장
    - `return: None`

# models.py

- class Retriever(nn.Module):
    - `init: bert_model, hiddensize`
        - bert 모델과 mlp 레이어로 구성
    - `mlp size: hidden/4`
    - forward: bert_model → mlp
        - `return: tensor (B,hidden/4)`

- class Generator(nn.Module):
    - `init:`
        - description
    - forward: asdf → bfewafw
        - `return: tensor (B, xxx)`

- 

# metric.py