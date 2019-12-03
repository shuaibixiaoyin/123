import json
import random
import numpy as np

with open("train/train_author.json","r") as f:
    author_data = json.load(f)
with open("train/train_pub.json","r") as f:
    pubs_dict = json.load(f)
with open("cna_data/whole_author_profile.json", "r") as f:
    test_author_data = json.load(f)
with open("cna_data/whole_author_profile_pub.json", "r") as f:
    test_pubs_dict = json.load(f)
with open("cna_test_data/cna_test_unass_competition.json", "r") as f:
    unass_papers = json.load(f)
with open("cna_test_data/cna_test_pub.json", "r") as f:
    unass_papers_dict = json.load(f)

name_train = set()
for name in author_data:
    persons = author_data[name]
    if(len(persons) > 5):
        name_train.add((name))

pid_aname_aid = {}
for author_name in name_train:
    persons = author_data[author_name]
    for person in persons:
        paper_list = persons[person]
        for paper_id in paper_list:
            pid_aname_aid[paper_id] = (author_name, person)

total_paper_list = list(pid_aname_aid.keys())
train_paper_list = random.sample(total_paper_list, 500)
train_instances = []
for paper_id in train_paper_list:
    # 保存对应的正负例
    pos_ins = set()
    neg_ins = set()

    paper_author_name = pid_aname_aid[paper_id][0]
    paper_author_id = pid_aname_aid[paper_id][1]

    pos_ins.add((paper_id, paper_author_id))

    # 获取同名的所有作者(除了本身)作为负例的candidate
    persons = list(author_data[paper_author_name].keys())
    persons.remove(paper_author_id)

    # 每个正例采样5个负例
    neg_author_list = random.sample(persons, 5)
    for i in neg_author_list:
        neg_ins.add((paper_id, i))

    train_instances.append((pos_ins, neg_ins))
    
from pyjarowinkler import distance
def clean_name(name):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", "").replace("-", " ").replace("_", ' ').split()]
    # x = [k.strip() for k in name.lower().strip().replace("-", "").replace("_", ' ').split()]
    full_name = ' '.join(x)
    name_part = full_name.split()
    if(len(name_part) >= 1):
        return full_name
    else:
        return None

def delete_main_name(author_list, name):
    score_list = []
    name = clean_name(name)
    author_list_lower = []
    for author in author_list:
        author_list_lower.append(author.lower())
    name_split = name.split()
    for author in author_list_lower:
        # lower_name = author.lower()
        score = distance.get_jaro_distance(name, author, winkler=True, scaling=0.1)
        author_split = author.split()
        inter = set(name_split) & set(author_split)
        alls = set(name_split) | set(author_split)
        score += round(len(inter)/len(alls), 6)
        score_list.append(score)

    rank = np.argsort(-np.array(score_list))
    return_list = [author_list_lower[i] for i in rank[1:]]

    return return_list, rank[0]
    
def clean_keyword(keywords):
    word = [x.lower() for x in keywords]
    all_words = " ".join(word)
    word = all_words.split()
    set_keyword = set(word)
    return set_keyword
    
def process_feature(pos_ins, paper_coauthors):
    feature_list = []
    paper = pos_ins[0]
    author = pos_ins[1]
    paper_name = pid_aname_aid[paper][0]
    # 从作者的论文列表中把该篇论文去掉
    doc_list = []
    for doc in author_data[paper_name][author]:
        if(doc != paper):
            doc_list.append(doc)
    for doc in doc_list:
        if doc == paper:
            print("error!")
            exit()

    # 保存作者的所有paper的coauthors以及各自出现的次数(作者所拥有论文的coauthors)
    candidate_authors_int = defaultdict(int)

    total_author_count = 0
    for doc in doc_list:
        doc_dict = pubs_dict[doc]
        author_list = []
        paper_authors = doc_dict['authors']
        paper_authors_len = len(paper_authors)
        paper_authors = random.sample(paper_authors, min(50, paper_authors_len))

        for author in paper_authors:                
            clean_author = clean_name(author['name'])
            if(clean_author != None):
                author_list.append(clean_author)
        if(len(author_list) > 0):
            # 获取paper中main author_name所对应的位置
            _, author_index = delete_main_name(author_list, paper_name)

            # 获取除了main author_name外的coauthor
            for index in range(len(author_list)):
                if(index == author_index):
                    continue
                else:
                    candidate_authors_int[author_list[index]] += 1
                    total_author_count += 1

    # author 的所有不同coauthor name
    author_keys = list(candidate_authors_int.keys())

    if ((len(author_keys) == 0) or (len(paper_coauthors) == 0)):
        feature_list.extend([0.] * 5)
    else:
        co_coauthors = set(paper_coauthors) & set(author_keys)
        coauthor_len = len(co_coauthors)

        co_coauthors_ratio_for_paper = round(coauthor_len / len(paper_coauthors), 6)
        co_coauthors_ratio_for_author = round(coauthor_len / len(author_keys), 6)

        coauthor_count = 0
        for coauthor_name in co_coauthors:
            coauthor_count += candidate_authors_int[coauthor_name]

        co_coauthors_ratio_for_author_count = round(coauthor_count / total_author_count, 6)

        # 计算了5维paper与author所有的paper的coauthor相关的特征：
        #    1. 不重复的coauthor个数
        #    2. 不重复的coauthor个数 / paper的所有coauthor的个数
        #    3. 不重复的coauthor个数 / author的所有paper不重复coauthor的个数
        #    4. coauthor个数（含重复）
        #    5. coauthor个数（含重复）/ author的所有paper的coauthor的个数（含重复）
        feature_list.extend([coauthor_len,co_coauthors_ratio_for_paper, co_coauthors_ratio_for_author, coauthor_count, co_coauthors_ratio_for_author_count])
    #最后加上了一维关键字的特征，代表该篇论文的关键字与该作者所有其他论文的关键字的重复个数
    set_doc_keywords = set()
    set_paper_keywords = set()
    if "keywords" in pubs_dict[paper]:
        paper_keywords = pubs_dict[paper]["keywords"]
        set_paper_keywords = clean_keyword(paper_keywords)
        for doc in doc_list:
        
            doc_dict = pubs_dict[doc]
            if "keywords" in doc_dict:
                doc_keywords = doc_dict['keywords']
                set_doc_keywords |= clean_keyword(doc_keywords)
        
    set_keywords = set_paper_keywords & set_doc_keywords
    feature_list.append(len(set_keywords))
    return feature_list
    
def possible_name(author_list, name):
    score_list = []
    name = clean_name(name)
    author_list_lower = []
    for author in author_list:
        author_list_lower.append(author.lower())
    name_split = name.split()
    for author in author_list_lower:
        # lower_name = author.lower()
        score = distance.get_jaro_distance(name, author, winkler=True, scaling=0.1)
        author_split = author.split()
        inter = set(name_split) & set(author_split)
        alls = set(name_split) | set(author_split)
        score += round(len(inter)/len(alls), 6)
        score_list.append(score)
    rank = np.argsort(-np.array(score_list))
    return author_list[rank[0]]
def convert(author_list,name):
    if name!='':
        name = name.replace('_0001','')
    set_name = set(name.split('_'))
    for x in author_list:
        set_x=set(x.split('_'))
        if set_name==set_x:
            name=x
    return name

from collections import defaultdict

pos_features = []
neg_features = []

for ins in train_instances:
    pos_set = ins[0]
    neg_set = ins[1]
    paper_id = list(pos_set)[0][0]
    paper_name = pid_aname_aid[paper_id][0]

    author_list = []
    # 获取paper的coauthors
    paper_coauthors = []
    paper_authors = pubs_dict[paper_id]['authors']
    paper_authors_len = len(paper_authors)
    # 只取前50个author以保证效率
    paper_authors = random.sample(paper_authors, min(50, paper_authors_len))

    for author in paper_authors:                
        clean_author = clean_name(author['name'])
        if(clean_author != None):
            author_list.append(clean_author)
    if(len(author_list) > 0):
        # 获取paper中main author_name所对应的位置
        _, author_index = delete_main_name(author_list, paper_name)

        # 获取除了main author_name外的coauthor
        for index in range(len(author_list)):
            if(index == author_index):
                continue
            else:
                paper_coauthors.append(author_list[index])


        for pos_ins in pos_set:
            pos_features.append(process_feature(pos_ins, paper_coauthors))

        for neg_ins in neg_set:
            neg_features.append(process_feature(neg_ins, paper_coauthors))

# 构建特征和标签
m_train_ins = []
for ins in pos_features:
    m_train_ins.append((ins, 1))

for ins in neg_features:
    m_train_ins.append((ins, 0))

x_train= []
y_train = []
for ins in m_train_ins:
    x_train.append(ins[0])
    y_train.append(ins[1])
    
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)#使用K折交叉验证
x_train_np=np.array(x_train)
y_train_np=np.array(y_train)
#训练模型
for index, (tr_idx, va_idx) in enumerate(kfold.split(x_train_np)):
    print('*' * 30)
    X_train, y_train, X_valid, y_valid = x_train_np[tr_idx], y_train_np[tr_idx], x_train_np[va_idx], y_train_np[va_idx]
    cate_features = []
    train_pool = Pool(X_train, y_train, cat_features=cate_features)
    eval_pool = Pool(X_valid, y_valid,cat_features=cate_features)
    cbt_model = CatBoostClassifier(iterations=10000,
                           learning_rate=0.1,
                           eval_metric='AUC',
                           use_best_model=True,
                           random_seed=42,
                           logging_level='Verbose',
                           early_stopping_rounds=300,
                           loss_function='Logloss',
                           )
    clf=cbt_model.fit(train_pool, eval_set=eval_pool, verbose=100)
    
new_test_author_data = {}
for author_id, author_info in test_author_data.items():
    author_name = author_info['name']
    author_papers = author_info['papers']
    newly_papers = []

    for paper_id in author_papers:

        paper_authors = test_pubs_dict[paper_id]['authors']
        paper_authors_len = len(paper_authors)

        # 只利用author数小于50的paper，以保证效率
        if(paper_authors_len > 50):
            continue
        author_list = []
        for author in paper_authors:                
            clean_author = clean_name(author['name'])
            if(clean_author != None):
                author_list.append(clean_author)
        if(len(author_list) > 0):
            _, author_index = delete_main_name(author_list, author_name)
            new_paper_id = str(paper_id) + '-' + str(author_index)
            newly_papers.append(new_paper_id)


    if(new_test_author_data.get(author_name) != None):
        new_test_author_data[author_name][author_id] = newly_papers
    else:
        tmp = {}
        tmp[author_id] = newly_papers
        new_test_author_data[author_name] = tmp
        
# test集的特征生成函数，与train类似
def process_test_feature(pair, new_test_author_data, test_pubs_dict, paper_coauthors):

    feature_list = []

    paper = pair[0]
    author = pair[1]
    paper_name = pair[2]

    doc_list = new_test_author_data[paper_name][author]
    # 保存作者的所有coauthors以及各自出现的次数(作者所拥有论文的coauthors)
    candidate_authors_int = defaultdict(int)

    total_author_count = 0
    for doc in doc_list:
        doc_id = doc.split('-')[0]
        author_index = doc.split('-')[1]
        doc_dict = test_pubs_dict[doc_id]
        author_list = []

        paper_authors = doc_dict['authors']
        paper_authors_len = len(paper_authors)
        paper_authors = random.sample(paper_authors, min(50, paper_authors_len))

        for author in paper_authors:                
            clean_author = clean_name(author['name'])
            if(clean_author != None):
                author_list.append(clean_author)
        if(len(author_list) > 0):
            for index in range(len(author_list)):
                if(index == author_index):
                    continue
                else:
                    candidate_authors_int[author_list[index]] += 1
                    total_author_count += 1

    author_keys = list(candidate_authors_int.keys())

    if ((len(author_keys) == 0) or (len(paper_coauthors) == 0)):
        feature_list.extend([0.] * 5)
    else:
        co_coauthors = set(paper_coauthors) & set(author_keys)
        coauthor_len = len(co_coauthors)
        co_coauthors_ratio_for_paper = round(coauthor_len / len(paper_coauthors), 6)
        co_coauthors_ratio_for_author = round(coauthor_len / len(author_keys), 6)
        coauthor_count = 0
        for coauthor_name in co_coauthors:
            coauthor_count += candidate_authors_int[coauthor_name]
        co_coauthors_ratio_for_author_count = round(coauthor_count / total_author_count, 6)
        feature_list.extend([coauthor_len, co_coauthors_ratio_for_paper, co_coauthors_ratio_for_author, coauthor_count, co_coauthors_ratio_for_author_count])
    set_doc_keywords = set()
    set_paper_keywords = set()
    if "keywords" in unass_papers_dict[paper]:
        paper_keywords = unass_papers_dict[paper]["keywords"]
        set_paper_keywords = clean_keyword(paper_keywords)
        for doc in doc_list:
            doc_id = doc.split('-')[0]
            doc_dict = test_pubs_dict[doc_id]
            if "keywords" in doc_dict:
                doc_keywords = doc_dict['keywords']
                set_doc_keywords |= clean_keyword(doc_keywords)
        
    set_keywords = set_paper_keywords & set_doc_keywords
    feature_list.append(len(set_keywords))
    return feature_list  
    
all_author_name = list(new_test_author_data.keys())
count = 0

# 存储paper的所有candidate author id
paper2candidates = defaultdict(list)
# 存储对应的paper与candidate author的生成特征
paper2features = defaultdict(list)

for u_p in unass_papers:
    paper_id = u_p.split('-')[0]
    author_index = int(u_p.split('-')[1])
    author_list = []

    # 获取paper的coauthors
    paper_coauthors = []
    paper_name = ''
    paper_authors = unass_papers_dict[paper_id]['authors']

    for author in paper_authors:                
        clean_author = clean_name(author['name'])
        if(clean_author != None):
            author_list.append(clean_author)
    if(len(author_list) > 0):
        for index in range(len(author_list)):
            if(index == author_index):
                continue
            else:
                paper_coauthors.append(author_list[index])

    if paper_authors[author_index]['name'].strip()!='':
        paper_name = '_'.join(clean_name(paper_authors[author_index]['name']).split())
        paper_name = convert(all_author_name,paper_name)
        paper_name = possible_name(all_author_name,paper_name)
    else:
        paper_name = ''
    if(new_test_author_data.get(paper_name) != None):
        candidate_author_list = new_test_author_data[paper_name]
        for candidate_author in candidate_author_list:
            pair = (paper_id, candidate_author, paper_name)
            paper2candidates[paper_id].append(candidate_author)
            paper2features[paper_id].append(process_test_feature(pair, new_test_author_data, test_pubs_dict, paper_coauthors))
        count += 1
print(count)
print(len(paper2candidates))

### 利用训练好的 CatBoost 模型去预测
result_dict = defaultdict(list)
for paper_id, ins_feature_list in paper2features.items():
    score_list = []
    for ins in ins_feature_list:
        # 利用CatBoost对一篇paper的所有可能的作者ID去打分，利用分数进行排序，取分数最高作者作为预测的作者
        prob_pred = clf.predict_proba([ins])[:, 1]
        score_list.append(prob_pred[0])
    rank = np.argsort(-np.array(score_list))
    predict_author = paper2candidates[paper_id][rank[0]]
    result_dict[predict_author].append(paper_id)

#生成结果文件
with open("result.json", 'w') as files:
    json.dump(result_dict, files, indent = 4)