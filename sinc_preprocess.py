import sys
import json
import pdb
from transformers import BertTokenizer

MODEL_PATH = 'hfl/chinese-macbert-base'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

def conv_to_json(fin_file, fout_file):
    """
    原始数据集转换为Query生成模型训练所需的格式
    """
    fout = open(fout_file, "w", encoding="utf-8")
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            input_query = line[:-1].split('\t')
            model_input, query = input_query[0], input_query[-1]
            model_input = preprocess_function(model_input, tokenizer)
            if(query == '不检索'):
                fout.write(str(json.dumps({'sentence':model_input, 'label':"0", 'label_des':query}, ensure_ascii=False) + '\n'))
            else:
                fout.write(str(json.dumps({'sentence':model_input, 'label':"1", 'label_des':query}, ensure_ascii=False) + '\n'))
    fout.close()

def conv_to_json_test(fin_file, fout_file):
    """
    原始数据集转换为Query生成模型训练所需的格式
    """
    fout = open(fout_file, "w", encoding="utf-8")
    cnt = 0
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            model_input = line[:-1]
            fout.write(str(json.dumps({'sentence':model_input, 'label_id':cnt}, ensure_ascii=False) + '\n'))
            cnt += 1
    fout.close()

def conv_to_json_ocnli(fin_file, fout_file):
    fout = open(fout_file, "w", encoding="utf-8")
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            input_query = line[:-1].split('\t')
            model_input, query = input_query[0], input_query[-1]
            model_input = preprocess_function(model_input, tokenizer)
            if(query == '不检索'):
                fout.write(str(json.dumps({'sentence1':model_input, "sentence2":"检索", 'label':"contradiction", 'label_des':"不检索"}, ensure_ascii=False) + '\n'))
            else:
                fout.write(str(json.dumps({'sentence1':model_input, "sentence2":"检索", 'label':"entailment", 'label_des':"需要检索"}, ensure_ascii=False) + '\n'))
    fout.close()

def preprocess_function(inputs, tokenizer, context_pos=2, max_source_length=512):
    """考虑加一个手动truncation，把context从前面截断"""
    max_len = max_source_length
    input_encode = tokenizer.encode(inputs)
    if(len(input_encode) > max_len):
        input_trunc = []
        sep_cnt = 0
        trunc_len = len(input_encode) - max_len
        for token in input_encode:
            if(token == tokenizer.convert_tokens_to_ids('[SEP]')):
                sep_cnt += 1
            if(sep_cnt < context_pos):
                input_trunc.append(token)
                continue
            if(sep_cnt == context_pos and token == tokenizer.convert_tokens_to_ids('[SEP]')):
                input_trunc.append(token)
                continue
            # context位置
            if(trunc_len > 0):
                trunc_len -= 1
                continue
            input_trunc.append(token)
        assert(len(input_trunc) <= max_len)
        input_encode = input_trunc
    return tokenizer.decode(input_encode[1:-1]).replace(" ", "")

def conv_to_json_ocnli_test(fin_file, fout_file):
    fout = open(fout_file, "w", encoding="utf-8")
    cnt = 0
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            input_query = line[:-1].split('\t')
            model_input = input_query[0]
            model_input = preprocess_function(model_input, tokenizer)
            fout.write(str(json.dumps({'sentence1':model_input, "sentence2":"检索", 'label_id':cnt}, ensure_ascii=False) + '\n'))
            cnt += 1
    fout.close()

def conv_to_json_ocnli_query(fin_file, fout_file):
    fout = open(fout_file, "w", encoding="utf-8")
    cnt = 0
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            input_query = line[:-1].split('\t')
            model_input = input_query[0]
            model_input = preprocess_function(model_input, tokenizer)
            fout.write(str(json.dumps({'sentence1':model_input, "sentence2":"检索", 'label_id':cnt}, ensure_ascii=False) + '\n'))
            cnt += 1
    fout.close()

def conv_to_json_ocnli_triple(fin_file, fout_file, ref_file):
    fin_all_search = open(ref_file, "r", encoding="utf-8")
    model_gen = []
    gt = []
    for line in fin_all_search:
        gen_gt = line[:-1].split('\t')
        model_gen.append(gen_gt[0])
        gt.append(gen_gt[1])
    fout = open(fout_file, "w", encoding="utf-8")
    with open(fin_file, encoding="utf-8") as f:
        for line, mg in zip(f, model_gen):
            input_query = line[:-1].split('\t')
            model_input, query = input_query[0], input_query[-1]
            model_input = preprocess_function(model_input, tokenizer)
            if(query == mg):
                fout.write(str(json.dumps({'sentence1':model_input, "sentence2":mg, 'label':"entailment", 'label_des':"模型生成和gt完全一致"}, ensure_ascii=False) + '\n'))
            elif(query != mg and query !=  '不检索'):
                fout.write(str(json.dumps({'sentence1':model_input, "sentence2":mg, 'label':"neutral", 'label_des':"模型生成和gt不完全一致"}, ensure_ascii=False) + '\n'))
                fout.write(str(json.dumps({'sentence1':model_input, "sentence2":query, 'label':"entailment", 'label_des':"模型生成和gt不完全一致"}, ensure_ascii=False) + '\n'))
            elif(query == '不检索'):
                fout.write(str(json.dumps({'sentence1':model_input, "sentence2":mg, 'label':"contradiction", 'label_des':"不检索"}, ensure_ascii=False) + '\n'))
    fout.close()


def conv_to_json_ocnli_triple_test(fin_file, fout_file, ref_file):
    fin_all_search = open(ref_file, "r", encoding="utf-8")
    model_gen = []
    for line in fin_all_search:
        model_gen.append(line[:-1])
    fout = open(fout_file, "w", encoding="utf-8")

    cnt = 0

    with open(fin_file, encoding="utf-8") as f:
        for line, mg in zip(f, model_gen):
            model_input = line[:-1]
            model_input = preprocess_function(model_input, tokenizer)
            fout.write(str(json.dumps({'sentence1':model_input, "sentence2":mg, 'label_id':cnt}, ensure_ascii=False) + '\n'))
            cnt += 1  
    fout.close()


if __name__ == '__main__':    
    conv_to_json_ocnli('DuSinc/query_topic_onlyquery/train.csv', 'CLUE/ocnli/train.50k.json')
    conv_to_json_ocnli('DuSinc/query_topic_onlyquery/dev.csv', 'CLUE/ocnli/dev.json')
    conv_to_json_ocnli_test('DuSinc/query_topic_onlyquery/test_.csv', 'CLUE/ocnli/test.json')

    """
    conv_to_json_ocnli_triple('DuSinc/query_topic_onlyquery/train_.csv', 'CLUE/ocnli/train.50k.json', 'train_output.txt')
    conv_to_json_ocnli_triple('DuSinc/query_topic_onlyquery/dev_.csv', 'CLUE/ocnli/dev.json', 'dev_output.txt')
    conv_to_json_ocnli_triple_test('DuSinc/query_topic_onlyquery/test_.csv', 'CLUE/ocnli/test.json', 'test_output.txt')
    """