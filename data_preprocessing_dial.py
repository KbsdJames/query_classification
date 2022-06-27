import sys
import json
import pdb
import re
from transformers import BertTokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu
import rouge
from rouge import Rouge


know_len = 200
MODEL_PATH = 'hfl/chinese-macbert-base'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

bleu = []
sentence_id = []

rouge = Rouge()

def cut_text(text,lenth): 
    textArr = re.findall('.{'+str(lenth)+'}', text) 
    textArr.append(text[(len(textArr)*lenth):]) 
    return textArr


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
    return tokenizer.decode(input_encode[1:-1])

def conv_to_json(fin_file, fout_file):
    """
    原始数据集转换为Query生成模型训练所需的格式
    """
    fout = open(fout_file, "w", encoding="utf-8")
    with open(fin_file, encoding="utf-8") as f:
        line_num = 0
        for line in f:
            line_num += 1
            input_query = line[:-1].split('\t')
            model_input, answer = input_query[0], input_query[-1]

            model_input = model_input.split('[SEP]')
            #pdb.set_trace()
            knowledge = model_input[0]
            if(len(bleu) < 1000):
                b = sentence_bleu([list(answer)], list(knowledge))
                if(b > 0.2):
                    bleu.append(b)
                    sentence_id.append(line_num)

            model_input = preprocess_function(model_input, tokenizer)
            """
            if(query == '不检索'):
                fout.write(str(json.dumps({'sentence':model_input, 'label':"0", 'label_des':query}, ensure_ascii=False) + '\n'))
            else:
                fout.write(str(json.dumps({'sentence':model_input, 'label':"1", 'label_des':query}, ensure_ascii=False) + '\n'))
    fout.close()"""


def conv_to_gen_summary(fin_file, fout_file, is_test=False):
    """
    原始数据集转换为知识对话生成模型训练所需的格式
    """
    fout = open(fout_file, "w", encoding="utf-8")
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            context = []
            topical = " ".join(data["user_topical"])
            location = data["user_location"]
            if is_test:
                context = [uttr["utterance"]
                           for uttr in data["conversation"][:-1]]
                if "use_knowledge" in data["conversation"][-1]:
                    knowledge = data["conversation"][-1]["use_knowledge"].replace(
                        " ", "")
                else:
                    continue
                # 对部分过长的知识进行截断，只保留前256个字符
                knowledge = knowledge.replace(
                    "\n", " ").replace("\t", " ")[:256]
                outstr = topical + "[SEP]" + location + "[SEP]" + knowledge + '[SEP]' + '[SEP]'.join(context)
                model_input = outstr.replace("\n", " ").replace("\t", " ").replace(" ", "");
                fout.write(model_input + '\n')
                continue
            for uttr in data["conversation"]:
                if is_test:
                    context.append(uttr["utterance"])
                    continue
                if "use_kg_label" in uttr:
                    if uttr["use_kg_label"] == "true":
                        try:
                            knowledge = uttr["use_knowledge"].replace(
                                "\n", " ").replace("\t", " ")[:256]
                        except:
                            print(json.dumps(uttr, ensure_ascii=False, indent=2))
                    else:
                        continue

                    response_lst = re.split('[,。！？，~～]', uttr["utterance"])
                    res = ""
                    for response in response_lst:
                        if(response == ''):
                            continue
                        score = rouge.get_scores(hyps=response, refs=knowledge, avg=True)
                        if(score['rouge-l']['p'] >= 0.45):
                            if(res == ""):
                                res = response
                            else:
                                res += ","+response
                    if(len(res) <= 3):
                        continue
                    res += '。'
                    outstr = topical + "[SEP]" + location + "[SEP]" + knowledge + '[SEP]' + '[SEP]'.join(context) + "\t" + res
                    outstr = outstr.replace("\n", "，")
                    if(len(outstr.split("\n")) > 2):
                        pdb.set_trace()
                    fout.write(outstr.replace(" ", "") + "\n")
                context.append(uttr["utterance"].replace(" ", "").replace("\n", " ").replace("\t", " "))
    fout.close()



def conv_to_gen_summary(fin_file, fout_file, is_test=False):
    """
    原始数据集转换为知识对话生成模型训练所需的格式
    """
    fout = open(fout_file, "w", encoding="utf-8")
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            context = []
            topical = " ".join(data["user_topical"])
            location = data["user_location"]
            if is_test:
                context = [uttr["utterance"]
                           for uttr in data["conversation"][:-1]]
                if "use_knowledge" in data["conversation"][-1]:
                    knowledge = data["conversation"][-1]["use_knowledge"].replace(
                        " ", "")
                else:
                    continue
                # 对部分过长的知识进行截断，只保留前256个字符
                knowledge = knowledge.replace(
                    "\n", " ").replace("\t", " ")[:256]
                outstr = topical + "[SEP]" + location + "[SEP]" + knowledge + '[SEP]' + '[SEP]'.join(context)
                model_input = outstr.replace("\n", " ").replace("\t", " ").replace(" ", "");
                fout.write(model_input + '\n')
                continue
            for uttr in data["conversation"]:
                if is_test:
                    context.append(uttr["utterance"])
                    continue
                if "use_kg_label" in uttr:
                    if uttr["use_kg_label"] == "true":
                        try:
                            knowledge = uttr["use_knowledge"].replace(
                                "\n", " ").replace("\t", " ")[:256]
                        except:
                            print(json.dumps(uttr, ensure_ascii=False, indent=2))
                    else:
                        continue

                    response_lst = re.split('[,。！？，~～]', uttr["utterance"])
                    res = ""
                    for response in response_lst:
                        if(response == ''):
                            continue
                        score = rouge.get_scores(hyps=response, refs=knowledge, avg=True)
                        if(score['rouge-l']['p'] >= 0.45):
                            if(res == ""):
                                res = response
                            else:
                                res += ","+response
                    if(len(res.replace(" ", "")) <= 3):
                        continue
                    res += '。'
                    outstr = topical + "[SEP]" + location + "[SEP]" + knowledge + '[SEP]' + '[SEP]'.join(context) + "\t" + res
                    outstr = outstr.replace("\n", "，")

                    fout.write(outstr.replace(" ", "") + "\n")
                context.append(uttr["utterance"].replace(" ", "").replace("\n", " ").replace("\t", " "))
    fout.close()

#conv_to_json("DuSinc/dialogue/train.csv", "DuSinc/dialogue/train.json")
#conv_to_json("DuSinc/dialogue/test.csv", "DuSinc/diaologue/test.json")

conv_to_gen_summary("DuSinc_release/train.txt", "DuSinc/summary/train_rougel_context.csv", is_test=False)
conv_to_gen_summary("DuSinc_release/dev.txt", "DuSinc/summary/dev_context.csv", is_test=False)
conv_to_gen_summary("DuSinc_release/test_dial_1.txt", "DuSinc/summary/test_context.csv", is_test=True)