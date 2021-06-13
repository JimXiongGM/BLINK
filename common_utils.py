import sys
import os
import json
import pickle
import datetime
import html
import numpy as np
import gzip
import argparse
import random

"""
超级常用的函数！
"""

# I/O


def read_json(path="test.json"):
    with open(path, "r", encoding="utf-8") as f1:
        res = json.load(f1)
    return res


def save_to_json(obj, path):
    if type(obj) == set:
        obj = list(obj)
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f1:
        json.dump(obj, f1, ensure_ascii=False, indent=4)
    print(f"SAVE: {path}")


def read_pkl(path="test.pkl"):
    with open(path, "rb") as f1:
        res = pickle.load(f1)
    return res


def save_to_pkl(obj, path):
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "wb") as f1:
        pickle.dump(obj, f1)
    print(f"SAVE: {path}")


def read_jsonl(path="test.jsonl"):
    with open(path, "r", encoding="utf-8") as f1:
        res = [json.loads(line.strip()) for line in f1]
    return res


def save_to_jsonl(obj, path):
    if type(obj) == set:
        obj = list(obj)
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f1:
        for line in obj:
            f1.write(json.dumps(line) + "\n")
    print(f"SAVE: {path}")


def save_to_gzip(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = bytes(data, "utf8")
    with gzip.open(path, "wb") as f:
        f.write(data)
    print(f"SAVE: {path}")

def get_filename(path):
    """
    去除路径和扩展名的文件名
    """
    return os.path.splitext(os.path.basename(path))[0]

# model
def freeze_to_layer_by_name(model, layer_name):
    """
    冻结层. 从0到layer_name，闭区间，只要layer_name in就算
    """
    if layer_name is None:
        return
    if layer_name == "all":
        index_start = len(model.state_dict())
    else:
        index_start = -1
        for index, (key, _value) in enumerate(model.state_dict().items()):
            if layer_name in key:
                index_start = index
                break

    if index_start < 0:
        print(f"Don't find layer name: {layer_name}")
        print(f"must in : \n{model.state_dict().keys()}")

    grad_nums = 0
    for index, i in enumerate(model.parameters()):
        if index >= index_start:
            i.requires_grad = True
            grad_nums += 1
        else:
            i.requires_grad = False
    print(
        f"freeze layers num: {index_start + 1}, active layers num: {grad_nums}.")


# func
def time_now():
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return date_time

def split_train_dev_test(items, ratio=[0.7, 0.2, 0.1], seed=123):
    """
    切分数据
    """
    assert abs(sum(ratio) - 1.0) < 1e-9
    ratio.sort()
    ratio = ratio[::-1]
    random.seed(seed)
    random.shuffle(items)
    if len(ratio) == 2:
        _num = int(len(items) * ratio[0])
        return items[:_num], items[_num:]
    elif len(ratio) == 3:
        _num1 = int(len(items) * ratio[0])
        _num2 = int(len(items) * (ratio[0]+ratio[1]))
        return items[:_num1], items[_num1:_num2], items[_num2:]
    else:
        raise ValueError("ratio长度不对")

# pandas
def reduce_mem_usage(df, verbose=True):
    """
    传入pandas dataframe，自动判断数据需要什么类型。
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                # numpy.iinfo()函数显示整数类型的机器限制。
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print("memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("decrease by {:.1f}%".format(
            100 * (start_mem - end_mem) / start_mem))
    return df


# html
def clean_html(text):
    """
    清理html转义字符
    """
    text = html.unescape(text)
    escape_map = {
        "%21": "!",
        "%2A": "*",
        "%22": '"',
        "%27": "'",
        "%28": "(",
        "%29": ")",
        "%3B": ";",
        "%3A": ":",
        "%40": "@",
        "%26": "&",
        "%3D": "=",
        "%2B": "+",
        "%24": "$",
        "%2C": ",",
        "%2F": "/",
        "%3F": "?",
        "%25": "%",
        "%23": "#",
        "%5B": "[",
        "%5D": "]",
    }
    for k, v in escape_map.items():
        text = text.replace(k, v)
    return text


# demo
def make_args():
    """
    命令行参数解析，示例用法。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data", type=str, help="数据路径")
    parser.add_argument("--load_model", action="store_true", help="是否加载")
    parser.set_defaults(load_model=True)

    args = parser.parse_args()
    return args

# 多参数，多进程
def _fun(x, y):
    return x+y


def multi_demo():
    """
    使用functools.partial将函数fun的参数y固定，从而达到传入多个参数的目的。
    """
    import multiprocessing
    import functools
    pool = multiprocessing.Pool(4)
    mapper = functools.partial(_fun, y=2)
    inputs = [1, 2, 3, 4]
    for res in pool.imap_unordered(mapper, inputs):
        print(res)  # 打印3,4,5,6


if __name__ == "__main__":
    multi_demo()
