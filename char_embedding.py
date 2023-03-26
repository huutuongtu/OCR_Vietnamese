import json
import torch
d = 0
max_len = 69
with open('/home/tuht/DL/vocab.json') as f:
    d = json.load(f)
def text_to_tensor(str):
    text = str
    # text = text.lower()
    # print(text)
    res = []
    for idex in text:
        res.append(d[idex])
    res = torch.tensor(res)
    # cur_len = res.shape[0]
    # if cur_len < max_len:
    # # Tính số lượng phần tử padding cần thêm vào
    #     pad_len = max_len - cur_len
    #     # print(pad_len)
    #     # Sử dụng hàm pad để thêm giá trị padding vào cuối tensor
    #     res = torch.nn.functional.pad(res, (0, pad_len), mode='constant', value=0)

    # print(res.shape)
    return res




key_list = list(d.keys())
val_list = list(d.values())

def tensor_to_text(ts):
    int_to_text = ts
    res = []
    for i in int_to_text:
        position = val_list.index(i)
        res.append(key_list[position])
    return res


# print(text_to_tensor("Số 3 Nguyễn Ngọc Vũ, Hà Nội"))
