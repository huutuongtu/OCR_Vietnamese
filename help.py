import pandas as pd

data = pd.read_csv("data.csv")
Vocab = []
max_length = 0
for i in range(len(data)):
    if len(data['Labels'][i])==69:
        print(data['Labels'][i])


print(max_length)


# ['pad', 'W', 'ỉ', 'I', 'ở', '/', 'ò', 'E', 'p', 'K', 's', 'â', 'ô', 'd', 'Ứ', '6', 'í', 'ạ', '3', 'V', 'c', '4', "'", 'ỗ', 'h', 'Ô', 'ồ', 'l', ':', 'ổ', 'Ê', 'ặ', 'ữ', 'Đ', 'C', 'á', 'u', 'r', 'ố', 'M', 'w', 'y', 'ụ', '1', 'ứ', 'ệ', 'ị', 'ư', 'o', 'ù', 'ủ', 'O', 'e', 'T', 'Y', 'k', '#', 'A', 'ỵ', '+', 'ý', '8', 'è', 'ĩ', 'ằ', 'ơ', '(', 'ợ', 'ũ', 'G', 'à', 'S', 'L', 'ỳ', '9', 'ă', '0', 'ẽ', 'ừ', '5', 'ấ', 'ó', 'đ', 'U', 'g', 'ọ', 'ờ', 'ỏ', 'H', 'ỡ', 'ã', 'm', 'Â', 'b', 'ế', 'ẻ', 'ầ', 'ề', 'ú', 'z', '2', 'Q', 'ẩ', 'v', 't', 'J', 'ắ', 'q', 'ì', 'X', 'ộ', '7', 'ả', 'ớ', 'D', 'Ơ', 'P', 'B', 'ễ', '.', 'x', 'F', 'ê', ')', 'ử', ',', 'ỷ', 'ể', 'ẵ', 'n', 'i', 'a', 'õ', 'é', 'ậ', 'ự', ' ', 'N', '-', 'R', 'ỹ']