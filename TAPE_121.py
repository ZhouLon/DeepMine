import numpy as np
import torch
from tape import UniRepModel, TAPETokenizer
model = UniRepModel.from_pretrained('babbler-1900')
tokenizer = TAPETokenizer(vocab='unirep')  # iupac is the vocab for TAPE models, use unirep for the UniRep model

sequence = 'MAKVQFKPRATTEAIFVHCSATKPSQNVGVREIRQWHKEQGWLDVGYHFIIKRDGTVEAGRDELAVGSHAKGYNHNSIGVCLVGGIDDKGKFDANFTPAQMQSLRSLLVTLLAKYEGSVLRAHHDVAPKACPSFDLKRWWEKNELVTSDRG'
token_ids = torch.tensor([tokenizer.encode(sequence)])
output = model(token_ids)
sequence_output = output[0]
pooled_output = output[1]
lista=sequence_output.tolist()
manyist=[]
x,y=0,0
for a in lista:
    for b in a:
        x+=b[0]
        y+=b[1]
print(x/1900,y/1900)

# i=i.tolist()['avg']#转化为字典形式后，提取其中的avg键
#     i = np.reshape(i, (1, -1))
#     i-=np.min(i)
#     i/=(np.max(i)-np.min(i))
#     i = np.around(i, 4)

