import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np


class convLSTMnet(nn.Module):
    def __init__(self):
        super(convLSTMnet, self).__init__()
        #self.embedding = nn.Embedding(21, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(21, 64, 16, stride=1)
        self.dropout = nn.Dropout(p=0.1)
        # 输入64个，隐藏层大小为100
        self.LSTM = nn.LSTM(input_size=12, hidden_size=100,
                            num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=197*500, out_features=1)

    def forward(self, x):
        # print(x)
        x = F.one_hot(x, num_classes=21)
        x = x.float()
        x = x.permute(0, 2, 1)
        # print(x.size())
        #torch.Size([batchsize, 128, 1000])
        x = F.relu(self.conv1(x))
        #torch.Size([batchsize, 64, 985])
        x = x.permute(0, 2, 1)
        #torch.Size([batchsize, 985, 64])
        x = F.max_pool1d(x, 5, stride=5)
        #torch.Size([batchsize, 985, 12])
        x = self.dropout(x)
        # print(x.size())
        output, (h_n, c_n) = self.LSTM(x)  # 最后时间步的output
        # h_n_last=h_n[-1]
        # print(h_n_last.size())
        # print(output_last.size())
        # x=output_last.eq(h_n_last)
        x = output
        #torch.Size([72, 100])
        print(x.size())
        x = x.contiguous().view(-1, 197*500)  # 应与nn.linear中一致
        # print(x.size())
        #torch.Size([72, 100])
        x = self.fc1(x)

        x = torch.sigmoid(x)
        return (x)


# class selfAttention(nn.Module):
#     def __init__(self, num_attention_heads, input_size, hidden_size):
#         super(selfAttention, self).__init__()

#         self.num_attention_heads = num_attention_heads
#         self.attention_head_size = int(hidden_size / num_attention_heads)
#         self.all_head_size = hidden_size

#         self.key_layer = nn.Linear(input_size, hidden_size)
#         self.query_layer = nn.Linear(input_size, hidden_size)
#         self.value_layer = nn.Linear(input_size, hidden_size)

#     def trans_to_multiple_heads(self, x):
#         new_size = x.size()[: -1] + (self.num_attention_heads,
#                                      self.attention_head_size)
#         x = x.view(new_size)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, x):
#         key = self.key_layer(x)
#         query = self.query_layer(x)
#         value = self.value_layer(x)
#         #torch.Size([72, 985, 24])

#         key_heads = self.trans_to_multiple_heads(key)
#         query_heads = self.trans_to_multiple_heads(query)
#         value_heads = self.trans_to_multiple_heads(value)
#         #torch.Size([72, 8, 985, 3])

#         attention_scores = torch.matmul(
#             query_heads, key_heads.permute(0, 1, 3, 2))
#         attention_scores = attention_scores / \
#             math.sqrt(self.attention_head_size)

#         attention_probs = F.softmax(attention_scores, dim=-1)

#         context = torch.matmul(attention_probs, value_heads)
#         context = context.permute(0, 2, 1, 3).contiguous()
#         #torch.Size([72, 985, 8, 3])
#         # can only concatenate tuple (not "int") to tuple
#         new_size = context.size()[: -2] + (self.all_head_size, )
#         #torch.Size([72, 985, 24])
#         context = context.view(*new_size)
#         #torch.Size([72, 23640])
#         return context


# class convATTnet(nn.Module):
#     def __init__(self):
#         super(convATTnet, self).__init__()
#         #self.onehot = F.one_hot(num_classes=21)
#         self.conv1 = nn.Conv1d(24, 64, 16, stride=1)
#         self.conv2 = nn.Conv1d(12, 64, 4, stride=1)
#         self.dropout = nn.Dropout(p=0.1)
#         self.selfattention = selfAttention(3, 21, 24)

#         self.fc1 = nn.Linear(in_features=491*24, out_features=1)

#     def forward(self, x):
#         x = F.one_hot(x, num_classes=21)
#         x = x.float()
#         #torch.Size([72, 1000, 21])
#         #x = self.one_hot(x)
#         # x = x.permute(0, 2, 1)
#         x = self.selfattention.forward(x)
#         x = x.permute(0, 2, 1)
#         x = F.relu(self.conv1(x))
#         #torch.Size([72, 64, 985])
#         x = x.permute(0, 2, 1)
#         x = F.max_pool1d(x, 5, stride=5)
#         x = x.permute(0, 2, 1)
#         #torch.Size([72, 12, 985])
#         x = F.relu(self.conv2(x))
#         #torch.Size([72, 64, 985])
#         x = x.permute(0, 2, 1)
#         x = F.max_pool1d(x, 5, stride=5)
#         x = self.dropout(x)
#         #torch.Size([72, 982, 12])
#         x = x.contiguous().view(-1, 491*24)
#         # print(x.size())
#         x = self.fc1(x)

#         x = torch.sigmoid(x)
#         return(x)


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(21, 128, padding_idx=0)
        # 输入64个，隐藏层大小为100
        self.fc1 = nn.Linear(in_features=64*22, out_features=1)
        self.transformer = nn.Transformer(
            nhead=16, num_encoder_layers=12, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # print(x)
        x = x.view(-1, 64*22)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return (x)


class selfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads,
                                     self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)
        #torch.Size([72, 985, 24])

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)
        #torch.Size([72, 8, 985, 3])

        attention_scores = torch.matmul(
            query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        # torch.Size([1, 8,985, 985])
        attention_probs = F.softmax(attention_scores, dim=-1)
        # print(attention_probs.size())
        # attention = attention_probs.mean(
        #     axis=1, keepdim=False)  # torch.Size([1, 985, 985])
        # attention = attention.squeeze(0)  # torch.Size([985, 985])
        # attention = attention.detach().cpu().numpy()
        # # attention = np.around(attention, decimals=3)
        # np.set_printoptions(precision=3)
        # np.savetxt("attention.csv", attention, delimiter="\t")

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        #torch.Size([72, 985, 8, 3])
        # can only concatenate tuple (not "int") to tuple
        new_size = context.size()[: -2] + (self.all_head_size, )
        #torch.Size([72, 985, 24])
        context = context.view(*new_size)
        #torch.Size([72, 23640])
        return context

# 最佳模型convATTnet.4


# class convATTnet(nn.Module):
#     def __init__(self):
#         super(convATTnet, self).__init__()
#         #self.onehot = F.one_hot(num_classes=21)
#         self.conv1 = nn.Conv1d(21, 64, 16, stride=1)
#         # self.maxpool = nn.max_pool1d(5, stride=5)
#         self.dropout = nn.Dropout(p=0.1)
#         self.selfattention = selfAttention(8, 12, 24)

#         self.fc1 = nn.Linear(in_features=985*24, out_features=1)

#     def forward(self, x):
#         x = F.one_hot(x, num_classes=21)
#         x = x.float().to(device)
#         # print(x)
#         #x = self.one_hot(x)
#         x = x.permute(0, 2, 1)
#         x = F.relu(self.conv1(x))
#         x = x.permute(0, 2, 1)
#         x = F.max_pool1d(x, 5, stride=5)
#         x = self.dropout(x)
#         #torch.Size([72, 985, 12])
#         x = self.selfattention.forward(x)
#         #torch.Size([72, 985, 24])
#         x = x.view(-1, 985*24)
#         # print(x.size())
#         x = self.fc1(x)
#         x = torch.sigmoid(x)
#         return (x)

class convATTnet(nn.Module):
    def __init__(self):
        super(convATTnet, self).__init__()
        #self.onehot = F.one_hot(num_classes=21)
        self.conv1 = nn.Conv1d(21, 64, 16, stride=1)
        #self.conv2 = nn.Conv1d(12, 64, 16, stride=1)
        #self.conv3 = nn.Conv1d(12, 64, 16, stride=1)
        self.maxpool = nn.MaxPool1d(5, stride=5)
        self.dropout = nn.Dropout(p=0.1)
        self.selfattention = selfAttention(8, 12, 24)
        # self.BatchNorm1d = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(in_features=985*24, out_features=1)

    def forward(self, x):
        x = F.one_hot(x, num_classes=21)
        x = x.float()
        infea = x
        # print(infea.size())
        #x = self.one_hot(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.relu(x)
        convfea = x.permute(0, 2, 1)
        # print(x.size())
        # x = self.BatchNorm1d(x)
        x = x.permute(0, 2, 1)
        x = self.maxpool(x)
        #torch.Size([72, 985, 12])
        #torch.Size([72, 970, 12])
        # x = self.BatchNorm1d(x)
        x = self.dropout(x)
        #torch.Size([72, 985, 12])
        x = self.selfattention.forward(x)
        attfea = x
        # print(attfea.size())
        #torch.Size([72, 985, 24])
        # print(x.size())
        x = x.view(-1, 985*24)
        # print(x.size())
        x = self.fc1(x)
        finalfea = x
        # print(x.size())
        x = torch.sigmoid(x)
        return(x, infea, convfea, attfea, finalfea)


class cam_convATTnet(nn.Module):
    def __init__(self):
        super(cam_convATTnet, self).__init__()
        #self.onehot = F.one_hot(num_classes=21)
        self.conv1 = nn.Conv1d(21, 64, 16, stride=1)
        self.maxpool = nn.MaxPool1d(5, stride=5)
        self.dropout = nn.Dropout(p=0.1)
        self.selfattention = selfAttention(8, 12, 24)

        self.fc1 = nn.Linear(in_features=985*24, out_features=1)

    def forward(self, x):
        x = x.squeeze(0)
        print(x.size())
        #torch.Size([1000, 21, 21])
        # x = x.float().to(device)
        # print(x)
        #x = self.one_hot(x)
        # print(x.size())
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x = self.maxpool(x)
        x = self.dropout(x)
        #torch.Size([72, 985, 12])
        x = self.selfattention.forward(x)
        #torch.Size([72, 985, 24])
        x = x.view(-1, 985*24)
        # print(x.size())
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return (x)


class ATTconvnet(nn.Module):
    def __init__(self):
        super(ATTconvnet, self).__init__()
        #self.onehot = F.one_hot(num_classes=21)
        self.conv1 = nn.Conv1d(24, 64, 16, stride=1)
        # self.maxpool = nn.max_pool1d(5, stride=5)
        self.dropout = nn.Dropout(p=0.1)
        self.selfattention = selfAttention(8, 21, 24)

        self.fc1 = nn.Linear(in_features=985*12, out_features=1)

    def forward(self, x):
        x = F.one_hot(x, num_classes=21)
        # x = x.float().to(device)
        # print(x)
        #x = self.one_hot(x)
        x = self.selfattention.forward(x)
        #torch.Size([72, 1000, 24])
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        #torch.Size([72, 64, 985])
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, 5, stride=5)
        x = self.dropout(x)
        #torch.Size([72, 985, 12])
        #torch.Size([72, 985, 12])
        x = x.contiguous().view(-1, 985*12)
        # print(x.size())
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return (x)


class conv1d(nn.Module):
    def __init__(self):
        super(conv1d, self).__init__()
        #self.onehot = F.one_hot(num_classes=21)
        self.conv1 = nn.Conv1d(21, 64, 16, stride=1)
        # self.maxpool = nn.max_pool1d(5, stride=5)
        self.dropout = nn.Dropout(p=0.1)
        #self.selfattention = selfAttention(8, 12, 24)

        self.fc1 = nn.Linear(in_features=985*12, out_features=1)

    def forward(self, x):
        x = F.one_hot(x, num_classes=21)
        # x = x.float().to(device)
        # print(x)
        #x = self.one_hot(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, 5, stride=5)
        x = self.dropout(x)
        #torch.Size([72, 985, 12])
        #torch.Size([72, 985, 24])
        x = x.contiguous().view(-1, 985*12)
        # print(x.size())
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return (x)


class ATTnet(nn.Module):
    def __init__(self):
        super(ATTnet, self).__init__()
        #self.onehot = F.one_hot(num_classes=21)
        #self.conv1 = nn.Conv1d(24, 64, 16, stride=1)
        # self.maxpool = nn.max_pool1d(5, stride=5)
        #self.dropout = nn.Dropout(p=0.1)
        self.selfattention = selfAttention(8, 21, 24)

        self.fc1 = nn.Linear(in_features=1000*24, out_features=1)

    def forward(self, x):
        x = F.one_hot(x, num_classes=21)
        # x = x.float().to(device)
        # print(x)
        #x = self.one_hot(x)
        x = self.selfattention.forward(x)
        #torch.Size([72, 1000, 24])
        #torch.Size([72, 985, 12])
        #torch.Size([72, 985, 12])
        x = x.contiguous().view(-1, 1000*24)
        # print(x.size())
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return (x)
