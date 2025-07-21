import torch as  th
x = th.zeros(5,3)
x = x.to("cuda:0")
print(x)