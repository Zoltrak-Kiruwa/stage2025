import torch as  th

x = th.ones(5,3)
x = x.to("cuda:0")
print(x)
y = th.ones(3,10)
z = x @ y # produit matriciel entre x et y
print(z)