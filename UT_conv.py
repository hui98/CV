import torch.nn as nn
import torch
import random
# 32 -> 10
'''
(i+2p-k)/s+1 
i = 32 s = 2 p = 0 k = ? -> o = 10? :k = 23
i = 64 s = 2 p = 1 k = ? -> o = 32? :k = 2
i = 128 s = 2 p = ? k->6 -> o = 64? :k = 2/
i = 10 s=1 p=0 k= ? -> o = 1? : k = 10

i = 32 
'''
'''
conv1 = nn.Conv2d(1, 1, 14, 2, 0)

input = torch.randn(1,1,32,32)

output = conv1(input)

print(output.size())

conv2 = nn.ConvTranspose2d(1,1,4,2,1)
out = conv2(input)
print(out.size())

conv3 = nn.ConvTranspose2d(1,1,2,2,0)
out3 = conv3(out)
print(out3.size())

transconv1 = nn.ConvTranspose2d(1, 1, 23, 1, 0)

out2 = transconv1(output)

print(out2.size())
'''
'''
BCE_loss = nn.BCELoss()

s = torch.Tensor([0.5,1])
t = torch.Tensor([1,1])

loss = BCE_loss(s,t)
print(loss)

bbb = torch.randn((1,2))
print(bbb.size())
'''
def Transconv_size_test(input_size,kernel,stride,padding):
    tensor = torch.randn(input_size)
    deconv = nn.ConvTranspose2d(1,1,kernel,stride,padding)
    tmp = deconv(tensor)
    print(tmp.size())
def Conv_size_test(input_size,kernel,stride,padding):
    tensor = torch.randn(input_size)
    deconv = nn.Conv2d(1,1,kernel,stride,padding)
    tmp = deconv(tensor)
    print(tmp.size())

Conv_size_test((1,1,32,32),16,2,1)



'''
tensor = torch.zeros(1,1,5,5)
print(tensor.size())

tensor = tensor+random.randint(0,255)
print(tensor)

class PILencode(object):
	def __call__(self,tensor):
		tmp = (tensor-127.5)/127.5
		return tmp
		
class PILdecode(object):
	def __call__(self,tensor):
		tmp = (tensor*127.5)+127.5
		return tmp

e = PILencode()
d = PILdecode()
tensor = e(tensor)
print(tensor)

tensor = d(tensor)
print(tensor)
'''