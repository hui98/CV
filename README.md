# 使用WGAN-GP生成动漫角色头像

## 1.WGAN-GP原理简述
  WGAN是GAN的一种，主要解决了原始GAN由于损失函数为KL散度，当Discriminator训练的较好时，损失很小，导致梯度消失，难以训练的问题。
  
  WGAN是从Wasserstrein距离中引出的，在代码上与原始GAN区别是：
  
  1.取消了Discriminator最后的sigmoid，将分类问题转化成回归问题
 
  2.去掉BCEloss，将Discriminator的输出作为输入与真实样本的距离计算损失
  
  3.需要对损失 dloss = D(t) - D(G(z)) 添加约束，WGAN采用的是clip的方法，限制Discriminator模型的输出范围。WGAN-GP采用Lipschitz约束，要求
  ΔD/ΔX <= k，为了让生成图片的分布空间与样本空间之间的点尽量满足该约束，每次选取x ∈ conv C{x = θ*t + (1-θ)*G(z)|0=<θ=<1}计算正则项。
  
## 2.生成效果
  
  


