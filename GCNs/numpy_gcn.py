import numpy as np
import pandas as pd
import scipy.sparse as sp
"""
真实场景中的图往往巨大无法读取到内存中，因此需要将矩阵压缩读取运算。
scipy.sparse包可以方便地形成、转化、运算、存储稀疏矩阵。
教程参考：
https://cmdlinetips.com/2018/03/sparse-matrices-in-python-with-scipy/
https://datascience.stackexchange.com/questions/31352/understanding-scipy-sparse-matrix-types
"""


## ============Constract the A and X matrixs:============
"""
这里的例子是一个有两种节点的网络：图书和读者，其中的关系为读者读了某本书，因此读者与读者之间并无直接关系。
"""
# 注意去重：
books = [list-of-books]
readers = [list-of-readers]
nodes = books + readers

# 创建features，也就是每个节点分配一个one-hot向量，然后拼起来
row_idx = range(len(nodes))
col_idx = range(len(nodes))
X = sp.csr_matrix(([1]*len(nodes),(row_idx,col_idx)))

# 创建邻接矩阵，注意这里应该是一个对称矩阵，reader-book和book-reader应该是相同的
# 构建一个dictionary来查询index，会快很多。用数组的index方法查询index太慢了。
relations = [list-of-book-reader-pair]
row_idx = []
col_idx = []
c_books = {book:i for i,book in enumerate(books)}
c_readers = {reader:i for i,reader in enumerate(readers)}
i = 0
for book,reader in zip(list(once_df.bib_id),list(once_df.borrower_id)):
    print(i)
    i += 1
    row_idx.append(c_books[book])
    col_idx.append(c_readers[reader]+len(books))
A_half = sp.coo_matrix(([1]*len(row_idx),(row_idx,col_idx)),shape=(len(nodes),len(nodes)))
A = A_half + A_half.T


## ============How to compute the famous A_hat in GCN:============
def cal_A_hat(A):
    dim = A.shape[0]
    A_ = A + sp.identity(A.shape[0])
    # D_diag为D矩阵的对角元素：
    D_diag  = np.squeeze(np.sum((A_), axis=1)) 
    D_inv_sqrt_diag = np.power(D_diag, -1/2)
    # 通过对角元素把对角矩阵给还原回来：
    D_inv_sqrt = sp.csr_matrix((np.array(D_inv_sqrt_diag).reshape(dim), (range(dim), range(dim)))) 
    A_hat = np.dot(np.dot(D_inv_sqrt,A_),D_inv_sqrt) 
    return A_hat

# ============ Compute node features：================
# 按照作者的说法，无监督的gcn通过随机参数得到的embedding都可以得到很好的效果
def relu(x):
    return (np.abs(x)+x)/2

units = 32
W1 = np.random.uniform(np.sqrt(6/(X.shape[1]+units)),size=(X.shape[1],units)) # (172050, 32)
M1 = np.dot(A_hat,X)
# 由于W不是稀疏矩阵，所以下面的矩阵乘法无法进行压缩，会极度消耗内存！！
# 可以考虑的方法是循环把node embedding给存下来
Z1 = np.dot(M1,W1)
H1 = relu(Z1)
# H1即为通过一层GCN得到的embedding，其中embedding的维度为32.
