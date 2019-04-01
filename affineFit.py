#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DIYer22@github
@mail: ylxx@live.com
Created on Mon Apr  1 22:52:26 2019
"""
from boxx import *

def affine_fit(from_pts, to_pts):  
    q = from_pts  
    p = to_pts  
    if len(q) != len(p) or len(q) < 1:  
        p- "原始点和目标点的个数必须相同."  
        return False  
  
    dim = len(q[0])  # 维度  
    if len(q) < dim:  
        p- "点数小于维度."  
        return False  
  
    # 新建一个空的 维度 x (维度+1) 矩阵 并填满  
    c = [[0.0 for a in range(dim)] for i in range(dim+1)]  
    for j in range(dim):  
        for k in range(dim+1):  
            for i in range(len(q)):  
                qt = list(q[i]) + [1]  
                c[k][j] += qt[k] * p[i][j]  
  
    # 新建一个空的 (维度+1) x (维度+1) 矩阵 并填满  
    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim+1)]  
    for qi in q:  
        qt = list(qi) + [1]  
        for i in range(dim+1):  
            for j in range(dim+1):  
                Q[i][j] += qt[i] * qt[j]  
  
    # 判断原始点和目标点是否共线，共线则无解. 耗时计算，如果追求效率可以不用。  
    # 其实就是解n个三元一次方程组  
    def gauss_jordan(m, eps=1.0/(10**10)):  
        (h, w) = (len(m), len(m[0]))  
        for y in range(0, h):  
            maxrow = y  
            for y2 in range(y+1, h):      
                if abs(m[y2][y]) > abs(m[maxrow][y]):  
                    maxrow = y2  
            (m[y], m[maxrow]) = (m[maxrow], m[y])  
            if abs(m[y][y]) <= eps:       
                return False  
            for y2 in range(y+1, h):      
                c = m[y2][y] / m[y][y]  
                for x in range(y, w):  
                    m[y2][x] -= m[y][x] * c  
        for y in range(h-1, 0-1, -1):    
            c = m[y][y]  
            for y2 in range(0, y):  
                for x in range(w-1, y-1, -1):  
                    m[y2][x] -= m[y][x] * m[y2][y] / c  
            m[y][y] /= c  
            for x in range(h, w):         
                m[y][x] /= c  
        return True  
  
      
    M = [Q[i] + c[i] for i in range(dim+1)]  
    if not gauss_jordan(M):  
        g.p- "错误，原始点和目标点也许是共线的."  
        return False  
  
      
    class transformation:  
        """对象化仿射变换."""  
        def To_Str(self):  
            res = ""  
            for j in range(dim):  
                str = "x%d' = " % j  
                for i in range(dim):  
                    str +="x%d * %f + " % (i, M[i][j+dim+1])  
                str += "%f" % M[dim][j+dim+1]  
                res += str + "\n"  
            return res  
        def getM(self):  
            res = ""  
            M_ = []
            for j in range(dim):  
                str = "x%d' = " % j  
                r = []
                M_.append(r)
                for i in range(dim):  
                    str +="x%d * %f + " % (i, M[i][j+dim+1])  
                    r.append(M[i][j+dim+1])
                str += "%f" % M[dim][j+dim+1]  
                r.append(M[dim][j+dim+1])
                res += str + "\n"  
            return np.float32(M_)
  
        def transform(self, pt):  
            res = [0.0 for a in range(dim)]  
            for j in range(dim):  
                for i in range(dim):  
                    res[j] += pt[i] * M[i][j+dim+1]  
                res[j] += M[dim][j+dim+1]  
            return res  
    return transformation()
  
#def test():  
if __name__ == "__main__":
    from_pt = ((38671803.6437, 2578831.9242), (38407102.8445, 2504239.2774), (38122268.3963, 2358570.38514),  
               (38126455.4595, 2346827.2602), (38177232.2601, 2398763.77833), (38423567.3485, 2571733.9203),  
               (38636876.4495, 2543442.3694), (38754169.8762, 2662401.86536), (38410773.8815, 2558886.6518),  
               (38668962.0430, 2578747.6349))  # 输入点坐标对  
    to_pt = ((38671804.6165, 2578831.1944), (38407104.0875, 2504239.1898), (38122269.2925, 2358571.57626),  
            (38126456.5675, 2346826.27022), (38177232.3973, 2398762.11714), (38423565.7744, 2571735.2278),  
            (38636873.6217, 2543440.7216), (38754168.8662, 2662401.86101), (38410774.5621, 2558886.0921),  
            (38668962.5493, 2578746.94))   # 输出点坐标对  
    from_pt = (0,0),(1,1),(3,3.1)
    to_pt = (0,1),(1,2),(3,4)
    from_pt = (-1,0),(0,1),(1,0)
    to_pt = (-1,0),(0,2),(1,0)
    trn = affine_fit(from_pt, to_pt)  
  
    if trn:  
        p- "转换公式:"  
        p- trn.To_Str()  
  
        err = 0.0  
        for i in range(len(from_pt)):  
            fp = from_pt[i]  
            tp = to_pt[i]  
            t = trn.transform(fp)  
            p- ("%s => %s ~= %s" % (fp, tuple(t), tp))  
            err += ((tp[0] - t[0])**2 + (tp[1] - t[1])**2)**0.5  
  
        p- "拟合误差 = %f" % err  
  
#if __name__ == "__main__":  
#    test()  


    pass
    
    
    
