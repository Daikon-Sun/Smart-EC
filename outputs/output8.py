#-*-coding:L1-*-
import json as j,sys
from functools import *
M=map
L=list
_,I,O=sys.argv
I,J=L(M(sorted,j.load(open(I))))
T=r"ºOþ#¼êVv­Çš—Ð–«¡aÞ›~pŸf¤¢7¡VªóiE›Z´fõ÷LñÑíYª\mGî<5¾ÛâV+åÊ+/{_48ç¨·ÛÇ}Þû+äóÊÈ\“¢¸‘v»Sd:º”ÞG¯H5È¯’{Â´_3…÷í”‹¾úúÇjÆŽ€zÛ_‘a%vLú<ê8PE?«®í8ù4PS„Ånju?Ã>Â¾/þ2þ'þv&þê,þ†xþ­©†&þ83þà§6œÆ+|f¢{ã+”Ù—Š@/þu,þ;öþ6]þ`<(|ÒsÔ3OÒC´Ÿ|K¢3_Ø¤6þWþ¢þUþ,þ´;þ…þ«D7R¨%þâþ#ïððbÎN¦úŠºýÂµ®Õt2ÓãMmfqëÀÆ¦«8v®²‰3+‰‡¾Cc×úõ‡™5ÎÄÍ½àz6ÉŠzã¬çNÍ›ñÑ Î”nGÙí¡ÓG·²qž£êì…YÕšÕí©Â&ûÄR:Z¶°ïÐ4 Pår­`ÞIhŠ`¨ÖÐ°Ê4gºÒJkŠG¡]íiKœþ$þµ-þ#".split("þ")
E=[1]
for i in range(1,167940):E+=[E[i-1]*219]
def D(v):return sum(E[i]*(ord(c)-35)for i,c in enumerate(v))
F=[1]
for i in range(1,187):F+=[F[i-1]*i]
def A(X,j):
 if j>=187:return 0
 r=X[0]//F[j];X[0]-=r*F[j];return r
def C(r,V,k,N):
 while N<=r:N+=1-V[k];k+=1;
 return k-1
def B(r,V,M):
 k=C(r,V,M[0],0)
 V[k]=1
 while V[M[0]]:M[0]+=1
 return k
j.dump(dict(zip(I,[J[i]for i in reduce(lambda x,y:x+L(M(len(x).__add__,y)),L(M(lambda x:[B(r,x[1],x[2])for r in(A(x[0],j)for j in range(len(x[1])-2,-1,-1))],L(M(lambda y:([y[0]],[0]*(y[1]+1),[0]),L(M(lambda t:L(M(lambda u:D(T[D(u)]),t.split("ÿ"))),r"$ÿ#þ&ÿ%þ$ÿ'þ$ÿ(þ*ÿ)þ,ÿ+þ.ÿ-þ0ÿ/þ2ÿ1þ2ÿ3þ5ÿ4þ7ÿ6þ9ÿ8þ:ÿ2þ<ÿ;".split("þ"))))))))])),open(O,"w"))
