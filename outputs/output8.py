#-*-coding:L1-*-
import json as j,sys
from functools import *
M=map
L=list
_,I,O=sys.argv
I,J=L(M(sorted,j.load(open(I))))
T=r"�O�#���Vv�ǚ�Ж��aޛ~�p�f��7�V��iE�Z�f��L���Y�\m�G�<5���V+��+/{_48稷��}��+����\����v�Sd:���G�H5ȯ�{´_3��픋����jƎ�z�_�a%vL�<�8PE?���8�4PS��nju?�>¾/��2�'�v&��,��x����&�83��6��+|f�{�+�ٗ�@/�u,�;��6]�`�<(|�s�3O�C��|K�3�_ؤ6�W���U�,��;����D7R�%���#����b�N������µ��t2��Mmfq��Ʀ�8v���3+���Cc�����5��ͽ�z6Ɋz��N͛�ѠΔnG���G��q�����Y՚���&���R:Z����4�P�r�`�Ih�`��а�4g��Jk�G�]�iK��$��-�#".split("�")
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
j.dump(dict(zip(I,[J[i]for i in reduce(lambda x,y:x+L(M(len(x).__add__,y)),L(M(lambda x:[B(r,x[1],x[2])for r in(A(x[0],j)for j in range(len(x[1])-2,-1,-1))],L(M(lambda y:([y[0]],[0]*(y[1]+1),[0]),L(M(lambda t:L(M(lambda u:D(T[D(u)]),t.split("�"))),r"$�#�&�%�$�'�$�(�*�)�,�+�.�-�0�/�2�1�2�3�5�4�7�6�9�8�:�2�<�;".split("�"))))))))])),open(O,"w"))
