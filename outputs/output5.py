import json as j,sys,re
_,I,O=sys.argv
D=j.load(open(I))
S=set(D[0])&set(D[1])
j.dump(dict(zip(*map(lambda x:sorted(filter(lambda z:z not in S,x),key=lambda y:re.sub('[^/]+(_YA_(?=h_)|(?<=\])_)[^/]+',lambda m:re.split('_YA_(?=h_)|(?<=\])_',m.group())[int(y[-1])-1],y))+list(S),D))),open(O,"w"))