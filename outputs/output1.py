import json as j,sys,re
_,I,O=sys.argv
j.dump(dict(zip(*map(lambda x:sorted(x,key=lambda y:re.sub('[^/]+_DR_[^/]+',lambda m:m.group().split('_DR_')[int(y[-10])-1],re.sub('TIU_FIGS_','',y))),j.load(open(I))))),open(O,"w"))