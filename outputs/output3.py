import json as j,sys,re
_,I,O=sys.argv
j.dump(dict(zip(*map(lambda x:sorted(x,key=lambda y:re.sub('\d+',lambda m:m.group().zfill(9),y)),j.load(open(I))))),open(O,"w"))