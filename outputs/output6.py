import json as j,sys,re
_,I,O=sys.argv
j.dump(dict(zip(*map(lambda x:sorted(x,key=lambda y:re.sub('([^/]+)_LB_(.+)',lambda m:m.group(int(y[-10])),y)),j.load(open(I))))),open(O,"w"))