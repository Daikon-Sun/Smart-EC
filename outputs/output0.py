import json as j,sys
_,I,O=sys.argv
j.dump(dict(zip(*map(sorted,j.load(open(I))))),open(O,"w"))