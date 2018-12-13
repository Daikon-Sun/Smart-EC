import json, sys, re, random, io, os
from collections import Counter
import importlib
from itertools import permutations
from ctypes import *

L, R = 35, 254
B = R - L
BS = [io.BytesIO(c_ubyte(i+L)).read() for i in range(256-L)]
OUT_FILENAME = ''
CUR_BEST = '.' * (10**7)

def write(filename, s, encoding=None):
    with open(filename, 'w', encoding=encoding) as f:
        f.write(s)

def update_success(content, encoding=None):
    global CUR_BEST
    if len(content) < len(CUR_BEST):
        print(len(content))
        write(OUT_FILENAME, content, encoding)
        CUR_BEST = content

def compress(n):
    if n == 0: return BS[0]
    rtn = b""
    while n > 0:
        rtn += BS[n % B]
        n //= B
    return rtn
def decompress(s): return sum(B ** i * c for i, c in enumerate(s))

def general(out_filename, imp, body, no_same, dct, keys_set, values_set):

    try:

        value_pos = dict()
        keys = list(dct.keys())
        values = list(dct.values())
        random.shuffle(keys)
        random.shuffle(values)
        ldct, gdct = dict(), globals()
        if no_same:
            S = keys_set & values_set
            gdct['S'] = S
        exec('keys, values = map(' + body + ', [keys, values])\n', gdct, ldct)
        keys = ldct['keys']
        values = ldct['values']
        l = len(keys)
        assert len(keys) == len(values)

        for i, v in enumerate(values):
            add_to_dict(value_pos, v, i)
        tar_pos = [-1 for i in range(l)]
        for i, key in enumerate(keys):
            tar_pos[i] = value_pos[dct[key]][-1]
            value_pos[dct[key]].pop()
        assert all(tar_po != -1 for tar_po in tar_pos)

        fac = [1]
        def get_fac(i):
            while len(fac) <= i: fac.append(fac[-1] * len(fac))
            return fac[i]

        def encode(perm):
            lp = len(perm)
            vis = [0] * (lp + 1)
            cur_mn = rtn = 0
            for j, p in enumerate(perm):
                k = vis[cur_mn:p].count(0)
                if k > 0:
                    rtn += k * get_fac(lp - j - 1)
                assert not vis[p]
                vis[p] = 1
                while vis[cur_mn]:
                    cur_mn += 1
            return rtn

        def dec0(code,j):
            r = code[0] // get_fac(j)
            code[0] -= r * get_fac(j)
            return r
        def find_nxt(r, vis, k):
            cnt = 0
            while cnt <= r:
                cnt += vis[k] == 0
                k += 1
            return k - 1
        def dec1(r, vis, cur_mn):
            k = find_nxt(r, vis, cur_mn[0])
            vis[k] = 1
            while vis[cur_mn[0]]:
                cur_mn[0] += 1
            return k
        def decode(code, ll, vis, cur_mn):
            return [dec1(r, vis, cur_mn) for r in (dec0(code,j) for j in range(ll-1, -1, -1))]

        code_mp = dict()
        def add_perm(bg, ed):
            perm = [tar_po - bg for tar_po in tar_pos[bg:ed]]
            code = compress(encode(perm))
            return code

        prv = 0
        mx = cnt = -1
        while prv < l and (prv != tar_pos[prv] or mx != cnt):
            mx = max(mx, tar_pos[prv])
            cnt += 1
            prv += 1

        data = []
        def add_data(lb, rb):
            global BS
            code = add_perm(lb, rb)
            len_code = compress(rb - lb)
            if len_code not in code_mp:
                code_mp[len_code] = len(code_mp)
            if code not in code_mp:
                code_mp[code] = len(code_mp)
            data.append(compress(code_mp[code]) + BS[-1] + compress(code_mp[len_code]))

        if prv > 0:
            add_data(0, prv)
            mx = cnt = -1

        for i in range(prv, l):
            if i > 0 and i-1 != tar_pos[i-1] and i == tar_pos[i] and cnt == mx:
                add_data(prv, i)
                prv = i
                mx = cnt = -1
            mx = max(mx, tar_pos[i] - prv)
            cnt += 1
        if l > prv:
            add_data(prv, l)

        code_table = ["" for i in range(len(code_mp))]
        for k, v in code_mp.items():
            code_table[v] = k
        code_table = BS[-2].join(code_table)
        data = BS[-2].join(data)
        if not no_same:
            D = 'reduce(lambda x,y:x+L(M(len(x).__add__,y)),L(M(lambda x:[B(r,x[1],x[2])for r in(A(x[0],j)for j in range(len(x[1])-2,-1,-1))],L(M(lambda y:([y[0]],[0]*(y[1]+1),[0]),L(M(lambda t:L(M(lambda u:D(T[D(u)]),t.split(\"' + BS[-1].decode('L1') + '\"))),r\"' + data.decode('L1') + '\".split(\"' + BS[-2].decode('L1') + '\"))))))))'
            content = '#-*-coding:L1-*-\n' + imp + '\nfrom functools import *\nM=map\nL=list\n_,I,O=sys.argv\nI,J=L(M(' + body + ',j.load(open(I))))\n' + \
                      'T=r\"' + code_table.decode('L1') + '\".split(\"' + BS[-2].decode('L1') + '\")\n' + \
                      'E=[1]\nfor i in range(1,{}):E+=[E[i-1]*{}]\n'.format(300000,B) + \
                      'def D(v):return sum(E[i]*(ord(c)-{})for i,c in enumerate(v))\nF=[1]\nfor i in range(1,{}):F+=[F[i-1]*i]\n'.format(L,len(fac)) + \
                      'def A(X,j):\n if j>={}:return 0\n r=X[0]//F[j];X[0]-=r*F[j];return r\ndef C(r,V,k,N):\n while N<=r:N+=1-V[k];k+=1;\n return k-1\ndef B(r,V,M):\n'.format(len(fac)) + \
                      ' k=C(r,V,M[0],0)\n V[k]=1\n while V[M[0]]:M[0]+=1\n return k\nj.dump(dict(zip(I,[J[i]for i in ' + D + '])),open(O,"w"))\n'
        else:
            U = 'reduce(lambda x,y:x+L(M(len(x).__add__,y)),L(M(lambda x:[B(r,x[1],x[2])for r in(A(x[0],j)for j in range(len(x[1])-2,-1,-1))],L(M(lambda y:([y[0]],[0]*(y[1]+1),[0]),L(M(lambda t:L(M(lambda u:D(T[D(u)]),t.split(\"' + BS[-1].decode('L1') + '\"))),r\"' + data.decode('L1') + '\".split(\"' + BS[-2].decode('L1') + '\"))))))))'
            content = '#-*-coding:L1-*-\n' + imp + '\nfrom functools import *\nM=map\nL=list\n_,I,O=sys.argv\nI=j.load(open(I))\nS=set(I[0])&set(I[1])\nI,J=L(M(' + body + ',I))\n' + \
                      'T=r\"' + code_table.decode('L1') + '\".split(\"' + BS[-2].decode('L1') + '\")\n' + \
                      'E=[1]\nfor i in range(1,{}):E+=[E[i-1]*{}]\n'.format(300000,B) + \
                      'def D(v):return sum(E[i]*(ord(c)-{})for i,c in enumerate(v))\nF=[1]\nfor i in range(1,{}):F+=[F[i-1]*i]\n'.format(L,len(fac)) + \
                      'def A(X,j):\n if j>={}:return 0\n r=X[0]//F[j];X[0]-=r*F[j];return r\ndef C(r,V,k,N):\n while N<=r:N+=1-V[k];k+=1;\n return k-1\ndef B(r,V,M):\n'.format(len(fac)) + \
                      ' k=C(r,V,M[0],0)\n V[k]=1\n while V[M[0]]:M[0]+=1\n return k\nS=L(S)\nj.dump(dict(zip(I+S,[J[i]for i in ' + U + ']+S)),open(O,"w"))\n'

        write(out_filename, content, 'L1')

        keys = list(dct.keys())
        values = list(dct.values())
        random.shuffle(keys)
        random.shuffle(values)
        json.dump([keys, values], open(sys.argv[1], 'w'))
        importlib.reload(module)
        cur_dct = json.load(open(sys.argv[2], 'r'))

        for cur_k, cur_v in cur_dct.items():
            if cur_k not in keys_set or cur_v not in values_set:
                print('not found fail...')
                return
        for cur_k, cur_v in cur_dct.items():
            if cur_v != dct[cur_k]:
                print('fail of general...')
                return

        print('success in general!')
        update_success(content, 'L1')

    except:
        print('exception error in general...')
        return

def get_wordType(wordType): return ['.', '[^/]'][wordType]
def get_addEnd(addEnd): return ['', '/'][addEnd]
def add_to_dict(dct, k, v, ds=0):
    if ds == 0:
        if k not in dct:
            dct[k] = []
        dct[k].append(v)
    elif ds == 1:
        if k not in dct:
            dct[k] = set()
        dct[k].add(v)

def test(out_filename, module, dct, keys_set, values_set, no_same=False, numSort=False, prefixes=None, seps=None, maxSplit=None, numPos=None, addEnd=0, wordType=0, fr_dct=None, ba_dct=None, redundants=None):

    try:
        imp = 'import json as j,sys'
        pre = '\n_,I,O=sys.argv' + ('\nD=j.load(open(I))\nS=set(D[0])&set(D[1])' if no_same else '') + '\nj.dump(dict(zip(*map('
        body = ''
        suf = ',' + ('D' if no_same else 'j.load(open(I))') + '))),open(O,"w"))'
        if not (numSort or seps):
            if no_same:
                body = 'lambda x:sorted(filter(lambda z:z not in S,x))'
            else:
                body = 'sorted'

        else:
            x = 'filter(lambda z:z not in S,x)' if no_same else 'x'
            y = 'y'
            if prefixes or redundants:
                removes = []
                if prefixes: removes += prefixes
                if redundants: removes += redundants
                if len(removes) == 1:
                    exprs = removes[0]
                else:
                    exprs = '(' + '|'.join(removes) + ')'
                y = 're.sub(\'' + exprs + '\',\'\',' + y + ')'

            if numSort and not seps:
                imp += ',re'
                body = 'lambda x:sorted(' + x + ',key=lambda y:re.sub(\'\\d+\',lambda m:m.group().zfill(9),' + y + '))'
            elif not numSort and seps and maxSplit and numPos:
                imp += ',re'
                lbody = 'lambda x:sorted(' + x + ',key=lambda y:re.sub('
                sep_expr = ''
                for i, sep in enumerate(seps):
                    if fr_dct and sep in fr_dct:
                        sep_expr += fr_dct[sep]
                    sep_expr += sep
                    if ba_dct and sep in ba_dct:
                        sep_expr += ba_dct[sep]
                    if i + 1 < len(seps):
                        sep_expr += '|'
                surr_sep_expr = '(' + sep_expr + ')' if len(seps) > 1 else sep_expr

                if maxSplit == 2:
                    exprs = '\'([^/]+)' + surr_sep_expr + '(' + get_wordType(wordType) + '+)' + get_addEnd(addEnd) + '\''
                    rbody = ',lambda m:' + 'm.group(int(y[' + str(numPos) + '])),' + y + '))'
                else:
                    exprs = '\'[^/]+' + surr_sep_expr + '[^/]+\''
                    if len(seps) == 1:
                        rbody = ',lambda m:m.group().split(\'' + sep_expr + '\')[int(y[' + str(numPos) + '])-1],' + y + '))'
                    else:
                        rbody = ',lambda m:re.split(\'' + sep_expr + '\',m.group())[int(y[' + str(numPos) + '])-1],' + y + '))'
                body = lbody + exprs + rbody

        content = imp + pre + body + ('+list(S)' if no_same else '') + suf
        # print('-' * 30)
        write(out_filename, content)

        keys = list(dct.keys())
        values = list(dct.values())
        random.shuffle(keys)
        random.shuffle(values)

        json.dump([keys, values], open(sys.argv[1], 'w'))
        importlib.reload(module)
        cur_dct = json.load(open(sys.argv[2], 'r'))

        for cur_k, cur_v in cur_dct.items():
            if cur_k not in keys_set or cur_v not in values_set:
                print('not found fail...')
                return

        for cur_k, cur_v in cur_dct.items():
            if cur_v != dct[cur_k]:
                print('fail but retry...')
                general(out_filename, imp, body, no_same, dct, keys_set, values_set)
                return

        print('success!')
        update_success(content)

    except:
        print('exception error...')
        return

def has_seps(s, seps):
    return any(sep in s for sep in seps)

def remove_prefixes(s, prefixes):
    for prefix in prefixes:
        s = re.sub('^' + prefix, '', s)
    return s

def find_prefixes(ksss, vsss):

    try:
        cnter = Counter()
        for kss, vss in zip(ksss, vsss):
            for ks, vs in zip(kss, vss):
                if (not any('$' in k for k in ks)) and (not any('$' in v for v in vs)):
                    st = set(ks) & set(vs)
                    for j, k in enumerate(ks):
                        if k in st:
                            if j > 0:
                                cnter['_'.join(ks[:j])+'_'] += 1
                            break
                    for j, v in enumerate(vs):
                        if v in st:
                            if j > 0:
                                cnter['_'.join(vs[:j])+'_'] += 1
                            break
        LEN = len(ksss)
        rtn = list(k for k, v in cnter.items() if v > 0.2 * LEN)
        print('prefixes = {}'.format(','.join(rtn)))
        return rtn
    except:
        return None


def find_redundants(dct, ksss, vsss):

    try:
        cnter = Counter()
        for kss, vss in zip(ksss, vsss):
            ks = [re.sub('[^a-zA-Z_]', '@', '_'.join(ks)) for ks in kss]
            vs = [re.sub('[^a-zA-Z_]', '@', '_'.join(vs)) for vs in vss]
            for k, v in zip(ks, vs):
                cnt = 0
                if '@' not in k and k not in vs:
                    cnter[k] += 1
                    cnt += 1
                if '@' not in v and v not in ks:
                    cnter[v] += 1
                    cnt += 1
                if cnt > 0:
                    break
        redundants = list(cnter.keys()) + [None]
        keys = dct.keys()
        values = dct.values()

        def calc_bad_pair(redundant, keys, values):
            def trans(s):
                if redundant:
                    return re.sub(redundant + '/', '', re.sub('\d+', lambda m:m.group().zfill(9), s))
                else:
                    return re.sub('\d+', lambda m:m.group().zfill(9), s)
            cur_keys = sorted(keys, key=trans)
            cur_values = sorted(values, key=trans)
            bad_pair = 0
            for key, value in zip(cur_keys, cur_values):
                if dct[key] != value:
                    bad_pair += 1
            return bad_pair

        bad_pairs = [calc_bad_pair(redundant, keys, values) for redundant in redundants]

        rtn = []
        for i in range(len(redundants) - 1):
            if bad_pairs[i] < bad_pairs[-1]:
                rtn.append(redundants[i] + '/')
        print('redundants = {}'.format(','.join(rtn)))
        return rtn
    except:
        return None


def find_seps_and_maxSplit_and_numPos(keys, values, prefixes):
    try:
        seps_dct, numPos_dct = dict(), dict()
        need_split = set()
        for key, value in zip(keys, values):
            ks = [remove_prefixes(k, prefixes) for k in key.split('/')]
            vs = [remove_prefixes(v, prefixes) for v in value.split('/')]
            for i, (k, v) in enumerate(zip(ks, vs)):
                if k != v and (k in v or v in k):
                    if k in v:
                        previous = '/'.join(vs[:i+1])
                        k, v = v, k
                        add_to_dict(numPos_dct, previous, value)
                    else:
                        previous = '/'.join(ks[:i+1])
                        add_to_dict(numPos_dct, previous, key)
                    if previous not in seps_dct:
                        seps_dct[previous] = [[], []]
                    cur_idx = k.find(v)
                    seps_dct[previous][0].append(cur_idx)
                    seps_dct[previous][1].append(cur_idx + len(v))
                    need_split.add(key)
                    break

        def add_neg(neg_fr_patts, neg_ba_patts, sep, s, j=None, split_i=None, fr_idxes=None, ba_idxes=None):
            idx = 0
            while True:
                idx = s.find(sep, idx)
                if idx == -1:
                    break
                if split_i is not None:
                    if j != split_i and idx >= 3:
                        add_to_dict(neg_fr_patts, sep, s[idx-3:idx], 1)
                    elif j == split_i and idx not in fr_idxes and idx >= 3:
                        add_to_dict(neg_fr_patts, sep, s[idx-3:idx], 1)
                else:
                    if idx >= 3:
                        add_to_dict(neg_fr_patts, sep, s[idx-3:idx], 1)
                idx += len(sep)
                if sep == '_' and s[idx-1-len(sep)] != ']':
                    continue
                if split_i is not None:
                    if j != split_i and idx + 3 <= len(s):
                        add_to_dict(neg_ba_patts, sep, s[idx:idx+3], 1)
                    elif j == split_i and idx not in ba_idxes and idx + 3 <= len(s):
                        add_to_dict(neg_ba_patts, sep, s[idx:idx+3], 1)
                else:
                    if idx + 3 <=len(s):
                        add_to_dict(neg_ba_patts, sep, s[idx:idx+3], 1)

        pos_fr_patts, pos_ba_patts, neg_fr_patts, neg_ba_patts = dict(), dict(), dict(), dict()
        maxSplit = 0
        cnter = Counter()
        for sep_dct_key, sep_dct_values in seps_dct.items():
            ks = sep_dct_key.split('/')
            idxes = sorted(zip(sep_dct_values[0], sep_dct_values[1]))
            if idxes[-1][1] != len(ks[-1]):
                continue
            sep = None
            for i in range(len(idxes) - 1):
                if idxes[i][1] >= idxes[i+1][0]:
                    sep = None
                    break
                cur_sep = ks[-1][idxes[i][1] : idxes[i+1][0]]
                if sep is None:
                    sep = cur_sep
                elif sep != cur_sep:
                    sep = None
                    break
            if sep:
                fr_idxes = [idx[1] for idx in idxes]
                ba_idxes = [idx[0] for idx in idxes]
                for j, k in enumerate(ks):
                    add_neg(neg_fr_patts, neg_ba_patts, sep, k, j, len(ks)-1, fr_idxes, ba_idxes)
                for j, (lb, rb) in enumerate(idxes):
                    s = ks[-1][lb:rb]
                    if len(s) >= 3:
                        if j > 0:
                            add_to_dict(pos_ba_patts, sep, s[:3], 1)
                        if j + 1 < len(idxes):
                            add_to_dict(pos_fr_patts, sep, s[-3:], 1)
                cnter[sep] += 1
                maxSplit = max(maxSplit, len(idxes))
        seps = list(cnter.keys())

        for sep in seps:
            for key, value in zip(keys, values):
                if key != value and key not in need_split and value not in need_split:
                    add_neg(neg_fr_patts, neg_ba_patts, sep, key, prefixes)
                    add_neg(neg_fr_patts, neg_ba_patts, sep, value, prefixes)

        def extract(seps, poss, negs, is_fr):
            rtn_dct = dict()
            for sep in seps:
                if sep not in poss or sep not in negs:
                    continue
                is_pos = True
                if len(poss[sep]) > len(negs[sep]):
                    vs = negs[sep]
                    is_pos = False
                else:
                    vs = poss[sep]
                vs = list(vs)
                min_len = min(len(v) for v in vs)
                cur_range = range(-1, -min_len-1, -1) if is_fr else range(min_len)
                for i in cur_range:
                    if any(v[i] != vs[0][i] for v in vs):
                        break
                if is_fr:
                    if i == -1:
                        continue
                    rtn = vs[0][i:]
                else:
                    if i == 0:
                        continue
                    rtn = vs[0][:i]
                if is_pos:
                    if is_fr:
                        rtn_dct[sep] = '(?<=' + rtn + ')'
                    else:
                        rtn_dct[sep] = '(?=' + rtn + ')'
                else:
                    if is_fr:
                        rtn_dct[sep] = '(?<!' + rtn + ')'
                    else:
                        rtn_dct[sep] = '(?!' + rtn + ')'
            return rtn_dct

        fr_dct = extract(seps, pos_fr_patts, neg_fr_patts, True)
        ba_dct = extract(seps, pos_ba_patts, neg_ba_patts, False)
        if '_' in seps and '_' not in fr_dct:
            fr_dct['_'] = '(?<=\\])'

        numPos = None
        for similar_keys in numPos_dct.values():

            if len(similar_keys) < 2:
                continue
            if any(len(similar_key) != len(similar_keys[0]) for similar_key in similar_keys):
                numPos = None
                break
            pos = 0
            while pos < len(similar_keys[0]):
                if not all(similar_key[pos] == similar_keys[0][pos] for similar_key in similar_keys):
                    break
                pos += 1

            cur_numPos = pos - len(similar_keys[0])
            if numPos is None:
                numPos = cur_numPos
            elif numPos != cur_numPos:
                numPos = None
                break

        print('sep = {}'.format(','.join(seps)))
        print('maxSplit = {}'.format(maxSplit))
        print('numPos = {}'.format(numPos))
        print('fr_dct = {}'.format(fr_dct))
        print('ba_dct = {}'.format(ba_dct))

        return seps, maxSplit, numPos, fr_dct, ba_dct

    except:
        return None, None, None, None, None

if __name__ == '__main__':

    module_filename = 'luluec.py'
    IN_FILENAME, OUT_FILENAME = sys.argv[1], sys.argv[2]
    with open(module_filename, 'w') as f:
        f.write('\n')
    module = importlib.import_module(module_filename[:-3])
    dct = json.load(open(IN_FILENAME, 'r'))
    keys_set = set(dct.keys())
    values_set = set(dct.values())
    keys = list(dct.keys())
    values = list(dct.values())
    ksss = list(map(lambda S: list(map(lambda s:s.split('_'), S.split('/'))), keys))
    vsss = list(map(lambda S: list(map(lambda s:s.split('_'), S.split('/'))), values))
    # print('len = {}'.format(len(dct)))

    numSort, seps, prefixes, maxSplit, numPos = False, None, None, None, None
    sys.argv[1], sys.argv[2] = 'luluinput.json', 'luluoutput.json'

    test(module_filename, module, dct, keys_set, values_set)
    test(module_filename, module, dct, keys_set, values_set, numSort=True)
    test(module_filename, module, dct, keys_set, values_set, no_same=True)
    test(module_filename, module, dct, keys_set, values_set, no_same=True, numSort=True)
    prefixes = find_prefixes(ksss, vsss)
    seps, maxSplit, numPos, fr_dct, ba_dct = find_seps_and_maxSplit_and_numPos(keys, values, prefixes)
    redundants = find_redundants(dct, ksss, vsss)
    for addEnd in [0, 1]:
        for wordType in [0, 1]:
            test(module_filename, module, dct, keys_set, values_set, numSort=numSort, prefixes=prefixes, seps=seps, maxSplit=maxSplit, numPos=numPos, addEnd=addEnd, wordType=wordType, redundants=redundants)
    if seps and fr_dct and ba_dct:
        fr_dct_keys = fr_dct.keys()
        ba_dct_keys = ba_dct.keys()
        for cur_seps in permutations(seps):
            for fr in range(1 << len(fr_dct)):
                cur_fr_dct = dict()
                for i, sep in enumerate(fr_dct_keys):
                    if (1 << i) & fr:
                        cur_fr_dct[sep] = fr_dct[sep]

                for ba in range(1 << len(ba_dct)):
                    cur_ba_dct = dict()
                    for j, sep in enumerate(ba_dct_keys):
                        if (1 << j) & ba:
                            cur_ba_dct[sep] = ba_dct[sep]
                    for no_same in [0, 1]:
                        test(module_filename, module, dct, keys_set, values_set, numSort=numSort, prefixes=prefixes, seps=cur_seps, maxSplit=maxSplit, numPos=numPos, addEnd=0, wordType=1, fr_dct=cur_fr_dct, ba_dct=cur_ba_dct, no_same=no_same, redundants=redundants)

    if os.path.exists(sys.argv[1]): os.remove(sys.argv[1])
    if os.path.exists(sys.argv[2]): os.remove(sys.argv[2])
    if os.path.exists(module_filename): os.remove(module_filename)
