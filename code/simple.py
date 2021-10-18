import numpy as np
import math
import random
import re
import copy
import hw2
from functools import cmp_to_key

def isKlass(s):
    return re.search("!", s) != None

def isGoal(s):
    return re.search("\+", s) != None or re.search("-", s) != None or isKlass(s)

def isNum(s):
    return re.search("\w", s).group().isupper()

def isWeight(s):
    if not isGoal(s):
        return 0
    elif "+" in s:
        return 1
    elif "-" in s:
        return -1
    return 0 # Not sure if this is right

def isSkip(s):
    return re.search("\?", s) != None

class Displ:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __repr__(self):
        return ("{" + ', '.join([f":{k} {v}" for k, v in self.__dict__.items() if k[0] != "_"]) + "}")

# Simple C style error codes
class Num:
    def __init__(self, col, name, val=None):
        if val:
            self._vals = np.array([val])
        else:
            self._vals = np.array([])
        self._col = col
        self._name = name
        self._mean = val if val else 0.0
        self._stdiv = 0.0
        self._pmean = 0.0
        self._pstdiv = 0.0
        self._max = 0.0
        self._min = 0.0

    def add(self, val):
        if (type(val) != type(2)) and (type(val) != type(2.1)):
            return 1
        self._vals = np.append(self._vals, val)
        self._pmean = self._mean
        self._mean = (self._mean*len(self._vals) + val)/len(self._vals)
        self._pstdiv = self._stdiv
        self._stdiv = np.std(self._vals)
        self._max = max(self._vals)
        self._min = min(self._vals)
        return 0

    def undo_add(self):
        # This stores one state in the past.
        self._vals = self._vals[:-1]
        self._mean = self._pmean
        self._stdiv = self._pstdiv
        if len(self._vals):
            self._max = max(self._vals)
            self._min = min(self._vals)
        else:
            self._max = 0.0
            self._min = 0.0
        return 0

    def dist(self, v1, v2, dist_type="aha"):
        if dist_type == "aha":
            if v1 == v2:
                return 0
            else:
                x = self.norm(v1)
                y = self.norm(v2)
                return abs(x - y)
        return 1
    
    def norm(self, x):
        if abs(self._max - self._min) < 1E-31:
            return 0
        return (x - self._min)/(self._max - self._min)

    def discretize(self, other, cohen=0.3, bins=None):
        best = 1
        rest = 0
        xys = [(good, best) for good in self._vals] + \
              [(bad, rest) for bad in other._vals]
        if bins == None:
            bins = math.sqrt(len(xys))           

        n1 = len(self._vals)
        n2 = len(other._vals)
        iota = cohen*(self._stdiv*n1 + other._stdiv*n2)/(n1 + n2)
        ranges = self.merge(self.unsuper(xys, bins, iota))

        # print(ranges)
        if len(ranges) > 1:
            for i, r in enumerate(ranges):
                best_count = 0
                rest_count = 0
                for item in r:
                    if item[1]:
                        best_count += 1
                    else:
                        rest_count += 1
                # print(r)
                yield Displ(at=self._col, name=self._name, low=r[0][0], 
                            high=r[-1][0], best=best_count, bests=n1,
                            rest=rest_count, rests=n2, first=i==0,
                            last=i==len(ranges)-1)

    def unsuper(self, dat, binsize, alsobinsize):
        dat.sort(key=lambda x: x[0])
        # print(dat)
        ret = []
        d_bin = []

        for idx, val in enumerate(dat):
            if (idx > 0) and (dat[idx-1][0] != val[0]) and (len(d_bin) >= binsize) and (len(d_bin) >= alsobinsize):
                ret.append(d_bin)
                d_bin = []
                
            d_bin.append(val)

        # print(ret)
        # print(d_bin)
        # print(binsize)
        # print(alsobinsize)

        if (len(d_bin) >= binsize) and (len(d_bin) >= alsobinsize):
            ret.append(d_bin)
        else:
            ret[-1] += d_bin
        # print("unsuper out = " + str(ret))

        return ret

    def merge(self, bins):
        # print("merge in = " + str(bins))
        ret = []
        remerge = False
        idx = 0
        while idx < len(bins) - 1:
            a = bins[idx]
            b = bins[idx+1]
            var_a = bin_variance(a)
            var_b = bin_variance(b)
            var_c = bin_variance(a + b)
            if var_c*.95 <= (var_a*len(a) + var_b*len(b))/(len(a) + len(b)):
                remerge = True
                ret.append(a+b)
                idx += 2
            else:
                ret.append(a)
                idx += 1
                if idx == len(bins) - 1:
                    ret.append(b)

        if remerge:
            return self.merge(ret)
        return ret

def bin_variance(items):
    return np.std([i[0] for i in items])


class Skip:
    def __init__(self, col, name, val=None):
        self._col = col
        self._name = name
        if val:
            self._vals = [val]
        else:
            self._vals = []

    def add(self, val):
        self._vals.append(val)
        return 0

    def undo_add(self):
        # This stores one state in the past.
        self._vals = self._vals[:-1]
        return 0

class Sym:
    def __init__(self, col, name, val=None):
        self._col = col
        self._name = name
        self._vals = []
        self._val_dict = {}
        if val:
            self._val_dict[val] = 1
            self._mode = [val]
        else:
            self._mode = []
        self._pmode = []
        self._pval = val

    def add(self, val):
        self._vals.append(val)
        self._pval = val
        self._val_dict[val] = self._val_dict.get(val, 0) + 1
        if not len(self._mode):
            self._mode.append(val)
        else:
            self._pmode = self._mode
            if self._val_dict[self._mode[0]] == self._val_dict[val]:
                self._mode.append(val)
            else:
                self._mode = [val] 
        return 0

    def undo_add(self):
        # This stores one state in the past.
        self._vals = self._vals[:-1]
        self._val_dict[self._pval] -= 1
        self._mode = self._pmode
        return 0

    def dist(self, idx1, idx2, dist_type="aha"):
        if dist_type=="aha":
            if idx1 == idx2:
                return 0
        return 1

    def discretize(self, other, cohen=None, bins=None):
        for key in set(self._val_dict.keys() | other._val_dict.keys()):
            yield Displ(at=self._col, name=self._name, low=key, high=key,
                        best=self._val_dict.get(key, 0), bests=len(self._vals),
                        rest=other._val_dict.get(key, 0),
                        rests=len(other._vals), first=False, last=False)

class Row:
    def __init__(self, dat, sample):
        self._sample = sample
        self._cells = dat
        self._ranges = [None]*len(dat)
        self._debug = False

    def __repr__(self):
        return str(self._cells)
    def __str__(self):
        return str(self._cells)
            
    def __lt__(self, row2):
        loss1 = 0
        loss2 = 0
        goalids = self._sample._ys
        # for idx in range(len(self._sample._cols)):
            # if self._sample._col_info[idx][0] == "y":
                # goalids.append(idx)
        n = len(goalids)
        for col in goalids:
            w = isWeight(self._sample._col_info[col][1])
            if self._debug:
                print(self._cells[col])
                print(row2._cells[col])
                print(w)
            a = self._sample._cols[col].norm(self._cells[col])
            b = self._sample._cols[col].norm(row2._cells[col])
            loss1 -= math.e**(w*(a - b)/n)
            loss2 -= math.e**(w*(b - a)/n)
        return loss1 < loss2

class Leaf:
    def __init__(self, name=None, c_idx=None, vals=None, low=None, high=None,
                 first=None, last=None):
        self.name = name
        self.c_idx = c_idx
        self.vals = vals
        if vals:
            if not low:
                low = min(vals)
            if not high:
                high = max(vals)
        self.low = low
        self.high = high
        self.first = first
        self.last = last

class Sample:
    def __init__(self):
        self._rows = []
        self._cols = []
        self._col_info = []
        self._ys = []
        self._xs = []

        self._p = 1
        self._enough = 1/2
        self._samples = 128
        self._far = .9
        self._support = 2

    def add(self, row):
        if len(row) != len(self._cols):
            if len(self._cols) == 0:
                for col, item in enumerate(row):
                    if isSkip(item):
                        self._col_info.append(("s", item))
                        self._cols.append(Skip(col, item))
                    else:
                        if isGoal(item):
                            self._col_info.append(("y", item))
                            self._ys.append(col)
                        else:
                            self._col_info.append(("x", item))
                            self._xs.append(col)
                        if isNum(item):
                            self._cols.append(Num(col, item))
                        else:
                            self._cols.append(Sym(col, item))
                return 0
            else:
                print("length mismatch")
                return 1
        else:
            for idx in range(len(row)):
                err = self._cols[idx].add(row[idx])
                if err:
                    print("Item " + str(row[idx]) + " in row " + str(row) + " is the wrong type. Skipping row.")
                    for i in range(idx):
                        self._cols[idx].undo_add()
                    return 1
        self._rows.append(row)
        return 0
    
    def clone(self):
        return copy.deep_copy(self)

    def shoot(self, rows=[]):
        ret = Sample()
        ret.add([i[1] for i in self._col_info])
        for row in rows:
            ret.add(row)
        return ret
    
    def read(self, filename):
        first, rest = hw2.csv(filename)
        self.add(first)
        for row in rest:
            self.add(row)
        # self._enough = math.sqrt(len(self._rows))

    def sort(self):
        # Brute force sort because it's late
        for i in range(len(self._rows)):
            for j in range(i):
                if self.zitler(self._rows[i-j], self._rows[i-j-1]):
                    self._rows[i-j], self._rows[i-j-1] = self._rows[i-j-1], self._rows[i-j]
                else:
                    break

    def zitler(self, row1, row2):
        goalids = self._ys
        # for idx in range(len(self._cols)):
            # if self._col_info[idx][0] == "y":
                # goalids.append(idx)
        s1 = 0.0
        s2 = 0.0
        e = 2.71828
        n = len(goalids)
        for idx in goalids:
            w = isWeight(self._col_info[idx][1])
            # minn = self._cols[idx]._min
        #     print(minn)
            # maxx = self._cols[idx]._max
         #    print(maxx)
            # if(minn == maxx):
                # continue
            x = self._cols[idx].norm(row1[idx])#(row1[idx] - minn)/(maxx - minn)
            y = self._cols[idx].norm(row2[idx])#(row2[idx] - minn)/(maxx - minn)
            s1 = s1 - e**(w*(x - y)/n)
            s2 = s2 - e**(w*(y - x)/n)
        return s1/n < s2/n

    def dist(self, row_idx1, row_idx2, dist_type="aha"):
        ret = 0
        num_items = 0
        for c_idx, item in enumerate(self._col_info):
            if item[0] == "s":
                continue
            num_items += 1
            ret += self._cols[c_idx].dist(self._rows[row_idx1][c_idx],
                                          self._rows[row_idx2][c_idx],
                                          dist_type)**self._p
        if num_items == 0:
            return 0
        return (ret/num_items)**(1/self._p)

    def neighbors(self, row_idx, row_list = None, dist_type="aha"):
        ret = []
        if not row_list:
            row_list = range(len(self._rows))
        for row in row_list:
            ret.append((self.dist(row_idx, row, dist_type), row))
        ret.sort(key=lambda x: x[0])
        return ret

    def faraway(self, row_idx, candidates=None):
        if not candidates:
            candidates = range(self._rows)
        n_samples = min(len(candidates), self._samples)
        sel = random.sample(candidates, n_samples)
        pile = self.neighbors(row_idx, sel)
        return pile[math.floor(len(pile)*self._far)]

    def div1(self, candidates=None, loud=False):
        if not candidates:
            candidates = range(self._rows)
        one = self.faraway(random.randint(0, len(candidates)-1), candidates)
        two = self.faraway(one[1], candidates)
        c = self.dist(one[1], two[1])
        if loud:
            print(" c=" + str(c))

        projs = []
        for row in candidates:
            a = self.dist(row, one[1])
            b = self.dist(row, two[1])
            projs.append((((a**2 + c**2 - b**2)/(2*c)), row))

        projs.sort(key=lambda x: x[0])
        mid = math.floor(len(projs)/2)
        return projs[:mid], projs[mid:]

    def divs(self, loud=False):
        leafs = []
        threshold = len(self._rows)**self._enough
        self._divs(range(len(self._rows)), 0, leafs, threshold, loud)
        return leafs

    def _divs(self, rows, level, leafs, threshold, loud=False):
        if loud:
            print("|.. "*level + " n=" + str(len(rows)), end="")
        if len(rows) < threshold:
            leaf = []
            for row in rows:
                leaf.append(Row(self._rows[row], self))
            leaf.sort()
            if loud:
                print("     goals=", end="")
                goals = []
                med = leaf[math.floor(len(leaf)/2)]
                for c_idx, info in enumerate(self._col_info):
                    if info[0] == "y":
                        goals.append(med._cells[c_idx])
                print(goals)
            leafs.append(leaf)
            return
        else:
            left, right = self.div1(rows, loud=loud)
            left_t = [i[1] for i in left]
            right_t = [i[1] for i in right]
            self._divs(left_t, level+1, leafs, threshold, loud)
            self._divs(right_t, level+1, leafs, threshold, loud)

    def leaf_clusters(self, loud=False):
        leaves = self.divs(loud)
        clusters = []
        for l_idx, leaf in enumerate(leaves):
            clusters.append((leaf[math.floor(len(leaf)/2)], l_idx))
        clusters.sort(key=lambda x: x[0])
        return leaves, clusters
    
    def value(self, rule, leaf):
        s = self._support
# this gave me subscript issues, so I'm doint something probably less efficient.
        # rules = Displ(plan=lambda b,r: b**s/(b+r) if b>r else 0,
                      # monitor=lambda b,r: r**s/(b+r) if r>b else 0,
                      # novel=lambda b,r: 1/(b+r))
        # return rules[rule](leaf.best/leaf.bests, leaf.rest/leaf.rests)
        b = leaf.best/leaf.bests
        r = leaf.rest/leaf.rests
        if rule == "plan":
            return b**s/(b+r) if b>r else 0
        elif rule == "monitor":
            return r**s/(b+r) if r>b else 0
        elif rule == "novel":
            return 1/(b+r)
        else:
            return None

    def values(self, rule, leaves):
        # print(leaves)
        # print(leaves[0])
        leaves = [(self.value(rule, leaf), leaf) for leaf in leaves]
        # return sorted([(n, leaf) for n, leaf in leaves if n > 0],key=first)
        # first doesn't work. Comparing "n"?
        return sorted([(n, leaf) for n, leaf in leaves if n > 0],
                      key=lambda x: x[0])

class Fft:
    def __init__(self, sample, branch, branches, stop=None, level=0, mul=None):
        if not mul:
            mul = 1/2
        stop = stop or 2*len(sample._rows)**mul
        midpt = math.floor(len(sample._rows)/2)
        best = sample.shoot(sample._rows[:midpt])
        rest = sample.shoot(sample._rows[midpt:])
        best.sort()
        rest.sort()
        bins = []
        for x in sample._xs:
            # xbins = []
            for bin in best._cols[x].discretize(rest._cols[x]):
                # Is this too much?
                bins.append(bin)
            # bins.append(xbins)
        bestIdea  = sample.values("plan",    bins)[-1][1]
        worstIdea = sample.values("monitor", bins)[-1][1]
        # pre = "|.. "*level
        for yes, no, idea in [(1, 0, bestIdea), (0, 1, worstIdea)]:
            leaf, tree = sample.shoot(), sample.shoot()
            for row in sample._rows:
                (leaf if match(idea, row) else tree).add(row)
            branch1 = copy.deepcopy(branch)
            med = leaf._rows[math.floor(len(leaf._rows)/2)]
            yvals = []
            for c_idx in leaf._ys:
                yvals.append(med[c_idx])
            branch1 += [Displ(at=idea.at, low=idea.low, high=idea.high,
                              type=yes, txt="if " + show(idea) + " then",
                              then=yvals, n=len(leaf._rows))]
            if len(tree._rows) <= stop:
                branch1 += [Displ(type=no, txt="  ", then=yvals,
                                  n=len(tree._rows))]
                branches += [branch1]
            else:
                Fft(tree, branch1, branches, stop=stop, level=level+1, mul=mul)


def match(leaf, row):
    v = row[leaf.at]
    if v=="?"       : return True
    elif leaf.first : return v <= leaf.high
    elif leaf.last  : return v <= leaf.low
    else            : return leaf.low <= v <= leaf.high

def show(leaf):
    if leaf.low == leaf.high : return f"{leaf.name} == {leaf.low}"
    elif leaf.first          : return f"{leaf.name} <= {leaf.high}"
    elif leaf.last           : return f"{leaf.name} >= {leaf.low}"
    else                    : return f"{leaf.low} <= {leaf.name} <= {leaf.high}"
