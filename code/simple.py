import numpy as np
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
    elif re.search("\+", s) != None:
        return 1
    elif re.search("-", s) != None:
        return -1
    return 0 # Not sure if this is right

def isSkip(s):
    return re.search("\?", s) != None

# Simple C style error codes
class Num:
    def __init__(self, val=None):
        if val:
            self._vals = np.array([val])
        else:
            self._vals = np.array([])
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
                x = (v1 - self._min)/(self._max - self._min)
                y = (v2 - self._min)/(self._max - self._min)
                return abs(x - y)
        return 1
        
class Skip:
    def __init__(self, val=None):
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
    def __init__(self, val=None):
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
        if dist_type="aha":
            if idx1 == idx2:
                return 0
        return 1
            

class Sample:
    def __init__(self):
        self._rows = []
        self._cols = []
        self._col_info = []

    def add(self, row):
        if len(row) != len(self._cols):
            if len(self._cols) == 0:
                for item in row:
                    if isSkip(item):
                        self._col_info.append(("s", item))
                        self._cols.append(Skip())
                    else:
                        if isGoal(item):
                            self._col_info.append(("y", item))
                        else:
                            self._col_info.append(("x", item))
                        if isNum(item):
                            self._cols.append(Num())
                        else:
                            self._cols.append(Sym())
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
    
    def clone(self):
        return copy.deep_copy(self)
    
    def read(self, filename):
        first, rest = hw2.csv(filename)
        self.add(first)
        for row in rest:
            self.add(row)

    def sort(self):
        # Brute force sort because it's late
        for i in range(len(self._rows)):
            for j in range(i):
                if self.zitler(self._rows[i-j], self._rows[i-j-1]):
                    self._rows[i-j], self._rows[i-j-1] = self._rows[i-j-1], self._rows[i-j]
                else:
                    break


        # print(self._col_info)
        # print(self._rows[0])
        # print(self._rows[1])
        # print(self.zitler(self._rows[0], self._rows[1]))
        # print(self.zitler(self._rows[1], self._rows[2]))
        # print(self.zitler(self._rows[2], self._rows[3]))
        # print(cmp_to_key(self.zitler))
        # print(sorted(self._rows, key=cmp_to_key(self.zitler)))

    def zitler(self, row1, row2):
        goalids = []
        for idx in range(len(self._cols)):
            if self._col_info[idx][0] == "y":
                goalids.append(idx)
        s1 = 0.0
        s2 = 0.0
        e = 2.71828
        n = len(goalids)
        for idx in goalids:
            w = isWeight(self._col_info[idx][1])
            minn = self._cols[idx]._min
        #     print(minn)
            maxx = self._cols[idx]._max
         #    print(maxx)
            if(minn == maxx):
                continue
            x = (row1[idx] - minn)/(maxx - minn)
            y = (row2[idx] - minn)/(maxx - minn)
            s1 = s1 - e**(w*(x - y)/n)
          #   print(s1)
            s2 = s2 - e**(w*(y - x)/n)
           #  print(s2)
        return s1/n < s2/n

    def neighbors(self, row_idx):
        for c_idx, item in enumerate(self._rows[row_idx]):
            print(item)
            if self._col_info[c_idx][0] == "s":
                print("Skip")
            else:
                base = self._cols[c_idx]
                max_dist = -1
                min_dist = 2
                max_val = ""
                min_val = ""
                for b_idx, val in enumerate(base._vals):
                    if b_idx == c_idx:
                        continue
                    dist = base.dist(item, val)
                    if dist < min_dist:
                        min_dist = dist
                        min_val = val
                    if dist > max_dist:
                        max_dist = dist
                        max_val = val
            print("Nearest Neighbor: " + str(min_val) + " at " + str(min_dist))
            print("Furthest Neighbor: " + str(max_val) + " at " + str(max_dist))
