import numpy as np
import re
import copy
import hw
from functools import cmp_to_key

def isKlass(s):
    return re.search("!", s) != None

def isGoal(s):
    return re.search("\+", s) != None or re.search("-", s) != None or isKlass(s)

def isNum(s):
    return re.search("\w", s).group().isupper()

def isWeight(s):
    if isGoal(s):
        return 0
    elif re.search("\+", s) != None:
        return -1
    elif re.search("-", s) != None:
        return 1
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
        self._pmax = 0.0
        self._max = 0.0
        self._pmin = 0.0
        self._min = 0.0

    def add(self, val):
        if (type(val) != type(2)) and (type(val) != type(2.1)):
            return 1
        self._vals = np.append(self._vals, val)
        self._pmean = self._mean
        self._mean = (self._mean*len(self._vals) + val)/len(self._vals)
        self._pstdiv = self._stdiv
        self._stdiv = np.std(self._vals)
        if val > self._max:
            self._pmax = self._max
            self._max = val
        if val < self._min:
            self._pmin = self._min
            self._min = val
        return 0

    def undo_add(self):
        # This stores one state in the past.
        self._vals = self._vals[:-1]
        self._mean = self._pmean
        self._stdiv = self._pstdiv
        self._max = self._pmax
        self._min = self._pmin
        return 0
        
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
        self._vals = {}
        if val:
            self._vals[val] = 1
            self._mode = [val]
        else:
            self._mode = []
        self._pmode = []
        self._pval = val

    def add(self, val):
        self._pval = val
        self._vals[val] = self._vals.get(val, 0) + 1
        if not len(self._mode):
            self._mode.append(val)
        else:
            self._pmode = self._mode
            if self._vals[self._mode[0]] == self._vals[val]:
                self._mode.append(val)
            else:
                self._mode = [val] 
        return 0

    def undo_add(self):
        # This stores one state in the past.
        self._vals[self._pval] -= 1
        self._mode = self._pmode
        return 0

class Sample:
    def __init__(self):
        self._rows = []
        self._cols = []
        self._col_info = []

    def add(self, row):
        if len(row) != len(self._cols):
            if len(self._cols) == 0:
                for item in row:
                    if isGoal(item):
                        self._col_info.append(("y", item))
                    else:
                        self._col_info.append(("x", item))
                    if isSkip(item):
                        self._cols.append(Skip())
                    else:
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
        first, rest = hw.csv(filename)
        self.add(first)
        for row in rest:
            self.add(row)

    def sort(self):
        # Brute force sort because it's late
        sorted(self._rows, key=cmp_to_key(self.zitler))

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
            maxx = self._cols[idx]._max
            if(minn == maxx):
                continue
            x = (row1[idx] - minn)/(maxx - minn)
            y = (row2[idx] - minn)/(maxx - minn)
            s1 = s1 - e**(w*(x - y)/n)
            s2 = s2 - e**(w*(y - x)/n)
        return s1/n < s2/n