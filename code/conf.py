CONFIG = dict(
    bins    = (float, .5, .455, 1,     "min bin size is n**bin", True),
    cohen   = (float, .35, .1, .35,    "ignore differences less than cohen*sd", True),
    depth   = (int,   5, 1, 20,     "dendogram depth. Unused", False),
    end     = (int,   4, 1, 10,      "stopping criteria. Unused", False),
    far     = (float, .9, .2, .99,    "where to find far samples", True),
    rule    = (str,   "plan", "", "", "assessment rule for a bin. Unused", False),
    loud    = (bool,   False, "", "", "loud mode: print debug on error", False),
    max     = (int,    500,  "", "",  "max samples held by 'nums'. Unused", False),
    p       = (int,    2, 1, 10,    "coefficeint on distance equation", True),
    seed    = (int,    10014, "", "", "random number seed. Unused", False),
    support = (int,    2, 2, 4,    "use x**support to score a range", True),
    todo    = (str,    "",  "", "",   "todo: function (to be run at start-up). Unused", False),
    Todo    = (str,    False,"", "",  "list available items for -t. Unused", False),
    verbose = (str,    False, "", "", "enable verbose prints. Unused", False),
    samples = (int,    128, 5, 500,  "???", True),
    forest  = (bool,   True, "", "",  "run a forest or a single random tree?", False)
    )
