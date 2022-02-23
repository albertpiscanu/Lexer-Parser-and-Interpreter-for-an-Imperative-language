from typing import List, Dict
import sys

Conf = (str, str)


class DFA:
    def __init__(self, string):
        self.alphabet: List[str] = list()
        self.initialState: str = None
        self.delta: Dict[str, Dict[str, str]] = dict()
        self.finalStates: List[str] = list()
        self.name: str = None
        self.ok: int = None
        self.state: int = None
        self.acceptari: int = None

        if string.split("\n")[0][0] == "\\" and string.split("\n")[0][1] == "n":
            self.alphabet = ['\n']
        else:
            self.alphabet = set(list(str(string.split("\n")[0])))
        self.name = str(string.split("\n")[1])
        self.initialState = str(string.split("\n")[2])
        string2 = '\n'.join(string.split('\n')[3:])

        for line in string2.splitlines():
            if line == string2.splitlines()[-1]:
                nums = [n for n in line.split()]
                for s1 in nums:
                    self.finalStates.append(s1)
            else:
                nums = [n for n in line.split(",")]
                dict2 = dict()

                if string.split("\n")[0][0] == "\\" and string.split("\n")[0][1] == "n":
                    dict2['\n'] = nums[2]
                    if nums[0] not in self.delta.keys():
                        self.delta[nums[0]] = {}
                        self.delta[nums[0]]['\n'] = nums[2]
                    else:
                        self.delta[nums[0]]['\n'] = nums[2]
                else:
                    dict2[nums[1]] = nums[2]
                    if nums[0] not in self.delta.keys():
                        self.delta[nums[0]] = {}
                        self.delta[nums[0]][nums[1]] = nums[2]
                    else:
                        self.delta[nums[0]][nums[1]] = nums[2]

    def step(self, conf) -> Conf:
        if conf[1][0] not in self.alphabet:
            return None
        if conf[0] not in self.delta.keys():
            return None
        if conf[1][0] not in self.delta[conf[0]].keys():
            if len(conf[1]) > 1:
                return conf[0], conf[1][1:]
            else:
                return conf[0], ""
        if len(conf[1]) > 1:
            return self.delta[conf[0]][conf[1][0]], conf[1][1:]
        else:
            return self.delta[conf[0]][conf[1][0]], ""

    def accept(self, word: str) -> bool:
        config = (self.initialState, word)
        k = 1
        while k == 1:
            if config[0] in self.sink_states():
                return False
            if config[1] == "":
                if config[0] in self.finalStates:
                    return True
                else:
                    return False
            elif config is None:
                return False
            config = self.step(config)
            if config is None:
                return False

    def is_sink_state(self, nod: str, drum: list) -> bool:
        drum.append(nod)
        if nod in self.finalStates:
            return False
        if nod in self.delta.keys():
            for node in self.delta[nod].values():
                if node not in drum and node in self.delta.keys():
                    ok = self.is_sink_state(node, drum)
                    if ok is False:
                        return False

    def sink_states(self) -> list:
        l = []
        for nod in self.delta.keys():
            if self.is_sink_state(nod, []) is None:
                l.append(nod)
        return l


def lexers(dfas: list, word: str, output: str):
    out = open(output, 'w')
    for i in range(0, len(dfas)):
        dfas[i].ok = 1
        dfas[i].state = dfas[i].initialState
        dfas[i].acceptari = 0
    found = 0
    m = 0
    j = 1
    token = ""
    lexeme = ""

    while m < len(word):
        for i in range(len(dfas) - 1, -1, -1):
            if word[j - 1:][0] not in dfas[i].alphabet:
                dfas[i].ok = 0
            if dfas[i].step((dfas[i].state, word[j - 1:])) is None:
                dfas[i].ok = 0
            else:
                dfas[i].state = dfas[i].step((dfas[i].state, word[j - 1:]))[0]
            if dfas[i].state in dfas[i].sink_states() or dfas[i].state not in dfas[i].delta.keys():
                dfas[i].ok = 0
            if dfas[i].accept(word[m:j]) is True:
                token = dfas[i].name
                lexeme = word[m:j]
                found = j
                dfas[i].acceptari = 1

        ok = 0
        for i in range(0, len(dfas)):
            ok += dfas[i].ok

        if ok == 0 or j + 1 > len(word):
            acc = 0
            for i in range(0, len(dfas)):
                acc += dfas[i].acceptari
            if acc == 0:
                out.close()
                out = open(output, 'w')
                if j == len(word) and ok > 0:
                    out.write("No viable alternative at character EOF, line 0")
                else:
                    out.write("No viable alternative at character "+str(j-1)+", line 0")
                out.close()
                break
            if token == "NEWLINE":
                out.write(token + " " + '\\' + 'n' + "\n")
            else:
                out.write(token + " " + lexeme + "\n")
            for i in range(0, len(dfas)):
                dfas[i].ok = 1
                dfas[i].state = dfas[i].initialState
                dfas[i].acceptari = 0
            m = found
            j = m + 1
        else:
            j = j + 1
    out.close()


def runlexer(lexer, input_file, output):
    with open(lexer, "r") as input:
        lines = input.read()
    with open(input_file, "r") as input:
        cuvant = input.read()

    lines = lines.replace("'", "")
    DFAs = []
    i = 0
    while i < len(lines):
        index = lines[i:].index("\n\n") if "\n\n" in lines[i:] else len(lines)
        index = index + i
        dfa = DFA(lines[i: index])
        DFAs.append(dfa)
        i = index + 2

    lexers(DFAs, cuvant, output)


