from abc import abstractmethod
from typing import List, Dict, Union, Any
import sys
import os
from Lexer import runlexer
TAB = '  '  # two whitespaces

store = {}
class Node:
    def __init__(self, *args):
        self.height = int(args[0])  # the level of indentation required for current Node

    def __str__(self):
        return 'prog'

    @staticmethod
    def one_tab(line):
        """Formats the line of an argument from an expression."""
        return TAB + line + '\n'

    def final_print_str(self, print_str):
        """Adds height number of tabs at the beginning of every line that makes up the current Node."""
        return (self.height * TAB).join(print_str)

    @abstractmethod
    def accept(self, visitor):
        pass


class InstructionList(Node):
    """begin <instruction_list> end"""

    def __init__(self, *args):  # args = height, [Nodes in instruction_list]
        super().__init__(args[0])
        self.list = args[1]

    def __str__(self):
        print_str = ['[\n']
        for expr in self.list:
            print_str.append(self.one_tab(expr.__str__()))
        print_str.append(']')

        return self.final_print_str(print_str)
    def executa_instructiuni(self):
        for expr in self.list:
            visitor.visit(expr)
    def accept(self, visitor):
        return visitor.visit(self)

class Expr(Node):
    """<expr> + <expr> | <expr> > <expr> | <expr> == <expr> | <variable> | <integer>"""

    def __init__(self, *args):  # args = height, '+' | '>' | '==' | 'v' | 'i', left_side, *right_side
        super().__init__(args[0])
        self.type = args[1]
        self.left = args[2]
        if len(args) > 3:
            self.right = args[3]
        else:
            # variable and integer have no right_side
            self.right = None

    def __str__(self):
        name = 'expr'
        if self.type == 'v':
            name = 'variable'
        elif self.type == 'i':
            name = 'integer'
        elif self.type == '+':
            name = 'plus'
        elif self.type == '*':
            name = 'multiply'
        elif self.type == '-':
            name = 'minus'
        elif self.type == '>':
            name = 'greaterthan'
        elif self.type == '==':
            name = 'equals'

        print_str = [name + ' [\n', self.one_tab(str(self.left))]
        if self.right:
            print_str.append(self.one_tab(str(self.right)))
        print_str.append(']')

        return self.final_print_str(print_str)

    def get_variable(self) -> str:
       return self.left

    def get_expr(self) -> Union[int, Any]:
        if self.type == 'v':
            return store[self.left]
        if self.type == 'i':
            return int(self.left)
        if self.type == '+':
            suma = self.left.get_expr() + self.right.get_expr()
            return suma
        if self.type == '-':
            diferenta = self.left.get_expr() - self.right.get_expr()
            return diferenta
        if self.type == '*':
            produs = self.left.get_expr() * self.right.get_expr()
            return produs
        if self.type == '==':
            return self.left.get_expr() == self.right.get_expr()
        if self.type == '>':
            return self.left.get_expr() > self.right.get_expr()
    def accept(self, visitor):
        return visitor.visit(self)


class While(Node):
    """while (<expr>) do <prog> od"""

    def __init__(self, *args):  # args = height, Node_expr, Node_prog
        super().__init__(args[0])
        self.expr = args[1]
        self.prog = args[2]

    def __str__(self):
        print_str = ['while [\n',
                     self.one_tab(self.expr.__str__()),
                     self.one_tab('do ' + self.prog.__str__()),
                     ']']
        return self.final_print_str(print_str)
    def executa_while(self):
        while visitor.visit(self.expr) is True:
            visitor.visit(self.prog)
    def accept(self, visitor):
        return visitor.visit(self)

class If(Node):
    """if (<expr>) then <prog> else <prog> fi"""

    def __init__(self, *args):  # args = height, Node_expr, Node_then, Node_else
        super().__init__(args[0])
        self.expr = args[1]
        self.then_branch = args[2]
        self.else_branch = args[3]

    def __str__(self):
        print_str = ['if [\n',
                     self.one_tab(self.expr.__str__()),
                     self.one_tab('then ' + self.then_branch.__str__()),
                     self.one_tab('else ' + self.else_branch.__str__()),
                     ']']
        return self.final_print_str(print_str)

    def executa_conditie(self):
        if visitor.visit(self.expr) is True:
            visitor.visit(self.then_branch)
        else:
            visitor.visit(self.else_branch)

    def accept(self, visitor):
        return visitor.visit(self)

class Assign(Node):
    """<variable> '=' <expr>"""

    def __init__(self, *args):  # args = height, Node_variable, Node_expr
        super().__init__(args[0])
        self.variable = args[1]
        self.expr = args[2]

    def __str__(self):
        print_str = ['assign [\n',
                     self.one_tab(self.variable.__str__()),
                     self.one_tab(self.expr.__str__()),
                     ']']
        return self.final_print_str(print_str)
    def update_store(self):
        var = self.variable.get_variable()
        exp = int(visitor.visit(self.expr))
        store.update({var: exp})

    def accept(self, visitor):
        return visitor.visit(self)

class Visitor():
    """Abstract Vistor Class"""
    @abstractmethod
    def visit(self, item):
        pass
class CartVisitor(Visitor):
    def visit(self, item):
        if isinstance(item, Assign):
            item.update_store()
        if isinstance(item, Expr):
            return item.get_expr()
        if isinstance(item, InstructionList):
            return item.executa_instructiuni()
        if isinstance(item, If):
            return item.executa_conditie()
        if isinstance(item, While):
            return item.executa_while()


class NFA:
    def __init__(self, expr):
        if isinstance(expr, simbol):
            self.alphabet: List[str] = expr.x
            self.initialState: tuple = (0,)
            self.finalState: tuple = (1,)
            self.delta: Dict[tuple, Dict[str, tuple]] = {self.initialState: {self.alphabet[0]: self.finalState}}
            self.states = 2
        else:
            self.alphabet: List[str] = None
            self.initialState: tuple = None
            self.finalState: tuple = None
            self.delta: Dict[tuple, Dict[str, tuple]] = None
            self.states = 0

    def renumerotare(self, nr_stari: int) -> dict:
        x = self.delta
        z = {}
        for sursa in x.keys():
            for tranzitii in x[sursa].keys():
                if len(x[sursa][tranzitii]) > 1:
                    for i in range(0, len(x[sursa][tranzitii])):
                        x[sursa][tranzitii][i] = tuple(y + nr_stari for y in x[sursa][tranzitii][i])
                else:
                    x[sursa][tranzitii] = tuple(y + nr_stari for y in x[sursa][tranzitii])
            z[tuple(y + nr_stari for y in sursa)] = x[sursa]
        return z

    def epsilon_tranzitii(self, current_state: tuple, vizitat: List[tuple]) -> tuple:
        if type(current_state) is not tuple:
            current_state = (current_state,)
        states = current_state
        if current_state not in (self.delta.keys()) or 'ε' not in self.delta[current_state].keys() \
                or current_state in vizitat:
            return tuple(set(sorted(states)))
        else:
            vizitat.append(current_state)
            for state in self.delta[current_state]['ε']:
                states += self.epsilon_tranzitii(state, vizitat)
        return tuple(set(sorted(states)))

    def step(self, dfa_state: tuple, simbol: str) -> tuple:
        next_state = tuple()
        for x in dfa_state:
            if (x,) in (self.delta.keys()) and simbol in self.delta[(x,)].keys():
                nfa_step = self.delta[(x,)][simbol]
                next_state += self.epsilon_tranzitii(nfa_step, [])
        if next_state == ():
            next_state = ('S',)
        return tuple(set(sorted(next_state)))

class expr:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.x) + ' ' + str(self.y)

class star(expr):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __str__(self):
        return 'STAR ' + str(self.x)

    def get_NFA(self) -> NFA:
        nfa = NFA(self)
        nfa1 = self.x.get_NFA()
        nfa.alphabet = nfa1.alphabet + 'ε'
        nfa.states = nfa1.states + 2
        nfa.initialState = (0,)
        nfa.finalState = (nfa.states - 1,)
        nfa.delta = {}
        nfa.delta.update({nfa.initialState: {'ε': [tuple(y + 1 for y in nfa1.initialState), nfa.finalState]}})
        nfa.delta.update(nfa1.renumerotare(1))
        nfa.delta.update({tuple(y + 1 for y in nfa1.finalState):
                              {'ε': [nfa.finalState, tuple(y + 1 for y in nfa1.initialState)]}})
        return nfa

class plus(expr):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __str__(self):
        return 'PLUS ' + str(self.x)

    def get_NFA(self) -> NFA:
        nfa = NFA(self)
        nfa1 = self.x.get_NFA()
        nfa.alphabet = nfa1.alphabet + 'ε'
        nfa.states = nfa1.states + 2
        nfa.initialState = (0,)
        nfa.finalState = (nfa.states - 1,)
        nfa.delta = {}
        nfa.delta.update({nfa.initialState: {'ε': tuple(y + 1 for y in nfa1.initialState)}})
        nfa.delta.update(nfa1.renumerotare(1))
        nfa.delta.update({tuple(y + 1 for y in nfa1.finalState):
                              {'ε': [nfa.finalState, tuple(y + 1 for y in nfa1.initialState)]}})

        return nfa

class union(expr):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __str__(self):
        return 'UNION ' + str(self.x) + ' ' + str(self.y)

    def get_NFA(self) -> NFA:
        nfa = NFA(self)
        nfa1 = self.x.get_NFA()
        nfa2 = self.y.get_NFA()
        nfa.alphabet = (nfa1.alphabet + nfa2.alphabet + 'ε')
        nfa.states = nfa1.states + nfa2.states + 2
        nfa.initialState = (0,)
        nfa.finalState = (nfa.states - 1,)
        nfa.delta = {}
        nfa.delta.update({nfa.initialState: {'ε': [tuple(y + 1 for y in nfa1.initialState),
                                                   tuple(y + nfa1.states + 1 for y in nfa2.initialState)]}})
        nfa.delta.update(nfa1.renumerotare(1))
        nfa.delta.update(nfa2.renumerotare(nfa1.states + 1))
        nfa.delta.update({tuple(y + 1 for y in nfa1.finalState): {'ε': nfa.finalState}})
        nfa.delta.update({tuple(y + nfa1.states + 1 for y in nfa2.finalState): {'ε': nfa.finalState}})
        return nfa

class concat(expr):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __str__(self):
        return 'CONCAT ' + str(self.x) + ' ' + str(self.y)

    def get_NFA(self) -> NFA:
        nfa1 = self.x.get_NFA()
        nfa2 = self.y.get_NFA()
        nfa = NFA(self)
        nfa.alphabet = (nfa1.alphabet + nfa2.alphabet + 'ε')
        nfa.initialState = nfa1.initialState
        nfa.finalState = tuple(x + nfa1.states for x in nfa2.finalState)
        nfa.delta = nfa1.delta
        nfa.delta.update({nfa1.finalState: {'ε': tuple(y + nfa1.states for y in nfa2.initialState)}})
        nfa.delta.update(nfa2.renumerotare(nfa1.states))
        nfa.states = nfa1.states + nfa2.states
        return nfa

class simbol(expr):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __str__(self):
        return str(self.x)

    def get_NFA(self) -> NFA:
        nfa = NFA(self)
        return nfa

class DFA:
    def __init__(self):
        self.alphabet: List[str] = None
        self.initialState: tuple = None
        self.finalStates: List[tuple] = []
        self.delta: Dict[tuple, Dict[str, tuple]] = {}
        self.renumerotare_stari: Dict[tuple, int] = {}
        self.stari: int = 0

    def generate_delta(self, current_state: tuple, nfa: NFA):
        if current_state not in self.renumerotare_stari.keys():
            self.renumerotare_stari.update({current_state: self.stari})
            self.stari += 1
        if nfa.finalState[0] in current_state and current_state not in self.finalStates:
            self.finalStates.append(current_state)
        for x in self.alphabet:
            next_state = nfa.step(current_state, x)
            if next_state == ('S',):
                if current_state in self.delta.keys():
                    self.delta[current_state].update({x: next_state})
                else:
                    self.delta.update({current_state: {x: next_state}})
                for y in self.alphabet:
                    if next_state in self.delta.keys():
                        self.delta[next_state].update({y: next_state})
                    else:
                        self.delta.update({next_state: {y: next_state}})
            elif current_state not in self.delta.keys() or x not in self.delta[current_state].keys() or \
                    self.delta[current_state][x] != next_state:
                if current_state in self.delta.keys():
                    self.delta[current_state].update({x: next_state})
                else:
                    self.delta.update({current_state: {x: next_state}})
                self.generate_delta(next_state, nfa)

    def NFA_to_DFA(self, nfa: NFA):
        self.alphabet = nfa.alphabet
        if 'ε' in self.alphabet:
            self.alphabet.remove('ε')
        self.initialState = nfa.epsilon_tranzitii((0,), [])
        self.generate_delta(self.initialState, nfa)

        if ('S',) in self.delta.keys():
            self.renumerotare_stari.update({('S',): self.stari})
            self.stari += 1

class DFA_simplificat:
    def __init__(self):
        self.alphabet: List[str] = None
        self.initialState: int = None
        self.finalStates: List[int] = []
        self.delta: Dict[int, Dict[str, int]] = {}
        self.stari: int = 0

    def simplifica_DFA(self, dfa: DFA):
        self.stari = dfa.stari
        self.alphabet = sorted(dfa.alphabet)
        self.initialState = dfa.renumerotare_stari[dfa.initialState]
        for x in dfa.finalStates:
            self.finalStates.append(dfa.renumerotare_stari[x])
        for sursa in dfa.delta.keys():
            for simbol in dfa.delta[sursa].keys():
                dest = dfa.renumerotare_stari[dfa.delta[sursa][simbol]]
                if dfa.renumerotare_stari[sursa] in self.delta.keys():
                    self.delta[dfa.renumerotare_stari[sursa]].update({simbol: dest})
                else:
                    self.delta.update({dfa.renumerotare_stari[sursa]: {simbol: dest}})

def afisare(dfa: DFA_simplificat, token:str, output: str):
    out = open(output, 'a')
    for x in dfa.alphabet:
        if x == '\n':
            out.write('\'' + '\\' + 'n' + '\'')
        else:
            out.write(x)
    out.write("\n")
    out.write(token + "\n")
    out.write(str(dfa.initialState) + "\n")
    for sursa in dfa.delta.keys():
        for simbol in dfa.delta[sursa].keys():
            dest = str(dfa.delta[sursa][simbol])
            if simbol == '\n':
                out.write(str(sursa) + ',' + '\'' + '\'' + '\\' + 'n' + '\'' + '\'' + ',' + dest + "\n")
            else:
                out.write(str(sursa) + ',' + '\'' + simbol + '\'' + ',' + dest + "\n")
    for x in dfa.finalStates:
        out.write(str(x) + " ")
    out.write("\n")
    out.write("\n")
    out.close()

def parsare(input: str) -> NFA:
    input = input.replace("   ","  ")
    lista = list(input.split(" "))
    if lista[len(lista) - 1] == '' and lista[len(lista) - 1] == '':
        lista.pop()
    expresii = ['CONCAT', 'UNION', 'STAR', 'PLUS']
    for i in range (0,len(lista)) :
        if lista[i] == '':
            lista[i] = ' '
    stack = []
    stack2 = []
    operanzi = 0
    operanzi_necesari = 0
    i = 0
    append = 0
    while append == 1 or i < len(lista):
        if append == 0:
            stack.append(lista[i])
            element = lista[i]
            i = i + 1
            if element not in expresii:
                stack2.append(simbol(element, None))
        else:
            element = stack[len(stack) - 1]

        if element == 'CONCAT' or element == 'UNION':
            operanzi_necesari = 2
            operanzi = 2
        elif element == 'STAR' or element == 'PLUS':
            operanzi_necesari = 1
            operanzi = 1
        elif element not in expresii:
            operanzi_necesari = operanzi_necesari - 1

        if operanzi_necesari == 0:
            if operanzi == 2:
                arg2 = stack.pop()
                arg1 = stack.pop()
                nume_expresie = stack.pop()
                expr2 = stack2.pop()
                expr1 = stack2.pop()
                if len(stack) > 0:
                    if stack[len(stack) - 1] not in expresii:
                        operanzi_necesari = 1
                        operanzi = 2
                    elif stack[len(stack) - 1] == 'CONCAT' or stack[len(stack) - 1] == 'UNION':
                        operanzi_necesari = 2
                        operanzi = 2
                    else:
                        operanzi_necesari = 1
                        operanzi = 1
                if nume_expresie == 'CONCAT':
                    stack.append(str(concat(arg1, arg2)))
                    stack2.append(concat(expr1, expr2))
                    append = 1
                elif nume_expresie == 'UNION':
                    stack.append(str(union(arg1, arg2)))
                    stack2.append(union(expr1, expr2))
                    append = 1
            else:
                arg1 = stack.pop()
                expr1 = stack2.pop()
                nume_expresie = stack.pop()
                if len(stack) > 0:
                    if stack[len(stack) - 1] not in expresii:
                        operanzi_necesari = 1
                        operanzi = 2
                    elif stack[len(stack) - 1] == 'CONCAT' or stack[len(stack) - 1] == 'UNION':
                        operanzi_necesari = 2
                        operanzi = 2
                    elif stack[len(stack) - 1] == 'STAR' or stack[len(stack) - 1] == 'PLUS':
                        operanzi_necesari = 1
                        operanzi = 1
                if nume_expresie == 'STAR':
                    stack.append(str(star(arg1, '')))
                    stack2.append(star(expr1, None))
                elif nume_expresie == 'PLUS':
                    stack.append(str(plus(arg1, '')))
                    stack2.append(plus(expr1, None))
                append = 1
        else:
            append = 0
    expr = stack2.pop()
    if str(expr) == '':
       expr = simbol(' ',None)
    nfa = expr.get_NFA()
    nfa.alphabet = set(nfa.alphabet)
    return nfa

def transf_in_prenex(regex: str) -> str:
    stack = []
    stack2 = []
    paranteze_deschise = 0
    parametrii = 2
    este_simbol = 0
    i = 0
    endline = 0
    lista = regex
    while i < len(lista):
        if lista[i] == '\'' and este_simbol == 1:
            este_simbol = 0
        elif lista[i] == '\'' and este_simbol == 0:
            este_simbol = 1
        if este_simbol == 1 and lista[i] != '\'' :
            if endline == 0 and lista[i] != "\\":
                stack.append(simbol(lista[i], None))
            if endline == 1:
                stack.append(simbol('\n', None))
                endline = 0
            if lista[i] == "\\":
                endline = 1
        if este_simbol == 0 and lista[i] != '\'':
            stack.append(lista[i])
        element = lista[i]
        if element == '(' and este_simbol == 0:
            paranteze_deschise += 1
        if element == ')' and este_simbol == 0:
            paranteze_deschise -= 1
            expr_intre_paranteze = []
            elem = stack.pop()
            while elem != '(':
                elem = stack.pop()
                if elem != '(':
                    expr_intre_paranteze.append(elem)
            if len(expr_intre_paranteze) == 1:
                stack.append(expr_intre_paranteze[0])
            if len(expr_intre_paranteze) == 2:
                stack.append(concat(expr_intre_paranteze[1], expr_intre_paranteze[0]))
            if len(expr_intre_paranteze) > 2:
                expr1 = expr_intre_paranteze[len(expr_intre_paranteze) - 1]
                j = len(expr_intre_paranteze) - 2
                while j >= 0:
                    if expr_intre_paranteze[j] == '|':
                        expr1 = union(expr1, expr_intre_paranteze[j - 1])
                        j -= 2
                    else:
                        expr1 = concat(expr1, expr_intre_paranteze[j])
                        j -= 1
                stack.append(expr1)

        if element == '+' and este_simbol == 0:
            arg1 = stack.pop()
            arg2 = stack.pop()
            stack.append(plus(arg2, None))
        if element == '*' and este_simbol == 0:
            arg1 = stack.pop()
            arg2 = stack.pop()
            stack.append(star(arg2, None))
        i += 1
    if len(stack) == 2:
        arg1 = stack.pop()
        arg2 = stack.pop()
        stack.append(concat(arg2, arg1))
    if len(stack) > 2:
        stack.reverse()
        expr = stack.pop()
        while stack:
            arg = stack.pop()
            if arg == '|':
                arg2 = stack.pop()
                expr = union(expr, arg2)
            else:
                expr = concat(expr, arg)
        stack.append(expr)
    return str(stack.pop())

def citire(fisier: str) -> list:
    with open(fisier, "r") as input:
        line = input.readlines()
    return line

def runcompletelexer(lexer: str, input:str, output:str):
    line = citire(lexer)
    tokens = []
    input_regex = []
    open("out", 'w').close()
    for i in range(0, len(line)):
        lista = line[i].split(' ', 1)
        tokens.append(lista[0])
        input_regex.append(lista[1].rstrip()[:-1])
    for i in range(0 ,len(tokens)):
        regex = transf_in_prenex(input_regex[i])
        nfa = parsare(regex)
        dfa = DFA()
        dfa.NFA_to_DFA(nfa)
        dfa_final = DFA_simplificat()
        dfa_final.simplifica_DFA(dfa)
        afisare(dfa_final, str(tokens[i]), "out")
    runlexer("out", input, output)
    open("out", 'w').close()
 
def runparser(fisier_input:str, fisier_output:str):
    open("tokens", 'w').close()
    runcompletelexer('specificatie.lex', fisier_input, 'tokens')
    with open('tokens', "r") as input:
        lines = input.readlines()
    stack = []
    indent = 0
    assigning = 0
    conditie = 0
    expresie_compusa = 0
    for x in lines:
        element = x.rstrip().split(' ', 1)
        if element[0] == 'KEYWORD':
            stack.append(element[1])
            if 'if' in element[1]:
                conditie = 1
                indent += 1
            if 'while' in element[1]:
                conditie = 1
                indent += 1
            if 'begin' in element[1]:
                indent += 1
            if 'then' in element[1] or 'do' in element[1]:
                conditie = 0
                arg_then = stack.pop()
                arg1 = stack.pop()
                arg2 = stack.pop()
                if expresie_compusa == 0:
                    arg3 = stack.pop()
                    stack.append(Expr(indent, arg2, arg3, arg1))
                else:
                    arg3 = stack.pop()
                    arg4 = stack.pop()
                    arg5 = stack.pop()
                    stack.append(Expr(indent, arg2,
                                      Expr(indent + 1, arg4, Expr(arg5.height + 1, arg5.type, str(arg5.left)),
                                           Expr(arg3.height + 1, arg3.type, str(arg3.left))), arg1))
                    expresie_compusa = 0
                stack.append(arg_then)
            if 'fi' in element[1]:
                lista_if = []
                elem = stack.pop()
                indent -= 1
                while 'if' not in str(elem):
                    lista_if.append(elem)
                    elem = stack.pop()
                if 'else' in lista_if:
                    stack.append(If(indent, lista_if[5], lista_if[3], lista_if[1]))
                else:
                    stack.append(If(indent, lista_if[3], lista_if[1]))
            if 'od' in element[1]:
                lista_do = []
                elem = stack.pop()
                indent -= 1
                while 'while' not in str(elem):
                    lista_do.append(elem)
                    elem = stack.pop()
                stack.append(While(indent, lista_do[3], lista_do[1]))
            if 'end' in element[1]:
                indent -= 1
                end = stack.pop()
                elem = stack.pop()
                lista_expresii = []
                while elem != 'begin':
                    lista_expresii.append(elem)
                    elem = stack.pop()
                lista_expresii.reverse()
                stack.append(InstructionList(indent, lista_expresii))

        if element[0] == 'COMPARATIE' or element[0] == 'OPERATIE':
            stack.append(element[1].split(' ')[1])
            if conditie == 1 and element[0] == 'OPERATIE':
                expresie_compusa = 1
        if element[0] == 'ASSIGN':
            assigning = 1
            stack.append(element[1].split(' ', 1)[0])
        if element[0] == 'INTEGER':
            stack.append(Expr(indent + 1, 'i', str(element[1])))
        if element[0] == 'VARIABLE':
            stack.append(Expr(indent + 1, 'v', str(element[1])))
        if element[0] == 'NEWLINE':
            if assigning == 1:
                arg1 = stack.pop()
                arg2 = stack.pop()
                if arg2 in ['+', '-', '*']:
                    arg3 = stack.pop()
                    arg4 = stack.pop()
                    stack.append(Assign(indent, Expr(indent + 1, 'v', arg4),
                                        Expr(indent + 1, arg2, Expr(arg3.height + 1, arg3.type, str(arg3.left)),
                                             Expr(arg1.height + 1, arg1.type, str(arg1.left)))))
                else:
                    stack.append(Assign(indent, Expr(indent + 1, 'v', arg2), arg1))
                assigning = 0



    arg = stack.pop()
    with open(fisier_output, 'w') as f:
        f.write(str(arg))
    return arg


if __name__ == "__main__":
    input_interpretor = 'tests/T3/prog/input/8.in'
    interpreter = runparser(input_interpretor, 'test')
    with open(input_interpretor, "r") as input:
        lines = input.readlines()
    for text in lines:
        print(text.rstrip())
    print()
    print(store)
    visitor = CartVisitor()
    interpreter.accept(visitor)
    print(store)
