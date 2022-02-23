KEYWORD (begin)|(while' ''(')|(')'' 'do)|(od)|(if' ''(')|(')'' 'then)|(end)|(else)|(fi);
ASSIGN (a|b|c|x|r|y)+' '=' ';
VARIABLE (a|b|c|r|x|y)+;
OPERATIE ' '('+'|'*'|-)' ';
INTEGER -*(0|1|2|3|4|5|6|7|8|9)+;
COMPARATIE (' '>' ')|(' '==' ');
NEWLINE '\n';
TAB ' '*;
