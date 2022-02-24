# Lexer-Parser-and-Interpreter-for-an-Imperative-language

Lexer implementation steps:

    •	reading the specification
    •	parsing the specification using a Pushdown automaton(PDA)
    •	Generating a DFA for each regex (regex -> NFA followed by NFA -> DFA conversion)
    •	Implementing the lexical analysis procedure

Parser implementation steps:

    •	writing a specification for lexical analysis of programs
    •	using the previously implemented lexer and a PDA to parse the Imperative language
    •	the output will be an Abstract Syntax Tree(AST)
	
I implemented the Interpreter using a behavioral design pattern – Visitor:
    
    •	the AST resulting from the Parser is the visitable structure
    •	The interpreter is the visitor
