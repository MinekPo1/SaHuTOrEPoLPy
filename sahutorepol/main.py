from typing import Generator, Generic, Literal, TypeVar, overload
import re
from simple_warnings import catch_warnings, warn, print_warning

from sahutorepol.Types import NamespaceContext
from sahutorepol import Types, TypeHints
from sahutorepol.Errors import SaHuTOrEPoLError, SaHuTOrEPoLKeyBoardInterrupt,\
	SaHuTOrEPoLWarning, show_error_or_warning, TracebackPoint, TracebackHint


class CodePointer(object):
	def __init__(self,code:str) -> None:
		self.code = iter(code)
		self.column = 0
		self.line = 1

	@property
	def pos(self) -> tuple[int, int]:
		return self.line, self.column

	def __next__(self) -> str:
		c = next(self.code)
		if c == '\n':
			self.column = 0
			self.line += 1
		else:
			self.column += 1
		return c


class RegexBank:
	variable_name = r'[a-zA-Z_]+'
	variable = rf'{variable_name}(-{variable_name})*'
	type_name = r'[a-zA-Z]'
	i_literal = r'-?([1-9]\d*|0)'
	f_literal = r'[1-9]\d*\.(\d*[1-9]|0)'
	b_literal = r'(yes|no)'
	s_literal = r'(?P<s>[\"\'])[^\n\r]*(?P=s)'
	do = r'do'


@overload
def check_var_name(name: str) -> tuple[bool, str]:
	...


@overload
def check_var_name(name: str) -> tuple[Literal[False], None]:
	...


def check_var_name(name):
	if "-" in name:
		# split the name into its parts
		parts = [check_var_name(i) for i in name.split("-")]

		return(
			all(i[0] for i in parts),
			(parts[-1][1] if all(i[1] for i in parts) else None)
		)
	if len(name) < 3 or len(name) > 8:
		return False, None
	if not(re.fullmatch(r"[a-zA-Z]*\_[a-zA-Z]*\_[a-zA-Z]*$",name)):
		return False, None
	t = name.replace("_", "")[0]
	return len(name) < 6, t


TContext = TypeVar('TContext', TypeHints.AST.Contexts, TypeHints.AST.Root)


class Code(Generic[TContext]):
	ast: TContext

	def __init__(self,ast: TContext) -> None:
		self.ast = ast

	def resolve_expr(self,expr: TypeHints.AST.Expression) -> Types.TypeLike:
		context: NamespaceContext
		context = NamespaceContext.get_current_context()
		match expr:
			case {"type": "multi_expression", "expressions": list(e)}:
				elements = [self.resolve_expr(i) for i in e]
				try:
					obj = type(elements[0])(elements.pop(0),elements.pop())
					while elements:
						obj = type(obj)(obj,elements.pop())
				except TypeError as ex:
					raise SaHuTOrEPoLError(
						ex.args[0],
						expr['pos']
					) from ex
				return obj
			case {"type": "var_expression", "name": name}:
				try:
					return context.vars[name]
				except KeyError as ex:
					raise SaHuTOrEPoLError(
						f"Variable {name} is not defined",
						expr['pos']
					) from ex
			case {"type": "literal_expression", "value": value}:
				return context.types[value['type']](value['value'])  # type:ignore
			case {"type":"function_call", "name": name, "args": list(args)}:
				args = [self.resolve_expr(i) for i in args]
				f: Types.BoundMethodOrFunction[Types.f,Types.Type] \
					| Types.f | Types.BuiltinFunction
				try:
					f = context.vars[name]  # type:ignore
				except KeyError as ex:
					raise SaHuTOrEPoLError(
						f"Function {name} is not defined",
						expr['pos']
					) from ex
				if not isinstance(
						f,
						(Types.f,Types.BuiltinFunction,Types.BoundMethodOrFunction)
					):
					raise SaHuTOrEPoLError(
						f"{name} is not a function (is of type {type(f)})",
						expr['pos']
					)
				if isinstance(f,Types.BoundMethodOrFunction):

					if f.return_type is None:
						raise SaHuTOrEPoLError(
							f"{name} is not a function (is of type {type(f)})",
							expr['pos']
						)
				try:
					with TracebackPoint(expr['pos']):
						return f(*args)
				except ValueError as ex:
					raise SaHuTOrEPoLError(
						ex.args[0],
						expr['pos']
					) from ex
			case {"type":"constructor_call", "name": name, "args": list(args)}:
				args = [self.resolve_expr(i) for i in args]
				try:
					c = context.types[name]
				except KeyError as ex:
					raise SaHuTOrEPoLError(
						f"Type {name!r} is not defined",
						expr['pos']
					) from ex
				if issubclass(c,Types.RuntimeMethodOrFunction) and args != []:
					raise SaHuTOrEPoLError(
						f"Type {name!r} cannot be instantiated with arguments",
						expr['pos']
					)
				return c(*args)

			case _:
				raise SaHuTOrEPoLError(f"Unknown expression type {expr}", expr['pos'])

	def _run(self,ast: TypeHints.AST.Contexts) -> None:
		context = NamespaceContext.get_current_context()
		td: dict[str,TypeHints.AST.TypeDef] | None
		if (td := (ast.get('type_defs',None))) is not None:
			for k,v in td.items():
				with NamespaceContext(True) as tns:
					self._run(v)
				context.types[k] = type(
					f"RuntimeType[{k}]",(Types.Type,),tns.vars._vars
				)
		for i in ast['children']:
			with TracebackHint(i['pos']):
				try:
					match i:
						case {"type": "var_def", "name": name}:
							v,t = check_var_name(name)
							if t is None:
								raise SaHuTOrEPoLError("Invalid type name",i['pos'])
							if t not in context.types:
								raise SaHuTOrEPoLError(f"Unknown type {t}",i['pos'])
							t = context.types[t]
							if not issubclass(t,Types.Type):
								raise SaHuTOrEPoLError(f"{t} is not a type",i['pos'])
							context.vars.define(name,t())
						case {
							"type": "func_def", "name": name, "args": list(args)
						}:
							v,t = check_var_name(name)
							if t is None:
								raise SaHuTOrEPoLError("Invalid type name",i['pos'])
							if t not in "fm":
								raise SaHuTOrEPoLError(
									f"Type {t} cannot be defined in this way.",
									i['pos']
								)
							if t == "f":
								context.vars.define(name,Types.f(Code(i),args))
							else:
								context.vars.define(name,Types.m(Code(i),args))
						case {"type": "method_call", "name": name, "args": list(args)}:
							v,t = check_var_name(name)
							if t is None:
								raise SaHuTOrEPoLError("Invalid type name",i['pos'])
							f = context.vars[name]
							if not isinstance(
								f,
								(Types.f,Types.BuiltinFunction,Types.m,Types.BuiltinMethod)
							) and not(callable(f)):
								raise SaHuTOrEPoLError(f"{f!r} is not callable",i['pos'])
							args = [self.resolve_expr(i) for i in args]
							try:
								with TracebackPoint(i['pos']):
									f(*args)
							except SaHuTOrEPoLError as e:
								raise e
							except Exception as e:
								raise SaHuTOrEPoLError(f"{e}",i['pos']) from e
						case {"type": "while", "expression": condition}:
							while self.resolve_expr(condition):
								self._run(i)
						case {"type": "if", "expression": condition}:
							if self.resolve_expr(condition):
								self._run(i)
						case {"type": "var_set", "name": name, "value": value}:
							v = self.resolve_expr(value)
							_,t = check_var_name(name)
							if not(isinstance(v,context.types[t])):
								v = Types.Type.convert(v,context.types[t])
							try:
								context.vars[name] = v
							except KeyError as ex:
								raise SaHuTOrEPoLError(
									f"Undefined variable {name}",
									i['pos']
								) from ex
						case _:
							raise SaHuTOrEPoLError(f"Unknown expression type {i}",i['pos'])
				except SaHuTOrEPoLKeyBoardInterrupt as ex:
					raise ex
				except KeyboardInterrupt as ex:
					raise SaHuTOrEPoLKeyBoardInterrupt(
						"Program interrupted by user",i['pos']
						) from ex

	def run(self) -> None:
		file = self.ast.get("file",None)
		with Types.NamespaceContext(), TracebackHint(self.ast['pos'],file):
			self._run(self.ast)

	@property
	def pos(self) -> tuple[int,int]:
		return self.ast['pos']


def split_expr(expr:str, sep: str)\
		-> Generator[str,None,None]:
	if sep == "":
		raise ValueError("Empty separator")
	sep_len = len(sep)
	cur = ""
	end = ""
	brackets = 0
	in_str = False
	str_char = None
	for i in expr:
		cur += i
		if i in "\"'":
			if in_str:
				if str_char == i:
					in_str = False
					str_char = None
					end = ""
			else:
				in_str = True
				str_char = i
		if i == "(":
			brackets += 1
		elif i == ")":
			brackets -= 1
			end = ""
		if brackets == 0 and not(in_str):
			end += i
			end = end[-sep_len:]
			if end == sep:
				yield cur[:-sep_len]
				cur = ""
				end = ""
	if brackets != 0:
		raise ValueError("Unbalanced brackets")
	if in_str:
		raise ValueError("Unbalanced quotes")
	if cur != "":
		yield cur


def parse_expr(expr:str,pos: tuple[int,int]) -> TypeHints.AST.Expression:
	expr = expr.strip(" \t")
	# remove unesery brackets
	while expr.startswith("(") and expr.endswith(")"):
		expr = expr[1:-1]
	if "." in expr:
		try:
			exprs: list[str] = list(split_expr(expr,"."))
		except ValueError as ex:
			raise SaHuTOrEPoLError(f"Invalid expression {expr}",pos) from ex
		if len(exprs) != 1:
			return {
				'type': 'multi_expression',
				'expressions': [parse_expr(i,pos) for i in exprs],
				'pos': pos
			}

	if re.fullmatch(RegexBank.i_literal,expr):
		return {
			'type': 'literal_expression',
			'value': {
				'type': 'i',
				'value':int(expr),
			},
			'pos': pos
		}
	if re.fullmatch(RegexBank.f_literal,expr):
		return {
			'type': 'literal_expression',
			'value': {
				'type': 'f',
				'value':float(expr),
			},
			'pos': pos
		}
	if re.fullmatch(RegexBank.s_literal,expr):
		return {
			'type': 'literal_expression',
			'value': {
				'type': 's',
				'value':expr[1:-1].encode('raw_unicode_escape').decode('unicode_escape'),
			},
			'pos': pos
		}
	if re.fullmatch(RegexBank.b_literal,expr):
		return {
			'type': 'literal_expression',
			'value': {
				'type': 'b',
				'value':expr == "yes",
			},
			'pos': pos
		}
	if re.fullmatch(RegexBank.variable,expr):
		return {
			'type': 'var_expression',
			'name': expr,
			'pos': pos
		}
	if m:=re.fullmatch(rf"({RegexBank.type_name})\((.*)\)",expr):
		if m.group(2) is None:
			args = []
		else:
			if m.group(2).count(",") >= 2:
				raise SaHuTOrEPoLError("Too many arguments",pos)
			try:
				args = list(split_expr(m.group(2),","))
			except ValueError as ex:
				raise SaHuTOrEPoLError(ex.args[0],pos) from ex
		return {
			'type': 'constructor_call',
			'name': m.group(1),
			'args': [parse_expr(i,pos) for i in args],
			'pos': pos
		}
	if m:=re.fullmatch(rf"({RegexBank.variable})\((.*)\)",expr):
		if m.group(3) is None:
			args = []
		else:
			try:
				args = list(split_expr(m.group(3),","))
			except ValueError as ex:
				raise SaHuTOrEPoLError(ex.args[0],pos) from ex
		if m.group(3):
			return {
				'type': 'function_call',
				'name': m.group(1),
				'args': [parse_expr(i,pos) for i in args],
				'pos': pos
			}
		return {
			'type': 'function_call',
			'name': m.group(1),
			'args': [],
			'pos': pos
		}
	raise SaHuTOrEPoLError(f"Invalid expression: {expr!r}",pos)


happyness_warning_messages = {
	14:  "Parser is made slightly sad",
	10:  "Parser is made quite sad",
	5:   "Parser is made very sad",
	0:   "Parser is made extremely sad",
	-2:  "Parser disappointed but not surprised",
	-8:  "Parser is somehow surprised again",
	-15: "Parser asks for the meaning of life",
	-25: "The parser asks for you to pay for its therapy",
	-50: "<Message that makes you feel bad>",
	-75:
		"Wow do these warnings not irritate you?"
		"This is the tenth and final one!",
	-76: "Ok this is actually the final low happyness warning message"
}


def parse(code:str, file_name: str)\
	-> TypeHints.AST.Root:  # sourcery no-metrics

	with TracebackHint((1,1),file_name):
		ptr = CodePointer(code)

		tree: TypeHints.AST.Root = {
			"pos": (0,0),
			"type": "root",
			"children": [],
			"type_defs": {},
			"file": file_name
		}
		happines = 15
		symbol = ""
		cur_indent = 0
		context: list[TypeHints.AST.Contexts] = [tree]
		p_do = False
		e_do = False
		d_exp= False
		new_line = True
		brackets = 0
		bracket_stack: list[tuple[int,int]] = []
		while True:
			c = next(ptr, None)
			# region strip and manage indent
			if p_do:
				if c != "\n":
					happines -= 1
					if happines in happyness_warning_messages:
						warn(
							SaHuTOrEPoLWarning(
								happyness_warning_messages[happines],
								ptr.pos
							)
						)
				p_do = False

			if c is None:
				break
			if not(new_line) and symbol == "":
				c
				symbol = " "
			if new_line:
				if brackets != 0:
					raise SaHuTOrEPoLError("Unclosed bracket",bracket_stack[-1])
				if c == "\t" and cur_indent % 1 == 0:
					cur_indent += 0.5
				elif c == " " and cur_indent % 1 == 0.5:
					cur_indent += 0.5
				elif c in " \t":
					raise SaHuTOrEPoLError(
						"Invalid indentation."
						"An indentation level must be tabulator followed by a space.",
						ptr.pos
					)
				elif c == "\n":
					cur_indent = 0
				else:
					if cur_indent % 1 == 0.5:
						raise SaHuTOrEPoLError(
							"Indent not completed",
							ptr.pos
						)
					if cur_indent > len(context) - 1:
						raise SaHuTOrEPoLError(
							"Unexpected indent",
							ptr.pos
						)
					if cur_indent == len(context) - 2:
						e_do = True
					if cur_indent < len(context) - 2:
						raise SaHuTOrEPoLError(
							f"Unexpected dedent (expected {len(context) - 2}, got {cur_indent})",
							ptr.pos
						)
					cur_indent = 0
					symbol += c
					new_line = False
			elif c in " \t" and symbol[-1] == " ":
				pass
			elif c in " \t\n":
				symbol += " "
				if c == "\n":
					new_line = True
			else:
				symbol += c
			# endregion

			# region make sure we don't split a string literal or contents of a bracket
			if symbol.count("\"") % 2 == 1:
				if c == "\n":
					raise SaHuTOrEPoLError(
						"Unclosed string literal",
						ptr.pos
					)
				continue
			if symbol.count("'") % 2 == 1:
				if c == "\n":
					raise SaHuTOrEPoLError(
						"Unclosed string literal",
						ptr.pos
					)
				continue

			if c == "(":
				brackets += 1
				bracket_stack.append(ptr.pos)
			elif c == ")":
				if brackets == 0:
					raise SaHuTOrEPoLError(
						"Rouge closing bracket",
						ptr.pos
					)
				brackets -= 1
				bracket_stack.pop()
			if brackets != 0:
				continue
			# endregion

			# region manage do and comments
			if len(symbol) == 2 and symbol[0] == " ":
				symbol = symbol[1]

			if d_exp and symbol == "do":
				symbol = ""
				d_exp = False
				p_do = True
			if d_exp and len(symbol) == 2:
				raise SaHuTOrEPoLError(
					"Do expected",
					ptr.pos
				)

			if re.fullmatch(r" ?\$\$",symbol):
				while c != "\n" and c is not None:
					c = next(ptr, None)
				symbol = ""

			if e_do:
				if symbol == "do":
					context.pop()
					e_do = False
					p_do = True
					symbol = ""

				if len(symbol) == 2:
					raise SaHuTOrEPoLError(
						"`do` expected",
						ptr.pos
					)
				continue
			# endregion

			# match to find tokens

			if context[-1]['type'] != "type_def":
				if m:=re.fullmatch(r'if ?\((.+)\)',symbol):
					args = m.group(1)
					context[-1]['children'].append({
						'type': 'if',
						'expression': parse_expr(args,ptr.pos),
						'children': [],
						'pos': ptr.pos,
					})
					context.append(context[-1]['children'][-1])  # type:ignore
					symbol = ""

				if m:=re.fullmatch(r'while ?\((.+)\)',symbol):
					args = m.group(1)
					context[-1]['children'].append({
						'type': 'while',
						'expression': parse_expr(args,ptr.pos),
						'children': [],
						'pos': ptr.pos,
					})
					context.append(context[-1]['children'][-1])  # type:ignore
					symbol = ""

				if m:=re.fullmatch(rf"({RegexBank.variable})\$(.+) ",symbol):
					name = m.group(1)
					args = m.group(3)
					v,t = check_var_name(name)
					if t is None:
						raise SaHuTOrEPoLError(
							f"Invalid variable name {name!r}",
							ptr.pos
						)
					if not(v):
						warn(
							SaHuTOrEPoLWarning(
								f"Variable name {name!r} longer than the recommended five characters",
								ptr.pos
							)
						)
					context[-1]['children'].append({
						'type': 'var_set',
						'name': name,
						'value': parse_expr(args,ptr.pos),
						'pos': ptr.pos,
					})
					symbol = ""
					d_exp = True

				if m:=re.fullmatch(rf"({RegexBank.variable})\((.+)\)",symbol):
					name = m.group(1)
					args = m.group(3)
					v,t = check_var_name(name)
					if t is None:
						raise SaHuTOrEPoLError(
							f"Invalid variable name {name!r}",
							ptr.pos
						)
					if not(v):
						warn(
							SaHuTOrEPoLWarning(
								f"Variable name {name!r} longer than the recommended five characters",
								ptr.pos
							)
						)
					if t not in "fm":
						raise SaHuTOrEPoLError(
							f"Invalid variable type {t!r}",
							ptr.pos
						)
					args = list(split_expr(args,","))
					context[-1]['children'].append(
						{
							'type': 'method_call',
							'name': name,
							'args': [parse_expr(i,ptr.pos) for i in args],
							'pos': ptr.pos,
						}
					)
					symbol = ""
					d_exp = True

			if context[-1]['type'] == "root":
				if m:=re.fullmatch(rf"({RegexBank.type_name}) ?\$",symbol):
					name = m.group(1)
					tree['type_defs'][name] = {
						'type': 'type_def',
						'name': name,
						'pos': ptr.pos,
						'children': [],
					}
					context.append(tree['type_defs'][name])
					symbol = ""

			if m:=re.fullmatch(rf"\$({RegexBank.variable}) ",symbol):
				name = m.group(1)
				v,t = check_var_name(name)
				if t is None:
					raise SaHuTOrEPoLError(
						f"Invalid variable name {name!r}",
						ptr.pos
					)
				if not(v):
					warn(
						SaHuTOrEPoLWarning(
							f"Variable name {name!r} longer than the recommended five characters",
							ptr.pos
						)
					)
				context[-1]['children'].append({
					'type': 'var_def',
					'name': name,
					'pos': ptr.pos,
				})
				symbol = ""
				d_exp = True

			if m:=re.fullmatch(rf"\$({RegexBank.variable}) ?\((.+)\)",symbol):
				name = m.group(1)
				args = m.group(3)
				v,t = check_var_name(name)
				if t is None:
					raise SaHuTOrEPoLError(
						f"Invalid variable name {name!r}",
						ptr.pos
					)
				if not(v):
					warn(
						SaHuTOrEPoLWarning(
							f"Variable name {name!r} longer than the recommended five characters",
							ptr.pos
						)
					)
				if t not in "fn":
					raise SaHuTOrEPoLError(
						f"Invalid variable type {t!r}",
						ptr.pos
					)
				args = [i.strip() for i in args.split(",")]
				for i in args:
					av,at = check_var_name(i)
					if at is None:
						raise SaHuTOrEPoLError(
							f"Invalid variable name {i!r}",
							ptr.pos
						)
					if not(av):
						warn(
							SaHuTOrEPoLWarning(
								f"Variable name {i!r} longer than the recommended five characters",
								ptr.pos
							)
						)

				context[-1]['children'].append({
					'type': 'func_def',
					'name': name,
					'args': args,
					'children': [],
					'pos': ptr.pos,
				})
				context.append(context[-1]['children'][-1])  # type:ignore
				symbol = ""

			if symbol.count("\"") % 2 == 0 and symbol.endswith(" do"):
				raise SaHuTOrEPoLError(
					f"Unrecognized symbol {symbol!r}",
					ptr.pos
				)

		if symbol not in ["", " "]:
			raise SaHuTOrEPoLError(
				f"Unexpected end of file, {symbol!r} left hanging",
				ptr.pos
			)

		return tree


help_s = {
tuple(): """
	Usage:
	sahutorepol parse [<options>] <file> [<output>]
	sahutorepol check [<options>] <file>
	sahutorepol run [<options>] <file>
	sahutorepol help [<command>]
""",
("help",): """
	Usage:
	sahutorepol parse [<options>] <file> [<output>]
	sahutorepol run [<options>] <file>
	sahutorepol help [<command>]

	Common options:
	-s  silent, don't show warnings
	-S  strict, if any warning is found, do not continue
	-r  raise, if any error or warning is found, raise it instead of showing it
""",
("help", "parse"): """
	Usage:
	sahutorepol parse [<options>] <file> [<output>]

	Options:
	-s  silent, don't show warnings
	-S  strict, if any warning is found, do not continue
	-r  raise, if any error or warning is found, raise it instead of showing it
	-y  yaml, output the parsed tree in yaml format, else use json
""",
("help", "check"): """
	Usage:
	sahutorepol parse [<options>] <file> [<output>]

	Options:
	-s  silent, don't show warnings
	-S  strict, if any warning is found, do not continue
	-r  raise, if any error or warning is found, raise it instead of showing it
	-p  parsable, output the error and/or warnings in parsable format
	-y  yaml, output the parsable output in yaml format, else use json
""",
("help", "run"): """
	Usage:
	sahutorepol run [<options>] <file>

	Options:
	-s  silent, don't show warnings
	-S  strict, if any warning is found, do not continue
	-r  raise, if any error or warning is found, raise it instead of showing it

"""

}


def main(*args):
	import sys
	import json
	import yaml

	match args:
		case ["parse", options, file, output]:
			method = "parse"
		case ["parse", options, file]:
			method = "parse"
			output = None
		case ["parse", file]:
			method = "parse"
			options = ""
			output = None

		case ["run", options, file]:
			method = "run"
			output = None
		case ["run", file]:
			method = "run"
			options = ""
			output = None

		case ["check", options, file]:
			method = "check"
			output = None
		case ["check", file]:
			method = "check"
			options = ""
			output = None

		case _:
			if tuple(args[:2]) in help_s:
				print(help_s[tuple(args[:2])])
			else:
				print(help_s[tuple()])
			sys.exit(0)

	match method:
		case "parse":
			with open(file) as f:
				err = False
				with catch_warnings(record=True) as w:
					try:
						t = parse(f.read(),file)
					except SaHuTOrEPoLError as ex:
						if "r" in options:
							raise ex
						t = None
						show_error_or_warning(ex)
						err = True
				if "s" not in options:
					for warning in w:
						if isinstance(warning,SaHuTOrEPoLWarning) and "s" not in options:
							show_error_or_warning(warning)
						else:
							print_warning(warning)
					if "S" in options and w:
						err = True
				if err:
					sys.exit(1)
				if t is None:
					raise RuntimeError
				if "y" in options:
					dump = yaml.dump(t,default_flow_style=False)
				else:
					dump = json.dumps(t,indent=2)
				if output is None:
					print(dump)
				elif "n" not in options:
					with open(output,"w") as f:
						f.write(dump)

		case "check":
			with open(file) as f:
				data = []
				err = False
				with catch_warnings(record=True) as w:
					try:
						parse(f.read(),file)
					except SaHuTOrEPoLError as ex:
						if "r" in options:
							raise ex
						if "p" in options:
							data.append(
								{
									"type": "error",
									"message": ex.message,
									"pos": ex.pos
								}
							)
							err = True
						else:
							show_error_or_warning(ex)
							err = True
				if "s" not in options:
					for warning in w:
						if isinstance(warning,SaHuTOrEPoLWarning) and "s" not in options:
							if "p" in options:
								data.append(
									{
										"type": "warning",
										"message": warning.message,
										"pos": warning.pos,
									}
								)
							show_error_or_warning(warning)
						else:
							print_warning(warning)
					if "S" in options and w:
						err = True
				if "p" in options:
					if "y" in options:
						dump = yaml.dump(data,default_flow_style=False)
					else:
						dump = json.dumps(data,indent=2)

					print(dump)
				if err:
					sys.exit(1)

		case "run":
			with open(file) as f:
				err = False
				with catch_warnings(record=True) as w:
					try:
						t = parse(f.read(),file)
						print("parsed")
						Code(t).run()
					except (SaHuTOrEPoLError,SaHuTOrEPoLKeyBoardInterrupt) as ex:
						if "r" in options:
							raise ex
						show_error_or_warning(ex)
						t = None
						err = True
				if "s" not in options:
					for warning in w:
						if isinstance(warning,SaHuTOrEPoLWarning) and "s" not in options:
							show_error_or_warning(warning)
						else:
							print_warning(warning)
					if "S" in options and w:
						err = True
				if err:
					sys.exit(1)
				if t is None:
					raise RuntimeError


if __name__ == "__main__":
	import sys
	# print(*sys.argv[1:])
	main(*sys.argv[1:])
