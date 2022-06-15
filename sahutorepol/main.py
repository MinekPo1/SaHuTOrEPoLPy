from typing import Any, Generator, Generic, Literal, TypeVar, overload
import re

from simple_warnings import warn
import pathlib
from os import environ

from sahutorepol.Types import NamespaceContext
from sahutorepol import Types, TypeHints
from sahutorepol.Errors import SaHuTOrEPoLError, SaHuTOrEPoLKeyBoardInterrupt,\
	SaHuTOrEPoLWarning, TracebackPoint, TracebackHint

import sys

# get library path from environment variable

if "SaHPath" in environ:
	libpath = [pathlib.Path(i) for i in environ["SaHPath"].split(":")]
else:
	libpath = []

# TODO[id=STD]: implement standard library


class LibraryPath:
	__hidden_var_name = "__library_path"
	previous: "LibraryPath | None"

	def __init__(self,*args:pathlib.Path,override=False,) -> None:
		self._paths = list(args)
		self.override = override
		self.previous = None

	@property
	def paths(self) -> list[pathlib.Path]:
		if self.previous is None or self.override:
			return self._paths
		return self.previous.paths.copy() + self._paths  # ew

	def __enter__(self):
		# fetch previous library path
		r = None
		i = 1
		while r is None:
			try:
				r = sys._getframe(i).f_locals.get(self.__hidden_var_name, None)
			except ValueError:
				break
			i += 1
		# if `r is None` we are the first library path
		# use `libpath` as previous library path in that case
		self.previous = LibraryPath(*libpath) if r is None else r

		sys._getframe(1).f_locals[self.__hidden_var_name] = self
		return self

	def __exit__(self, *args):
		if self.previous is not None:
			sys._getframe(1).f_locals[self.__hidden_var_name] = self.previous
		else:
			del sys._getframe(1).f_locals[self.__hidden_var_name]

	@classmethod
	def get_current_library_path(cls) -> 'list[pathlib.Path] | None':
		r = None
		i = 1
		while r is None:
			try:
				r = sys._getframe(i).f_locals.get(cls.__hidden_var_name, None)
			except ValueError:
				break
			i += 1
		return r.paths if r is not None else None


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
	n_literal = r'-?([1-9]\d*|0),(\d*[1-9]|0)'
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

	@classmethod
	def _resolve_expr(cls,expr: TypeHints.AST.Expression) -> Generator[Any, None, Types.TypeLike]:
		context: NamespaceContext
		context = NamespaceContext.get_current_context()
		with TracebackHint(expr['pos']):
			match expr:
				case {"type": "multi_expression", "expressions": list(e)}:
					elements = []
					for i in e:
						temp = yield from cls._resolve_expr(i)
						elements.append(temp)
					if isinstance(elements[0], Types.MethodOrFunction):
						raise SaHuTOrEPoLError(
							"Cannot create a function or method inside of a multi expression.",
							expr['pos']
						)
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
				case {"type":"function_call", "name": name, "args": list(oargs)}:
					args = []
					for i in oargs:
						temp = yield from cls._resolve_expr(i)
						args.append(temp)
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
				case {"type":"constructor_call", "name": name, "args": list(oargs)}:
					args = []
					for i in oargs:
						temp = yield from cls._resolve_expr(i)
						args.append(temp)
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
					if isinstance(c.__init__,Types.RuntimeMethodOrFunction):
						temp = yield from c.__init__.iter(*args)
						return temp
					return c(*args)

				case _:
					raise SaHuTOrEPoLError(f"Unknown expression type {expr}", expr['pos'])

	@classmethod
	def resolve_expr(cls,expr: TypeHints.AST.Expression) -> Types.TypeLike:
		with TracebackHint(expr['pos']), NamespaceContext():
			gen = cls._resolve_expr(expr)
			try:
				next(gen)
			except StopIteration as ex:
				return ex.value
			else:
				raise RuntimeError("Unexpectedly did not raise StopIteration")

	@classmethod
	def _run(cls,ast: TypeHints.AST.Contexts, file: str | None = None) -> Generator:
		context = NamespaceContext.get_current_context()
		if "file" in ast:
			file = ast['file']  # type:ignore
		if file is None:
			raise SaHuTOrEPoLError("No file specified", ast['pos'])
		import_vars: dict[str,dict[str,Types.TypeLike | type[Types.Type]]] = {}
		if (imports := (ast.get('imports',None))) is not None:
			for k,v in imports.items():
				with (
						NamespaceContext(True) as import_np,
						TracebackPoint(v['pos']), TracebackHint((0,0),k)
					):
					yield from cls._run(v,file)
					import_vars[k] = import_np.vars._g_vars
					import_vars[k].update(import_np.types._types)

		td: dict[str,TypeHints.AST.TypeDefOrInclude] | None
		if (td := (ast.get('type_defs',None))) is not None:
			for k,v in td.items():
				if isinstance(v,TypeHints.AST.TypeDef):
					with NamespaceContext(True) as type_ns:
						yield from cls._run(v,file)
					context.types[k] = type(
						f"RuntimeType[{k}]",(Types.Type,),type_ns.vars._vars
					)
				else:
					pot_type = import_vars[v['source']][k]
					if isinstance(pot_type,type):
						context.types[k] = pot_type
					else:
						raise SaHuTOrEPoLError(
							"Attempted to import a variable as a type",
							v['pos']
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
						case {"type": "method_call", "name": name, "args": list(oargs)}:
							v,t = check_var_name(name)
							if t is None:
								raise SaHuTOrEPoLError("Invalid type name",i['pos'])
							f = context.vars[name]
							if not isinstance(
								f,
								(Types.f,Types.BuiltinFunction,Types.m,Types.BuiltinMethod)
							) and not(callable(f)):
								raise SaHuTOrEPoLError(f"{f!r} is not callable",i['pos'])
							args = []
							for i in oargs:
								temp = yield from cls._resolve_expr(i)
								args.append(temp)
							try:
								with TracebackPoint(i['pos']):
									if hasattr(f,"iter"):
										yield from f.iter(args)  # type:ignore
									else:
										f(*args)
							except SaHuTOrEPoLError as e:
								raise e
							except Exception as e:
								raise SaHuTOrEPoLError(f"{e}",i['pos']) from e
						case {"type": "while", "expression": condition}:
							v = yield from cls._resolve_expr(condition)
							while Types.b(v).value:
								yield from cls._run(i,file)
								v = yield from cls._resolve_expr(condition)
						case {"type": "if", "expression": condition}:
							v = yield from cls._resolve_expr(condition)
							if Types.b(v).value:
								yield from cls._run(i,file)
						case {"type": "var_set", "name": name, "value": value}:
							v = yield from cls._resolve_expr(value)
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
						case {"type": "var_inc", "name": name, "source": source}:
							val = import_vars[source][name]
							# this is not really needed since it
							if isinstance(val,type):
								raise SaHuTOrEPoLError(
									"Attempted to import a type as a variable",
									i['pos']
								)
							context.define(name,val)  # type:ignore

						case _:
							raise SaHuTOrEPoLError(f"Unknown expression type {i}",i['pos'])
				except SaHuTOrEPoLKeyBoardInterrupt as ex:
					raise ex
				except KeyboardInterrupt as ex:
					raise SaHuTOrEPoLKeyBoardInterrupt(
						"Program interrupted by user",i['pos']
						) from ex
				# yield the current position and file
				yield i['pos'],file

	def run(self) -> None:
		file = self.ast.get("file",None)
		if not(isinstance(file,str)):
			raise ValueError("No file specified, corrupted AST?")
		with Types.NamespaceContext(), TracebackHint(self.ast['pos'],file):
			for _ in self._run(self.ast):
				pass

	def __iter__(self) -> Generator:
		with Types.NamespaceContext(), TracebackHint(self.ast['pos'],self.ast['file']):
			yield from self._run(self.ast)

	@property
	def pos(self) -> tuple[int,int]:
		return self.ast['pos']


def split_expr(expr:str, sep: str)\
		-> Generator[str,None,None]:
	if not(sep):
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
	if re.fullmatch(RegexBank.n_literal,expr):
		return {
			'type': 'literal_expression',
			'value': {
				'type': 'n',
				'value':float(expr.replace(",",".")),
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
	if m:=re.fullmatch(rf"({RegexBank.variable})\((.*)\)",expr):
		if m.group(3) is None:
			args = []
		else:
			try:
				args = list(split_expr(m.group(3),"|"))
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


cached_parsed_files: dict[str,TypeHints.AST.Root] = {}
member_cache: dict[str,dict[str,TypeHints.AST.Members]] = {}


def get_file_name(lib_name: str) -> str:
	raise NotImplementedError(
		"Library localisation has not yet been implemented."
	)
	# TODO[id=lib_loc]: implement library localisation


def check_for_member(name:str, lib_name: str) -> Literal["Type", "Var"] | None:
	file_name = get_file_name(lib_name)
	if file_name not in member_cache:
		member_cache[file_name] = {}
		if file_name not in cached_parsed_files:
			tree = parse(open(file_name).read(),file_name)
		elif cached_parsed_files[file_name] is None:
			raise SaHuTOrEPoLError(
				f"File {file_name} has been attempted to be parsed multiple times, "
				"most likely due to a circular import",
				(0,0)
			)
		else:
			tree = cached_parsed_files[file_name]
		for k,i in tree['type_defs'].items():
			member_cache[file_name][k] = i
		for i in tree['children']:
			if i['type'] == 'func_def':
				member_cache[file_name][i['name']] = i
	member = member_cache[file_name].get(name,None)
	if member is None:
		return None
	return "Type" if member['type'] in ['type_def','type_include'] else "Var"


def parse(code:str, file_name: str, do_resolve: bool = True)\
	-> TypeHints.AST.Root:  # sourcery no-metrics

	if do_resolve:
		file_name = str(pathlib.Path(file_name).resolve())

	if file_name in cached_parsed_files:
		if cached_parsed_files[file_name] is None:
			raise SaHuTOrEPoLError(
				f"File {file_name} has been attempted to be parsed multiple times, "
				"most likely due to a circular import",
				(0,0)
			)
		return cached_parsed_files[file_name]

	with TracebackHint((1,1),file_name):
		ptr = CodePointer(code)

		tree: TypeHints.AST.Root = {
			"pos": (0,0),
			"type": "root",
			"children": [],
			"imports": {},
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
			# region: strip and manage indent
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

			# region: make sure we don't split a string literal or contents of a bracket
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

			# region: manage do and comments
			# check for shellbangs
			if ptr.line == 1 and symbol == "#!":
				while next(ptr) != "\n":
					pass
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
					args = list(split_expr(args,"|"))
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
					type_def: TypeHints.AST.TypeDef = {
						'type': 'type_def',
						'name': name,
						'pos': ptr.pos,
						'children': [],
					}
					tree['type_defs'][name] = type_def
					context.append(type_def)
					symbol = ""

				if m:=re.fullmatch(
					rf"\$({RegexBank.type_name}) ?\$([a-zA-Z0-9\-\.]+)",symbol
				):
					name = m.group(1)
					lname= m.group(2)
					if check_for_member(name,lname) is None:
						raise SaHuTOrEPoLError(
							f"Member {name!r} not found in {lname!r}",
							ptr.pos
						)
					elif check_for_member(name,lname) != "Type":
						raise SaHuTOrEPoLError(
							f"Member {name!r} is not a type",
							ptr.pos
						)
					tree["type_defs"][name] = {
						'type': 'type_include',
						'name': name,
						'source': lname,
						'pos': ptr.pos
					}
					d_exp = True
					symbol = ""

				if m:=re.fullmatch(
					rf"\$({RegexBank.variable_name}) ?\$([a-zA-Z0-9\-\.]+)",symbol
				):
					name = m.group(1)
					lname= m.group(2)
					if check_for_member(name,lname) is None:
						raise SaHuTOrEPoLError(
							f"Member {name!r} not found in {lname!r}",
							ptr.pos
						)
					elif check_for_member(name,lname) != "Var":
						raise SaHuTOrEPoLError(
							f"Member {name!r} is not a variable",
							ptr.pos
						)
					context[-1]['children'].append({
						'type': 'var_inc',
						'name': name,
						'source': lname,
						'pos': ptr.pos
					})
					d_exp = True
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
				args = [i.strip() for i in args.split("|")]
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
					"file": tree["file"]
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

		cached_parsed_files[file_name] = tree

		return tree
