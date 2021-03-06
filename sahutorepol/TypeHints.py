from typing import Literal, TypeAlias, TypedDict


class AST:
	class Element(TypedDict):
		pos: tuple[int, int]

	class Context(Element):
		children: list["AST.Elements"]

	class Root(Context):
		type: Literal['root']
		file: str
		type_defs:dict[str,"AST.TypeDefOrInclude"]
		imports: dict[str, "AST.Root"]

	class If(Context):
		type: Literal['if']
		expression: "AST.Expression"

	class While(Context):
		type: Literal['while']
		expression: "AST.Expression"

	class TypeDef(Context):
		type: Literal['type_def']
		name: str

	class TypeInclude(Element):
		type: Literal['type_include']
		name: str
		source: str

	TypeDefOrInclude = TypeDef | TypeInclude

	class EmptyVarDef(Element):
		type: Literal['var_def']
		name: str

	class FuncOrMethodDef(Context):
		type: Literal['func_def']
		name: str
		args: list[str]
		file: str

	class VarInclude(Element):
		type: Literal['var_inc']
		name: str
		source: str

	VarDef: TypeAlias = EmptyVarDef | FuncOrMethodDef | VarInclude

	class FileCapture(Context):
		type: Literal['file_capture']
		name: str

	class VarSet(Element):
		type: Literal['var_set']
		name: str
		value: "AST.Expression"

	class MethodCall(Element):
		type: Literal['method_call']
		name: str
		args: list["AST.Expression"]

	class Expresions:
		class Base(TypedDict):
			pos: tuple[int, int]

		class Multi(Base):
			type: Literal['multi_expression']
			expressions: list["AST.Expression"]

		class Var(Base):
			type: Literal['var_expression']
			name: str

		class Literal_(Base):
			type: Literal['literal_expression']
			value: "AST.Expresions.LiteralValue"

		class LiteralValues:
			class s(TypedDict):
				type: Literal['s']
				value: str

			class i(TypedDict):
				type: Literal['i']
				value: int

			class f(TypedDict):
				type: Literal['n']
				value: float

			class b(TypedDict):
				type: Literal['b']
				value: bool

			any: TypeAlias = s | i | f | b

		class FunctionCall(Base):
			type: Literal['function_call']
			name: str
			args: list["AST.Expression"]

		class ConstructorCall(Base):
			type: Literal['constructor_call']
			name: str
			args: list["AST.Expression"]

		LiteralValue: TypeAlias = LiteralValues.any

	Expression: TypeAlias = Expresions.Multi | Expresions.Var \
		| Expresions.Literal_ | Expresions.FunctionCall | Expresions.ConstructorCall

	Elements: TypeAlias = Root | While | If | VarDef | VarSet\
		| MethodCall
	Contexts: TypeAlias = Root | While | If | FuncOrMethodDef | TypeDef

	Members: TypeAlias = TypeDefOrInclude | FuncOrMethodDef
