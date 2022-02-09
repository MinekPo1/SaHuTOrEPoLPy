from typing import Literal, TypeAlias, TypedDict


class AST:
	class Element(TypedDict):
		pos: tuple[int, int]

	class Context(Element):
		children: list["AST.Elements"]

	class Root(Context):
		type: Literal['root']
		file: str
		type_defs:dict[str,"AST.TypeDef"]

	class If(Context):
		type: Literal['if']
		expression: "AST.Expression"

	class While(Context):
		type: Literal['while']
		expression: "AST.Expression"

	class TypeDef(Context):
		type: Literal['type_def']
		name: str

	class EmptyVarDef(Element):
		type: Literal['var_def']
		name: str

	class FuncOrMethodDef(Context):
		type: Literal['func_def']
		name: str
		args: list[str]

	VarDef: TypeAlias = EmptyVarDef | FuncOrMethodDef

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
				type: Literal['f']
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

	Members: TypeAlias = TypeDef | FuncOrMethodDef
