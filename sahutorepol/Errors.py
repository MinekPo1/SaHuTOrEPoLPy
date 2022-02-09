from sys import _getframe
from typing import ClassVar
import pathlib


class _dotraceback(object):
	traceback: list['TracebackPoint']
	str_type: ClassVar[str] = "Unassigned"

	def __init__(self,message: str,pos: tuple):
		self.message = message
		self.pos = pos
		self.traceback = []


class SaHuTOrEPoLError(_dotraceback,Exception):
	"""
		Error raised by the parser or interpreter.
	"""
	pos: tuple[int, int]
	str_type = "Error"

	def __str__(self) -> str:
		return (
			f"{self.message} "
			f"at line {self.pos[0]}, column {self.pos[1]}"
		)

	def __repr__(self) -> str:
		return (
			f"{type(self).__name__}:{self.message} "
			f"at line {self.pos[0]}, column {self.pos[1]}"
		)


class SaHuTOrEPoLWarning(_dotraceback,Warning):
	"""
		Warning raised by the parser or interpreter.
	"""
	pos: tuple[int, int]
	str_type = "Warning"

	def __str__(self) -> str:
		return (
			f"{self.message} "
			f"at line {self.pos[0]}, column {self.pos[1]}"
		)

	def __repr__(self) -> str:
		return (
			f"{type(self).__name__}:{self.message} "
			f"at line {self.pos[0]}, column {self.pos[1]}"
		)


class SaHuTOrEPoLKeyBoardInterrupt(_dotraceback,KeyboardInterrupt):
	"""
		Raised when the user interrupts the program.
	"""
	pos: tuple[int, int]
	str_type = "Interrupt"

	def __str__(self) -> str:
		return (
			f"{self.message} "
			f"at line {self.pos[0]}, column {self.pos[1]}"
		)

	def __repr__(self) -> str:
		return (
			f"{type(self).__name__}:{self.message} "
			f"at line {self.pos[0]}, column {self.pos[1]}"
		)


def format_path(path: str) -> str:
	p_path = pathlib.Path(path)
	if p_path.is_absolute():
		return p_path.as_posix()
	return f"./{p_path.as_posix()}"


def show_error_or_warning(
		error: SaHuTOrEPoLError | SaHuTOrEPoLKeyBoardInterrupt | SaHuTOrEPoLWarning,
		file_dict: dict[str, list[str] | str] = None
	):
	"""
	Show the error message and the code snippet where it occurred.
	"""
	if file_dict is None:
		file_dict = {}
	print()

	i = None

	# print traceback
	for j,i in enumerate(error.traceback):
		if isinstance(i, TracebackHint) and j != len(error.traceback)-1:
			continue
		mes = f"In {i.name} on line" if i.name is not None else "Line"
		if i.file is not None:
			mes += f" in {format_path(i.file)}:"
		mes += f"{i.pos[0]}"
		print(mes)
		if i.file is None:
			continue
		if i.file not in file_dict:
			try:
				with open(i.file) as f:
					file_dict[i.file] = f.readlines()
			except Exception:
				file_dict[i.file] = "Unable to read file"

		if isinstance(file_dict[i.file], str):
			print(file_dict[i.file])
		else:
			line = file_dict[i.file][i.pos[0] - 1]
			print(line,end="")
			tabs = line[:i.pos[1] - 1].count("\t")
			print("\t"*tabs+" " * (i.pos[1] - 1-tabs) + "^")

	print(f"{error.str_type}: {error.message}")


class TracebackPoint(object):
	"""
		A point in the traceback. Use as a context manager.
	"""
	def __init__(self, pos: tuple[int, int], file: str = None, name:str = None)\
			-> None:
		self.pos = pos
		self.file = file
		self.name = name

	def __enter__(self) -> None:
		self.local_ref = _getframe(1).f_locals
		self.__prev = _getframe(1).f_locals.get("__traceback_point__", None)
		self.prev = self.get_closest_point() if self.__prev is None else self.__prev
		if self.file is None:
			if self.prev is None:
				raise ValueError("No file given and no previous traceback point.")
			self.file = self.prev.file
		if self.name is None and self.prev is not None:
			self.name = self.prev.name
		self.local_ref["__traceback_point__"] = self

	def __exit__(self, exc_type, exc_val, exc_tb) -> None:
		self.local_ref["__traceback_point__"] = self.__prev
		# if a exception was raised, we want to add self to the traceback

		if exc_type is not None and issubclass(exc_type, _dotraceback):
			exc_val.traceback.insert(0,self)

	@classmethod
	def get_closest_point(cls):
		"""
			Get the closest point in the traceback.
		"""
		c = None
		i = 0
		try:
			while c is None:
				i+=1
				c = _getframe(i).f_locals.get("__traceback_point__", None)
		except ValueError:
			return None
		return c


class TracebackHint(TracebackPoint):
	"""
		Allows adding lines with out showing up in the traceback.
	"""
	pass
