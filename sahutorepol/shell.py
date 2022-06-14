from sys import _getframe
from typing import IO, Generic, Optional, TypeVar
from sahutorepol import SaHuTOrEPoLWarning
from sahutorepol import Types
from sahutorepol.Errors import (
	SaHuTOrEPoLError, TracebackHint, format_error_or_warning
)
from sahutorepol.Types import TypeLike
from simple_warnings import catch_warnings

from sahutorepol.main import parse, NamespaceContext, Code, parse_expr


T = TypeVar('T', str, bytes)

class Shell:
	write_to: None | IO[str]

	class OutIO:
		def __init__(self, shell:"Shell"):
			self.shell = shell
			self.buffer = ""
			self.since_last_flush = 0

		def write(self, s:str):
			if self.shell.write_to is not None:
				self.shell.write_to.write(s)
				if self.since_last_flush == 100:
					self.shell.write_to.flush()
					self.since_last_flush = 0
				self.since_last_flush += 1
			self.buffer += s

		def writelines(self, lines:list):
			for line in lines[:-1]:
				self.write(line+"\n")
			self.write(lines[-1])

	def __init__(self, write_to:None | IO[str] = None):
		self.namespace = NamespaceContext()
		self.previous_lines:list[str] = []
		self.times_ran = 0
		self.output_buffer:list[str] = []
		self.out_io = self.OutIO(self)
		Types.S.stdout = self.out_io  # type:ignore
		self.file_dict = {}
		self.write_to = write_to

	def write(self, line:str):
		if line.startswith("?"):
			match line[1:].split():
				case ["plines"]:
					self.output(repr(self.previous_lines))
				case ["obuff"]:
					self.output(repr(self.output_buffer))
				case ["fdict"]:
					self.output(repr(self.file_dict))
				case ["fdict",x]:
					try:
						if x.isnumeric():
							self.output(repr(self.file_dict[f"<shell:{x}>"]))
						else:
							self.output(repr(self.file_dict[x]))
					except KeyError:
						self.output(f"No file with name {x}")
				case ["echo", *x]:
					self.output(" ".join(x))
			return

		# check if the line is not finished
		# this can be the case if we are in a function or method definition,
		# a while loop or a if statement
		finished = True
		namespace_context = self.namespace  # so the namespace is able to find itself
		_ = namespace_context

		# start of function or method definition
		# essentials checks for this: `$*+(*+)`
		if line.startswith("$") and line.count("(") > 1 and line.endswith(")"):
			finished = False

		# start of while|if
		if line.startswith("while") or line.startswith("if"):
			finished = False

		# indent
		if line.startswith("\t") or line.startswith(" "):
			finished = False

		# comment
		if line.startswith("$$"):
			finished = finished or bool(len(self.previous_lines))

		self.previous_lines.append(line)
		if not(finished):
			return
		self.file_dict[f"<shell:{self.times_ran}>"] = self.previous_lines
		lines = "\n".join(self.previous_lines)
		self.previous_lines = []

		# add lines to the file_dict
		with catch_warnings(True) as w:
			if lines.startswith(":"):
				try:
					with TracebackHint((0,0),file=f"<shell:{self.times_ran}>"):
						expr = parse_expr(lines[1:], (1,1))
					with self.namespace, TracebackHint((0,0),file=f"<shell:{self.times_ran}>"):
						res = Code.resolve_expr(expr)
					self.output(res)
				except Exception as ex:
					self.output(ex)
			else:
				try:
					c = Code(parse(lines+"\n", f"<shell:{self.times_ran}>",False))
					with self.namespace:
						c.run()
				except Exception as ex:
					self.output(ex)

		self.times_ran += 1

		for i in w:
			self.output(i)

	def write_lines(self, lines:list[str]):
		for line in lines:
			self.write(line)

	def output(self, result: Exception | Warning | TypeLike | str, force_write=True):
		if isinstance(result,(SaHuTOrEPoLError,SaHuTOrEPoLWarning)):
			text = format_error_or_warning(result,file_dict=self.file_dict)
			self.out_io.writelines(
				text.split("\n")
			)
		elif isinstance(result,(Warning,Exception)):
			text = f"{result.__class__.__name__}: {result}"
			text+= "\n"
			self.out_io.write(text)
		elif isinstance(result,str):
			self.out_io.writelines((result+"\n").split("\n"))
		else:
			try:
				text=Types.s(result).value
			except TypeError:
				text=str(result)
			text+= "\n"
			self.out_io.write(text)

	def read(self):
		out = "".join([i + "\n" for i in self.output_buffer])
		self.output_buffer = []
		return out

	def read_lines(self):
		return self.output_buffer.pop(0)


class Inject:
	class IOStealer:
		@staticmethod
		def readline():
			if (iow:=Inject.IOWrapper.get()) is None:
				raise ValueError("No IOWrapper found")
			return iow.readline()

		@staticmethod
		def write(s):
			if (iow:=Inject.IOWrapper.get()) is None:
				raise ValueError("No IOWrapper found")
			return iow.write(s)

		@staticmethod
		def flush():
			pass

	class IOWrapper(Generic[T]):
		__hidden_var_name = "__hidden_IOWrapper_shh"

		def __init__(self, wrapped_write: IO[T], wrapped_read:Optional[IO[T]] = None)\
				-> None:
			# get the outside frame's locals
			self._loc_ref = _getframe(1).f_locals
			self.wrapped_write = wrapped_write
			self.wrapped_read = wrapped_write if wrapped_read is None else wrapped_read

		def __enter__(self):
			# add self to the outside locals
			self.prev = self._loc_ref.get(self.__hidden_var_name,None)
			self._loc_ref[self.__hidden_var_name] = self
			return self

		def __exit__(self, exc_type, exc_val, exc_tb):
			# remove self from the outside locals
			self._loc_ref[self.__hidden_var_name] = self.prev

		def readline(self) -> T:
			return self.wrapped_read.readline()

		def write(self, s: T):
			self.wrapped_write.write(s)

		@classmethod
		def get(cls) -> 'Inject.IOWrapper | None':
			frame = _getframe(1)
			while cls.__hidden_var_name not in frame.f_locals:
				frame = frame.f_back
				if frame is None:
					return None
			return frame.f_locals[cls.__hidden_var_name]
