from pathlib import Path
from typing import IO, Optional, TypeVar, Generic
from parameterized import parameterized
import unittest
from sys import _getframe
from io import StringIO

# make sure we use the local version of the module
import sys

sys.path.insert(0,str(Path(__file__).parent.parent))

from sahutorepol import Code, SaHuTOrEPoLError        # noqa: E402
from sahutorepol import parse                         # noqa: E402
from sahutorepol.Errors import show_error_or_warning  # noqa: E402
import sahutorepol                                    # noqa: E402


class IOStealer:
	@staticmethod
	def readline():
		if (iow:=IOWrapper.get()) is None:
			raise ValueError("No IOWrapper found")
		return iow.readline()

	@staticmethod
	def write(s):
		if (iow:=IOWrapper.get()) is None:
			raise ValueError("No IOWrapper found")
		return iow.write(s)


T = TypeVar('T', str, bytes)


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
	def get(cls) -> 'IOWrapper | None':
		frame = _getframe(1)
		while cls.__hidden_var_name not in frame.f_locals:
			frame = frame.f_back
			if frame is None:
				return None
		return frame.f_locals[cls.__hidden_var_name]


def run(fname: str | Path):
	if isinstance(fname,str):
		fname = Path(fname)
	with fname.open() as f:
		code = Code(parse(f.read(),fname.as_posix()))
	try:
		code.run()
		return True, None
	except SaHuTOrEPoLError as ex:
		return False, ex


class GenericTests(unittest.TestCase):
	test_path = Path(r"..\SaHuTOrEPoL\tests")

	# TODO: try adding the name_func to the parameterized decorator
	@parameterized.expand(
		((i.name.removesuffix(".test"),i,) for i in test_path.glob("**/*.test"))
	)
	def test(self,name,fname: Path):
		test_details = []
		with fname.open() as f:
			for i in f.read().split("==="):
				test_details.append([j for j in i.split("---") if not(j.startswith("$$"))])
		for i,test in enumerate(test_details):
			subtest_i = StringIO()
			subtest_o = StringIO()
			with (
				self.subTest(subtest=i),
				IOWrapper(subtest_o,subtest_i)
			):
				code_file = fname.parent / (fname.name.removesuffix(".test") + ".sah")
				# place the input into the buffer
				for j in test[0].split("\n"):
					if j.startswith("[") and j.endswith("]"):
						subtest_i.write(j[1:-1])
						subtest_i.write("\n")
					if j:
						subtest_i.write(j+"\n")
				subtest_i.seek(0)
				# run the code
				success, error = run(code_file)
				subtest_o.seek(0)
				# check the output
				for j in test[1].split("\n"):
					if j.startswith("@"):
						match j:
							case "@success":
								if not success:
									print(f"Note: for test {name} subtest {i}, error:")
									show_error_or_warning(error)  # type:ignore
								self.assertTrue(success,"Code had an unexpected error")
							case "@failure":
								self.assertFalse(success,"Code did not have an expected error")
							case _:
								raise ValueError("Unknown @ directive")
						continue
					if j.startswith("[") and j.endswith("]"):
						j = j[1:-1]
					if j:
						self.assertEqual(j,subtest_o.readline().rstrip("\n"))


# inject the IOStealer

sahutorepol.S.stdin  = IOStealer()  # type:ignore
sahutorepol.S.stdout = IOStealer()  # type:ignore

if __name__ == "__main__":
	unittest.main()
