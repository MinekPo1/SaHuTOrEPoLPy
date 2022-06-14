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
from sahutorepol.shell import Inject
import sahutorepol                                    # noqa: E402

IOStealer = Inject.IOStealer
IOWrapper = Inject.IOWrapper


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
					if j:
						if j.startswith("[") and j.endswith("]"):
							j = j[1:-1]
						self.assertEqual(j,subtest_o.readline().rstrip("\n"))


# inject the IOStealer

sahutorepol.S.stdin  = IOStealer()  # type:ignore
sahutorepol.S.stdout = IOStealer()  # type:ignore

if __name__ == "__main__":
	unittest.main()
