import traceback
import sys


class WarningCatcher(object):
	__hidden_var_name = '__hidden warning catcher__'
	warnings: list[Warning] = []

	def __init__(self,record=True,frames_up=1) -> None:
		self.log = record
		if record:
			self.warnings = []
		self.__locals = sys._getframe(frames_up).f_locals

	def __enter__(self) -> 'WarningCatcher':
		self.__prev = self.__locals.get(self.__hidden_var_name, None)
		self.__locals[self.__hidden_var_name] = self
		return self

	def __exit__(self, *args) -> None:
		if self.__prev is not None:
			self.__locals[self.__hidden_var_name] = self.__prev

	def warn(self, warning: Warning) -> None:
		if self.log:
			self.warnings.append(warning)
		if self.__prev is not None:
			self.__prev.warn()

	@classmethod
	def get_current_warning_catcher(cls) -> 'WarningCatcher | None':
		r = None
		i = 1
		while r is None:
			try:
				r = sys._getframe(i).f_locals.get(cls.__hidden_var_name, None)
			except ValueError:
				break
			i += 1
		return r

	def __iter__(self):
		return iter(self.warnings)

	def __get_item__(self, index):
		return self.warnings[index]

	def __bool__(self):
		return bool(self.warnings)


def catch_warnings(record=True) -> WarningCatcher:
	"""
		Return a WarningCatcher object.
	"""
	return WarningCatcher(record,2)


def print_warning(warning: Warning) -> None:
	traceback.print_exception(warning.__class__, warning, warning.__traceback__)


showwarning = print_warning


def warn(warning: Warning | str) -> None:
	"""
		Rise a warning.
	"""
	wc = WarningCatcher.get_current_warning_catcher()
	if isinstance(warning, str):
		warning = Warning(warning)
	if wc is not None:
		wc.warn(warning)
	else:
		print_warning(warning)
