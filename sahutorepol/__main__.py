import functools
from pathlib import Path
import sys
from typing import IO
import io
import click
from sahutorepol.main import LibraryPath
from sahutorepol.shell import Shell, Inject
from simple_warnings import catch_warnings, print_warning
import yaml
import json

from sahutorepol import Code, SaHuTOrEPoLError, SaHuTOrEPoLWarning, parse, S
from sahutorepol.Errors import (
	SaHuTOrEPoLKeyBoardInterrupt, show_error_or_warning
)


# from https://stackoverflow.com/a/510364/12469275
class _Getch:
	"""Gets a single character from standard input.  Does not echo to the
	screen."""
	def __init__(self):
		try:
			self.impl = _GetchWindows()
		except ImportError:
			self.impl = _GetchUnix()

	def __call__(self):
		return self.impl()


class _GetchUnix:
	def __init__(self):
		import tty

	def __call__(self):
		import tty, termios
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(sys.stdin.fileno())
			ch = sys.stdin.read(1)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		return ch


class _GetchWindows:
	def __init__(self):
		import msvcrt

	def __call__(self):
		import msvcrt
		return msvcrt.getch()


getch = _Getch()


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


def old_main(*args):
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


def strip_python_tags(s):
	# from: https://stackoverflow.com/a/55828059
	result = []
	for line in s.splitlines():
		idx = line.find("!!python/")
		if idx > -1:
			line = line[:idx]
		result.append(line)
	return '\n'.join(result)


@click.group("parsing and checking",invoke_without_command=True)
@click.option(
	"--raise","-r","do_raise",is_flag=True,
)
@click.option(
	"--silent","-s",is_flag=True,
)
@click.option(
	"--strict","-S",is_flag=True,
)
@click.pass_context
def cli(
		ctx:click.Context,do_raise:bool,silent:bool,strict:bool
	):
	ctx.ensure_object(dict)
	ctx.obj["raise"]  = do_raise
	ctx.obj["silent"] = silent
	ctx.obj["strict"] = strict
	# if no command is issued, go to a shell like interface
	if ctx.invoked_subcommand is None:
		shell = Shell(sys.stdout)
		while True:
			try:
				cmd = input("  ... " if shell.previous_lines else "SaH > ")
			except EOFError or KeyboardInterrupt:
				print()
				break
			shell.write(cmd)


def parse_command(func):
	@click.argument(
		"file",type=click.File()
	)
	@click.option(
		"--add-path","-ap",type=click.Path(exists=True),multiple=True,
	)
	@click.option(
		"path","--path",type=click.Path(exists=True),envvar="SaHPath"
	)
	@click.option(
		"--dont-include-here","-d",is_flag=True,
	)
	@click.pass_context
	# @click.option(
	# 	"std","--std",type=click.Path(exists=True),deafult=
	# )
	# REQUIRE[STD]: ./main.py#STD
	# TODO: When the std library is implemented, allow for the default (the one comming with the interpreter) to be overridden.
	@functools.wraps(func)
	def wrapper(
			ctx: click.Context,file: IO[str],add_path:list[str],dont_include_here:bool,path:str,
			*args,**kwargs
		):
		# get options from context
		do_raise = ctx.obj["raise"]
		silent   = ctx.obj["silent"]
		strict   = ctx.obj["strict"]
		fin_path = []
		if not(dont_include_here):
			fin_path.append(Path("."))
		# path.append(Path(std))
		if add_path is not None:
			fin_path.extend(Path(p) for p in add_path)
		if path is not None:
			fin_path.extend(Path(p) for p in path.split(":"))
		ctx.obj["path"] = fin_path
		with catch_warnings(record=True) as w:
			try:
				with LibraryPath(*fin_path):
					t = parse(file.read(),file.name)
				# print("parsed")
			except (SaHuTOrEPoLError,SaHuTOrEPoLKeyBoardInterrupt) as ex:
				if do_raise:
					raise ex from ex
				show_error_or_warning(ex)
				sys.exit(1)
			if strict or not silent:
				for warning in w:
					if silent:
						continue
					if isinstance(warning,SaHuTOrEPoLWarning):
						show_error_or_warning(warning)
					else:
						print_warning(warning)
				if not strict and w:
					sys.exit(1)
			if t is None:
				raise RuntimeError
		if not silent:
			for warning in w:
				if isinstance(warning,SaHuTOrEPoLWarning):
					show_error_or_warning(warning)
				else:
					print_warning(warning)
		if w.warnings and strict:
			sys.exit(1)
		ctx.obj['ast'] = t
		return func(ctx,t,*args,**kwargs)
	return wrapper


@cli.command("parse")
@parse_command
@click.argument(
	"output",type=click.File(mode="w"),
)
@click.option(
	"--yaml","-y","yaml_",is_flag=True,
)
def parse_(ctx:click.Context,t,output:IO,yaml_:bool):
	# write
	if yaml_:
		dump = yaml.dump(t,default_flow_style=False)
		dump = strip_python_tags(dump)
	else:
		dump = json.dumps(t,indent=2)
	output.write(dump)


@cli.command("run")
@parse_command
def run(ctx:click.Context,t):
	do_raise = ctx.obj["raise"]
	silent   = ctx.obj["silent"]
	strict   = ctx.obj["strict"]
	with catch_warnings(record=True) as w:
		try:
			Code(t).run()
		except (SaHuTOrEPoLError,SaHuTOrEPoLKeyBoardInterrupt) as ex:
			if do_raise:
				raise ex from ex
			show_error_or_warning(ex)
			sys.exit(1)
	if not silent:
		for warning in w:
			if isinstance(warning,SaHuTOrEPoLWarning):
				show_error_or_warning(warning)
			else:
				print_warning(warning)
	if w.warnings and strict:
		sys.exit(1)


@cli.command("debug")
@click.option(
	"--out-height", "-h", type=int, default=16,
)
@parse_command
def debug(ctx:click.Context,t,out_height:int):
	do_raise = ctx.obj["raise"]
	silent   = ctx.obj["silent"]
	strict   = ctx.obj["strict"]
	# inject into stdin and stdout
	S.stdin  = Inject.IOStealer()  # type:ignore
	S.stdout = Inject.IOStealer()  # type:ignore

	# create the input manager
	class InputManager:
		def readline(self):
			# go to the bottom of the screen
			S.stdout.write("\x1b[{}A".format(out_height))
			# print the prompt
			print(": ",end="",flush=True)
			# read the input
			return input()

	im = InputManager()
	o  = io.StringIO()

	# clear the screen
	print("\033[2J\033[1;1H",end="")

	with catch_warnings(record=True) as w:
		try:
			with Inject.IOWrapper(o,im):
				for i in Code(t):
					# clear the console
					print("\033[2J\033[1;1H",end="")

					# print the last something lines of the output
					print(*o.getvalue().splitlines()[-out_height:],sep="\n")

					# pad the output with empty lines
					for _ in range(max(out_height - len(o.getvalue().splitlines()),0)):
						print()


					print()
					print(i[1].replace('\\','/').split('/')[-1]+f":{i[0]}")

					print(":",end="",flush=True)
					c = getch()
					if (c == b"\x03"    # ctrl-c
					 or c == b"\x1a"):  # ctrl-d
						raise KeyboardInterrupt

		except (SaHuTOrEPoLError,SaHuTOrEPoLKeyBoardInterrupt) as ex:
			if do_raise:
				raise ex from ex
			show_error_or_warning(ex)
			sys.exit(1)
	if not silent:
		for warning in w:
			if isinstance(warning,SaHuTOrEPoLWarning):
				show_error_or_warning(warning)
			else:
				print_warning(warning)
	if w.warnings and strict:
		sys.exit(1)


if __name__ == "__main__":
	# print(*sys.argv[1:])
	cli()
