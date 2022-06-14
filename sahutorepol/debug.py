from main import parse, Code
import cProfile

with open(r"..\..\SaHuTOrEPoL\examples\count_to_1000.sah") as f:
	code = Code(parse(f.read(),"count_to_1000.sah"))

with cProfile.Profile() as pr:
	code.run()

pr.dump_stats(r"..\profile.prof")
