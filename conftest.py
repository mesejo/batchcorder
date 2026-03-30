from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser


pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser(), SkipParser()],
    patterns=["README.md"],
).pytest()
