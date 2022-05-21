#!/usr/bin/env python3


# import sys
# sys.path.append("../../.")
from autocards.autocards import Autocards
from pathlib import Path


prefix = "On Napoléon : "
file = Path("/path/to/Autocards/examples_script/napoleon/napoleon.txt")

if not file.exists():
    print("File not found!")
    raise SystemExit()
else:
    full_text = file.read_text()[:1_000]

auto = Autocards()
auto.consume_var(full_text)
auto.to_json("output.json", prefix="")

auto.print(prefix)
