from autocards.autocards import Autocards
from fastcore.script import *

@call_parse
def main(
    mode: Param(
        "The type of data that will be autocarded",
        str,
        choices = ["pdf", "epub", "raw", "textfile", "url"]
    ),
    output: Param(
        "The method to output the file as",
        str,
        choices = ["anki", "csv", "json", "print"]
    ),
    content: Param("The content to be autocarded", str),
):
    "Accelerating learning through machine-generated flashcards."

    auto = Autocards()
    if mode == "pdf":
        auto.consume_pdf(content)
    elif mode == "epub":
        auto.consume_epub(content)
    elif mode == "raw":
        auto.consume_var(content)
    elif mode == "textfile":
        auto.consume_textfile(content)
    elif mode == "url":
        auto.consume_web(content)

    if output == "anki":
        auto.to_anki(deckname="autocards_export", tags=[mode])
    elif output == "csv":
        auto.to_csv("output.csv", prefix="")
    elif output == "json":
        auto.to_json("output.json", prefix="")
    elif output == "print":
        auto.pprint(prefix='', jeopardy=False)