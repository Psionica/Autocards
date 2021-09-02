# Autocards

- Automatically create questions and answers from various input formats (PDF files, webpages, wikipedia, epub files, etc) then export them into your favorite flashcards software (like [anki](https://apps.ankiweb.net/) or [SuperMemo](https://supermemo.guru/wiki/SuperMemo))
- To see a real world example, the complete output of [this article](https://www.biography.com/political-figure/philip-ii-of-macedon) can be found [in this folder](./output_example/). It's the direct output with no post processing whatsoever.
- Code is PEP compliant and with docstrings. Contributions and PR are extremely appreciated
- Learn more by reading [the official write-up](https://psionica.org/tools/autocards/).

## Install guide:

- `git clone https//github.com/Psionica/Autocards`
- `cd Autocards`
- `pip install -r ./requirements.txt`
- open a python console: `ipython3`
- install punkt by running `!python -m nltk.downloader punkt`

### Autocards usage

```
# loading
from autocards import Autocards
a = Autocards()

# eating the input text using one of the following ways:
a.consume_var(my_text, per_paragraph=True)
a.consume_user_input(title="")
a.consume_textfile(filename, per_paragraph=True)
a.consume_pdf(pdf_path, per_paragraph=True)
a.consume_web(source, mode="url", element="p")
# =>  * element is the html element, like p for paragraph
#     * mode can be "url" or "local"

# three ways to get the results back: printing, pandas, export
out = a.string_output(prefix='', jeopardy=False)
#    =>  * prefix is a text that will be appended before the qa
#        * jeopardy is when switching question and answer
a.print(prefix='', jeopardy=False)
a.pprint(prefix='', jeopardy=False)  # pretty printing
df = a.pandas_output(prefix='')
a.to_csv("output.csv", prefix="", jeopardy=False)
a.to_json("output.json", prefix="", jeopardy=False)

# Also note that a user provided his own terrible scripts that you can get inspiration from, they are located in the folder `examples_script`
```
