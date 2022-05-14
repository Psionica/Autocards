from autocards.pipelines import qg_pipeline

from tqdm import tqdm
from pathlib import Path
import pandas as pd
import time
import re
import os
from contextlib import suppress

import json
import urllib.request
import requests
from tika import parser
from bs4 import BeautifulSoup
from pprint import pprint
from epub_conversion.utils import open_book, convert_epub_to_lines

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Autocards:
    """
    Main class used to create flashcards from text. The variable
    'store_content' defines whether the original paragraph is stored in the
    output. This allows to store context alongside the question and answer pair
    but dramatically increase size. The variable notetype refers to the type
    of flashcard that must be created: either cloze, basic or both. The
    variable wtm allow to specify wether you want to remove the mention of
    Autocards in your cards.
    """

    def __init__(self,
                 store_content=True,
                 in_lang="any",
                 out_lang="en",
                 cloze_type="anki",
                 model = "valhalla/distilt5-qa-qg-hl-12-6",
                 ans_model = "valhalla/distilt5-qa-qg-hl-12-6"):
        print("Loading backend, this can take some time...")
        self.store_content = store_content
        self.model = model
        self.ans_model = ans_model

        if len(out_lang) != 2 or len(in_lang) not in [2, 3]:
            print("Output and input language has to be a two letter code like 'en' or 'fr'")
            raise SystemExit()
        if in_lang == "any":  # otherwise the user might thought that the
            in_lang = "en"    # input has to be in english
        if in_lang != "en":
            print("The document will automatically be translated before \
creating flashcards. Expect lower quality cards than usual.")
            try:
                print("Loading input translation model...")
                from transformers import pipeline
                self.in_trans = pipeline(f"translation_{in_lang}_to_en",
                                      model = f"Helsinki-NLP/opus-mt-{in_lang}-en")
            except Exception as e:
                print(f"Was not able to load translation pipeline: {e}")
                print("Resetting input language to english.")
                in_lang = "en"
        if out_lang != "en":
            print("The flashcards will be automatically translated after being \
created. This can result in lower quality cards. Expect lowest quality cards \
than usual.")
            try:
                print("Loading output translation model...")
                from transformers import pipeline
                self.out_trans = pipeline(f"translation_en_to_{out_lang}",
                                      model = f"Helsinki-NLP/opus-mt-en-{out_lang}")
            except Exception as e:
                print(f"Was not able to load translation pipeline: {e}")
                print("Resetting output language to english.")
                out_lang = "en"
        self.in_lang = in_lang
        self.out_lang = out_lang

        self.cloze_type = cloze_type
        self.qg = qg_pipeline('question-generation',
                              model=model,
                              ans_model=ans_model)
        self.qa_dic_list = []

        if self.cloze_type not in ["anki", "SM"]:
            print("Invalid cloze type, must be either 'anki' or \
'SM'")
            raise SystemExit()

    def _call_qg(self, text, title):
        """
        Call question generation module, then turn the answer into a
        dictionnary containing metadata (clozed formating, creation time,
        title, source text)
        """
        to_add = []
        to_add_cloze = []
        to_add_basic = []
        if self.in_lang != "en":
            text_orig = str(text)
            text = self.in_trans(text)[0]["translation_text"]
        else:
            text_orig = ""

        try:
            to_add = self.qg(text)
            to_add_cloze = [qa for qa in to_add if qa["note_type"] == "cloze"]
            to_add_basic = [qa for qa in to_add if qa["note_type"] == "basic"]
        except IndexError:
            tqdm.write(f"\nSkipping section because no cards \
could be made from that text: '{text}'")
            to_add_basic.append({"question": "skipped",
                                 "answer": "skipped",
                                 "cloze": "",
                                 "note_type": "basic"})

        cur_time = time.asctime()

        if self.store_content is False:
            # don't store content, to minimize the size of the output file
            stored_text = ""
            stored_text_orig = ""
        else:
            stored_text = text
            stored_text_orig = text_orig

        # loop over all newly added qa to format the text:
        if to_add_basic != []:
            for i in range(0, len(to_add_basic)):
                if to_add_basic[i]["note_type"] == "basic":
                    if self.out_lang != "en":
                        to_add_basic[i]["question_orig"] = to_add_basic[i]["question"]
                        to_add_basic[i]["answer_orig"] = to_add_basic[i]["answer"]
                        to_add_basic[i]["question"] = self.out_trans(to_add_basic[i]["question"])[0]["translation_text"]
                        to_add_basic[i]["answer"] = self.out_trans(to_add_basic[i]["answer"])[0]["translation_text"]
                    else:
                        to_add_basic[i]["answer_orig"] = ""
                        to_add_basic[i]["question_orig"] = ""

                    clozed_fmt = to_add_basic[i]['question'] + "<br>{{c1::"\
                        + to_add_basic[i]['answer'] + "}}"
                    to_add_basic[i]["basic_in_clozed_format"] = clozed_fmt

        if to_add_cloze != []:
            for i in range(0, len(to_add_cloze)):
                if to_add_cloze[i]["note_type"] == "cloze":  # cloze formating
                    if self.out_lang != "en":
                        to_add_cloze[i]["cloze_orig"] = to_add_cloze[i]["cloze"]
                        cl_str_ut = to_add_cloze[i]["cloze_orig"]
                        cl_str_ut = cl_str_ut.replace("generate question: ", "")
                        cl_str_ut = cl_str_ut.replace("<hl> ", "{{c1::", 1)
                        cl_str_ut = cl_str_ut.replace(" <hl>", "}}", 1)
                        cl_str_ut = cl_str_ut.replace(" </s>", "")
                        cl_str_ut = cl_str_ut.strip()
                        to_add_cloze[i]["cloze_orig"] = cl_str_ut

                        cl_str = to_add_cloze[i]["cloze"]
                        cl_str = cl_str.replace("generate question: ", "")
                        cl_str = cl_str.replace("\"", "'")
                        cl_str = cl_str.replace("<hl> ", "\"").replace(" <hl>", "\"")
                        cl_str = cl_str.replace(" </s>", "")
                        cl_str = cl_str.strip()
                        cl_str = self.out_trans(cl_str)[0]["translation_text"]
                        cl_str = cl_str.replace("\"", "{{c1::", 1)
                        cl_str = cl_str.replace("\"", "}}", 1)
                        to_add_cloze[i]["cloze"] = cl_str
                    else:
                        to_add_cloze[i]["cloze_orig"] = ""

                        cl_str = to_add_cloze[i]["cloze"]
                        cl_str = cl_str.replace("generate question: ", "")
                        cl_str = cl_str.replace("<hl> ", "{{c1::", 1)
                        cl_str = cl_str.replace(" <hl>", "}}", 1)
                        cl_str = cl_str.replace(" </s>", "")
                        cl_str = cl_str.strip()
                        to_add_cloze[i]["cloze"] = cl_str

                    to_add_cloze[i]["basic_in_clozed_format"] = ""

        # merging cloze of the same text as a single qa with several cloze:
        if to_add_cloze != []:
            for i in range(0, len(to_add_cloze)-1):
                if self.cloze_type == "SM":
                    tqdm.write("SM cloze not yet implemented, luckily \
SuperMemo supports importing from anki format. Hence the anki format will \
be used for your input.")
                    self.cloze_type = "anki"

                if self.cloze_type == "anki" and len(self.qa_dic_list) != i:
                    cl1 = re.sub(r"{{c\d+::|}}|\s", "",
                                 to_add_cloze[i]["cloze"])
                    cl2 = re.sub(r"{{c\d+::|}}|\s", "",
                                 to_add_cloze[i+1]["cloze"])
                    if cl1 == cl2:
                        match = re.findall(r"{{c\d+::(.*?)}}",
                                           to_add_cloze[i]["cloze"])
                        match.extend(re.findall(r"{{c\d+::(.*?)}}",
                                                to_add_cloze[i+1]["cloze"]))
                        clean_cloze = re.sub(r"{{c\d+::|}}", "",
                                             to_add_cloze[i]["cloze"])
                        if "" in match:
                            match.remove("")
                        match = list(set(match))
                        for cloze_number, q in enumerate(match):
                            q = q.strip()
                            new_q = "{{c" + str(cloze_number+1) + "::" +\
                                    q + "}}"
                            clean_cloze = clean_cloze.replace(q, new_q)
                        clean_cloze = clean_cloze.strip()

                        to_add_cloze[i]['cloze'] = clean_cloze + "___TO_REMOVE___"
                        to_add_cloze[i+1]['cloze'] = clean_cloze

        to_add_full = to_add_cloze + to_add_basic
        for qa in to_add_full:
            qa["date"] = cur_time
            qa["source_title"] = title
            qa["source_text"] = stored_text
            qa["source_text_orig"] = stored_text_orig
            if qa["note_type"] == "basic":
                self.qa_dic_list.append(qa)
            elif not qa["cloze"].endswith("___TO_REMOVE___"):
                self.qa_dic_list.append(qa)

        tqdm.write(f"Number of question generated so far: {len(self.qa_dic_list)}")

    def _sanitize_text(self, text):
        "correct common errors in text"
        text = text.strip()
        # occurs sometimes in epubs apparently:
        text = text.replace("\xa0", " ")
        # wikipedia style citation:
        text = re.sub(r"\[\d*\]", "", text)
        return text

    def consume_var(self, text, title="untitled variable",
                    per_paragraph=False):
        "Take text as input and create qa pairs"
        text = text.replace('\xad ', '')
        text = text.strip()
        self.title = title

        if per_paragraph:
            print("Consuming text by paragraph:")
            for paragraph in tqdm(text.split('\n\n'),
                                  desc="Processing by paragraph",
                                  unit="paragraph"):
                paragraph = paragraph.replace("\n", " ")
                self._call_qg(paragraph, title)
        else:
            print("Consuming text:")
            text = re.sub(r"\n\n*", ". ", text)
            text = re.sub(r"\.\.*", ".", text)
            text = self._sanitize_text(text)
            self._call_qg(text, title)

    def consume_user_input(self, title="untitled user input"):
        "Take user input and create qa pairs"
        user_input = input("Enter your text below then press Enter (press\
 enter twice to validate input):\n>")
        user_input = user_input.strip()

        print("\nFeeding your text to Autocards...")
        user_input = self._sanitize_text(user_input)
        self.consume_var(user_input, title, per_paragraph=False)
        print("Done feeding text.")

    def consume_pdf(self, pdf_path, per_paragraph=True):
        "Take pdf file as input and create qa pairs"
        if not Path(pdf_path).exists():
            print(f"PDF file not found at {pdf_path}!")
            return None

        print("Warning: pdf parsing is usually of poor quality because \
there are no good cross platform libraries. Consider using consume_textfile() \
after preprocessing the text yourself.")
        title = pdf_path.replace("\\", "").split("/")[-1]
        raw = str(parser.from_file(pdf_path))
        safe_text = raw.encode('utf-8', errors='ignore')
        safe_text = str(safe_text).replace("\\n", "\n").replace("\\t", " ").replace("\\", "")

        text = self._sanitize_text(safe_text)

        self.consume_var(text, title, per_paragraph)

    def consume_textfile(self, filepath, per_paragraph=False):
        "Take text file as input and create qa pairs"
        if not Path(filepath).exists():
            print(f"File not found at {filepath}")
        text = open(filepath).read()
        text = self._sanitize_text(text)
        filename = str(filepath).split("/")[-1]
        if per_paragraph is False and len(text) > 300:
            ans = input("The text is more than 300 characters long, \
are you sure you don't want to try to split the text by paragraph?\n(y/n)>")
            if ans != "n":
                per_paragraph = True
        self.consume_var(text,
                         filename,
                         per_paragraph=per_paragraph)

    def consume_epub(self, filepath, title="untitled epub file"):
        "Take an epub file as input and create qa pairs"
        book = open_book(filepath)
        text = " ".join(convert_epub_to_lines(book))
        text = re.sub("<.*?>", "", text)
        text = text.replace("&nbsp;", " ")
        text = text.replace("&dash;", "-")
        text = re.sub("&.*?;", " ", text)
        # make paragraph limitation as expected in self.consume_var:
        text = text.replace("\r", "\n\n")
        text = re.sub("\n\n\n*", "\n\n", text)
        text = self._sanitize_text(text)
        self.consume_var(text, title, per_paragraph=True)

    def consume_web(self, source, mode="url", element="p"):
        "Take html file (local or via url) and create qa pairs"
        if mode not in ["local", "url"]:
            return "invalid arguments"
        if mode == "local":
            soup = BeautifulSoup(open(source), 'xml')
        elif mode == "url":
            res = requests.get(source, timeout=15)
            html = res.content
            soup = BeautifulSoup(html, 'xml')

        try:
            el = soup.article.body.find_all(element)
        except AttributeError:
            print("Using fallback method to extract page content")
            el = soup.find_all(element)

        title = ""
        with suppress(Exception):
            title = soup.find_all('h1')[0].text
        if title == "":
            with suppress(Exception):
                title = soup.find_all('h1').text
        if title == "":
            with suppress(Exception):
                title = soup.find_all('title').text
        if title == "":
            print("Couldn't find title of the page")
            title = source
        title = title.strip()
        self.title = title

        valid_sections = []  # remove text sections that are too short:
        for section in el:
            section = ' '.join(section.get_text().split())
            if len(section) > 40:
                valid_sections += [section]
            else:
                print(f"Ignored string because too short: {section}")

        if not valid_sections:
            print("No valid sections found, change the 'element' argument\
 to look for other html sections than 'p'. Find the relevant 'element' using \
 the 'inspect' functionnality in your favorite browser.")
            return None

        for section in tqdm(valid_sections,
                            desc="Processing by section",
                            unit="section"):
            section = self._sanitize_text(section)
            self._call_qg(section, title)

    def clear_qa(self):
        "Delete currently stored qa pairs"
        self.qa_dic_list = []

    def string_output(self, prefix='', jeopardy=False):
        "Return qa pairs to the user"
        if prefix != "" and prefix[-1] != ' ':
            prefix += ' '
        if len(self.qa_dic_list) == 0:
            print("No qa generated yet!")
            return None

        res = []
        for qa_pair in self.qa_dic_list:
            if qa_pair['note_type'] == "basic":
                if jeopardy:
                    string = f"\"{prefix}{qa_pair['answer']}\",\" {qa_pair['question']}\""
                else:
                    string = f"\"{prefix}{qa_pair['question']}\",\" {qa_pair['answer']}\""
            elif qa_pair['note_type'] == "cloze":
                string = f"\"{prefix}{qa_pair['cloze']}\""
            res.append(string)
        return res

    def print(self, *args, **kwargs):
        "Print qa pairs to the user"
        print(self.string_output(*args, **kwargs))

    def pprint(self, *args, **kwargs):
        "Prettyprint qa pairs to the user"
        pprint(self.string_output(*args, **kwargs))

    def _combine_df_columns(self, row, col_names):
        combined = ""
        for col in col_names:
            combined += f"{col.upper()}: {dict(row)[col]}<br>\n"
        return "#"*15 + "Combined columns:<br>\n" + combined + "#"*15

    def pandas_df(self, prefix=''):
        if len(self.qa_dic_list) == 0:
            print("No qa generated yet!")
            return None
        "Output a Pandas DataFrame containing qa pairs and metadata"
        df = pd.DataFrame(columns=list(self.qa_dic_list[0].keys()))
        for qa in self.qa_dic_list:
            df = df.append(qa, ignore_index=True)
        for i in df.index:
            for c in df.columns:
                if pd.isna(df.loc[i, c]):
                    # otherwise export functions break:
                    df.loc[i, c] = ""
        if self.in_lang == "en":
            df = df.drop(columns=["source_text_orig"], axis=1)
        if self.out_lang == "en":
            df = df.drop(columns=["cloze_orig", "question_orig", "answer_orig"],
                         axis=1)
        df["combined_columns"] = [self._combine_df_columns(df.loc[x, :], df.columns)
                             for x in df.index ]
        return df

    def to_csv(self, filename="Autocards_export.csv", prefix=''):
        "Export qa pairs as csv file"
        if len(self.qa_dic_list) == 0:
            print("No qa generated yet!")
            return None
        if prefix != "" and prefix[-1] != ' ':
            prefix += ' '

        df = self.pandas_df(prefix)

        for i in df.index:
            for c in df.columns:
                df.loc[i, c] = str(df.loc[i, c]).replace(",", r"\,")

        if ".csv" in filename:
            filename = filename.replace(".csv", "")
        df[df["note_type"] == "cloze"].to_csv(f"{filename}_cloze.csv")
        df[df["note_type"] != "cloze"].to_csv(f"{filename}_basic.csv")
        print(f"Done writing qa pairs to {filename}_cloze.csv and {filename}_basic.csv")

    def to_json(self, filename="Autocards_export.json", prefix=''):
        "Export qa pairs as json file"
        if len(self.qa_dic_list) == 0:
            print("No qa generated yet!")
            return None
        if prefix != "" and prefix[-1] != ' ':
            prefix += ' '

        df = self.pandas_df(prefix)

        if ".json" in filename:
            filename = filename.replace(".json", "")
        df[df["note_type"] == "cloze"].to_json(f"{filename}_cloze.json")
        df[df["note_type"] != "cloze"].to_json(f"{filename}_basic.json")
        print(f"Done writing qa pairs to {filename}_cloze.json and \
{filename}_basic.json")

    def _ankiconnect_invoke(self, action, **params):
        "send requests to ankiconnect addon"

        def request_wrapper(action, **params):
            return {'action': action, 'params': params, 'version': 6}

        requestJson = json.dumps(request_wrapper(action, **params)
                                 ).encode('utf-8')
        try:
            response = json.load(urllib.request.urlopen(
                                    urllib.request.Request(
                                        'http://localhost:8765',
                                        requestJson)))
        except (ConnectionRefusedError, urllib.error.URLError) as e:
            print(f"{e}: is Anki open? Is the addon 'anki-connect' enabled?")
            raise SystemExit()
        if len(response) != 2:
            raise Exception('response has an unexpected number of fields')
        if 'error' not in response:
            raise Exception('response is missing required error field')
        if 'result' not in response:
            raise Exception('response is missing required result field')
        if response['error'] == "Model name already exists":
            print("Note type model already existing.")
        if response['error'] is not None:
            raise Exception(response['error'])
        return response['result']

    def to_anki(self, deckname="Autocards_export", tags=[""]):
        "Export cards to anki using anki-connect addon"
        df = self.pandas_df()
        df["generation_order"] = [str(int(x)+1) for x in list(df.index)]
        columns = df.columns.tolist()
        columns.remove("combined_columns")
        tags.append(f"Autocards::{self.title.replace(' ', '_')}")
        with suppress(ValueError):
            tags.remove("")

        # model formatting
        note_list = []
        for entry in df.index:
            note_list.append({"deckName": deckname,
                              "modelName": "Autocards",
                              "tags": tags,
                              "fields": df.loc[entry, :].to_dict()
                              })

        template_content = [{"Front": "",
                             "Back": ""}]

        # send new card type to anki
        try:
            self._ankiconnect_invoke(action="createModel",
                                     modelName="Autocards",
                                     inOrderFields=[
                                         "combined_columns"] + columns,
                                     cardTemplates=template_content)
        except Exception as e:
            print(f"{e}")

        # create new deck
        self._ankiconnect_invoke(action="createDeck", deck=deckname)

        # send notes to anki
        out = self._ankiconnect_invoke(action="addNotes", notes=note_list)

        if None in out:
            print(f"{len(note_list) - len(list(set(out)))} cards were not \
sent correctly.")
        if list(set(out)) != [None]:
            print("Cards sent to anki collection.\nYou can now open anki and use \
'change note type' to export the fields you need to your prefered notetype.")
            return out
        else:
            print("An error happened: no cards were successfuly sent to anki.")
            return out
