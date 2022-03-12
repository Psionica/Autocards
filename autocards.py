import json
import os
import re
import time
import urllib.request
from contextlib import suppress
from pathlib import Path
from pprint import pprint

import pandas as pd
import requests
from bs4 import BeautifulSoup
from epub_conversion.utils import open_book, convert_epub_to_lines
from tika import parser
from tqdm import tqdm
from torch.cuda import empty_cache

from pipelines import question_generation_pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "true"





class Autocards:
    """
    Main class used to create flashcards from text. The variable notetype refers to the type
    of flashcard that must be created: either cloze, basic or both. The
    variable wtm allow specifying whether you want to remove the mention of
    Autocards in your cards.
    """

    def __init__(self,
                 store_original_paragraph_in_output=True,
                 original_content_language="en",
                 output_language="en",
                 cloze_type="anki",
                 model="valhalla/distilt5-qa-qg-hl-6-4",
                 ans_model="valhalla/distilt5-qa-qg-hl-6-4"):
        self.cloze_type = cloze_type
        self.original_content_language = original_content_language
        self.output_language = output_language
        self.original_content_translator = None
        self.output_translator = None
        self.title = None
        print("Loading backend, this can take some time...")
        self.store_original_paragraph_in_output = store_original_paragraph_in_output
        self.model = model
        self.ans_model = ans_model

        self._validate_if_language_code_has_two_letter()
        self.software_cloze_type = cloze_type

        self.question_generation_pipeline = question_generation_pipeline('question-generation',
                                                                         model=model,
                                                                         ans_model=ans_model)
        self.question_answering_dictionary_list = []

        self._validate_cloze_type_input()

    def _validate_if_language_code_has_two_letter(self):
        if len(self.output_language) != 2 or len(self.original_content_language) not in [2, 3]:
            print("Output and input language has to be a two letter code like 'en' or 'fr'")
            raise SystemExit()

    def _validate_cloze_type_input(self):
        if self.cloze_type not in ["anki", "SM"]:
            print("Invalid cloze type, must be either 'anki' or \
'SM'")
            raise SystemExit()

    def load_translation_model_for_output_content_language(self):
        if self.output_language != "en":
            print("The flashcards will be automatically translated after being \
created. This can result in lower quality cards. Expect lowest quality cards \
than usual.")
            try:
                print("Loading output translation model...")
                from transformers import pipeline
                self.output_translator = pipeline(f"translation_en_to_{self.output_language}",
                                                  model=f"Helsinki-NLP/opus-mt-en-{self.output_language}")
            except Exception as e:
                print(f"Was not able to load translation pipeline: {e}")
                print("Resetting output language to english.")
                self.output_language = "en"

    def load_translation_model_for_original_content_language(self):
        if self.original_content_language != "en":
            print("The document will automatically be translated before creating flashcards. Expect lower quality "
                  "cards than usual.")
            try:
                print("Loading input translation model...")
                from transformers import pipeline
                self.original_content_translator = pipeline(f"translation_{self.original_content_language}_to_en",
                                                             model=f"Helsinki-NLP/opus-mt-{self.original_content_language}-en")
            except Exception as e:
                print(f"Was not able to load translation pipeline: {e}")
                print("Resetting input language to english.")
                self.original_content_language = "en"

    def _call_question_generation_module(self, text, title):
        """
        Call question generation module, then turn the answer into a
        dictionary containing metadata (clozed formatting, creation time,
        title, source text)
        """
        cloze_notes_list = []
        basic_notes_list = []
        original_text = ""
        original_content_language_is_not_english = self.original_content_language != "en"
        if original_content_language_is_not_english:
            original_text = str(text)
            text = self.original_content_translator(text)[0]["translation_text"]

        try:
            question_answering_list_from_text = self.question_generation_pipeline(text)
            empty_cache()
            cloze_notes_list = []
            basic_notes_list = []
            for question_answer in question_answering_list_from_text:
                if question_answer["note_type"] == "cloze":
                    cloze_notes_list.append(question_answer)
                if question_answer["note_type"] == "basic":
                    basic_notes_list.append(question_answer)

        except IndexError:
            tqdm.write(f"\nSkipping section because no cards \
could be made from that text: '{text}'")

        current_time = time.asctime()
        stored_text = ""
        stored_text_orig = ""
        
        if self.store_original_paragraph_in_output:
            stored_text = text
            stored_text_orig = original_text

        self.add_extracted_basic_notes_to_list(basic_notes_list)
        self.add_extracted_cloze_notes_to_list(cloze_notes_list)

        # merging cloze of the same text as a single question_answer with several cloze:
        self.merge_cloze_as_single_question_answer_with_several_cloze(current_time, stored_text, stored_text_orig,
                                                                      title, basic_notes_list, cloze_notes_list)

        tqdm.write(f"Number of question generated so far: {len(self.question_answering_dictionary_list)}")

    def merge_cloze_as_single_question_answer_with_several_cloze(self, current_time, stored_text, stored_text_orig,
                                                                 title, basic_notes_list, cloze_notes_list):
        if cloze_notes_list:
            for iterator in range(0, len(cloze_notes_list) - 1):
                if self.cloze_type == "SM":
                    tqdm.write("SM cloze not yet implemented, luckily \
SuperMemo supports importing from anki format. Hence the anki format will \
be used for your input.")
                    self.cloze_type = "anki"

                if self.cloze_type == "anki" and len(self.question_answering_dictionary_list) != iterator:
                    cl1 = re.sub(r"{{c\d+::|}}|\s", "",
                                 cloze_notes_list[iterator]["cloze"])
                    cl2 = re.sub(r"{{c\d+::|}}|\s", "",
                                 cloze_notes_list[iterator + 1]["cloze"])
                    if cl1 == cl2:
                        match = re.findall(r"{{c\d+::(.*?)}}",
                                           cloze_notes_list[iterator]["cloze"])
                        match.extend(re.findall(r"{{c\d+::(.*?)}}",
                                                cloze_notes_list[iterator + 1]["cloze"]))
                        clean_cloze = re.sub(r"{{c\d+::|}}", "",
                                             cloze_notes_list[iterator]["cloze"])
                        if "" in match:
                            match.remove("")
                        match = list(set(match))
                        for cloze_number, q in enumerate(match):
                            q = q.strip()
                            new_q = "{{c" + str(cloze_number + 1) + "::" + \
                                    q + "}}"
                            clean_cloze = clean_cloze.replace(q, new_q)
                        clean_cloze = clean_cloze.strip()

                        cloze_notes_list[iterator]['cloze'] = clean_cloze + "___TO_REMOVE___"
                        cloze_notes_list[iterator + 1]['cloze'] = clean_cloze
        to_add_full = cloze_notes_list + basic_notes_list
        for qa in to_add_full:
            qa["date"] = current_time
            qa["source_title"] = title
            qa["source_text"] = stored_text
            qa["source_text_orig"] = stored_text_orig
            if qa["note_type"] == "basic":
                self.question_answering_dictionary_list.append(qa)
            elif not qa["cloze"].endswith("___TO_REMOVE___"):
                self.question_answering_dictionary_list.append(qa)

    def add_extracted_cloze_notes_to_list(self, to_add_cloze_notes):
        if to_add_cloze_notes:
            for iterator in range(0, len(to_add_cloze_notes)):
                if to_add_cloze_notes[iterator]["note_type"] == "cloze":  # cloze formatting
                    if self.output_language != "en":
                        to_add_cloze_notes[iterator]["cloze_orig"] = to_add_cloze_notes[iterator]["cloze"]
                        original_clozed_notes_string = to_add_cloze_notes[iterator]["cloze_orig"]
                        original_clozed_notes_string = original_clozed_notes_string.replace("generate question: ", "")
                        original_clozed_notes_string = original_clozed_notes_string.replace("<hl> ", "{{c1::", 1)
                        original_clozed_notes_string = original_clozed_notes_string.replace(" <hl>", "}}", 1)
                        original_clozed_notes_string = original_clozed_notes_string.replace(" </s>", "")
                        original_clozed_notes_string = original_clozed_notes_string.strip()
                        to_add_cloze_notes[iterator]["cloze_orig"] = original_clozed_notes_string

                        clozed_notes_string = to_add_cloze_notes[iterator]["cloze"]
                        clozed_notes_string = clozed_notes_string.replace("generate question: ", "")
                        clozed_notes_string = clozed_notes_string.replace("\"", "'")
                        clozed_notes_string = clozed_notes_string.replace("<hl> ", "\"").replace(" <hl>", "\"")
                        clozed_notes_string = clozed_notes_string.replace(" </s>", "")
                        clozed_notes_string = clozed_notes_string.strip()
                        clozed_notes_string = self.output_translator(clozed_notes_string)[0]["translation_text"]
                        clozed_notes_string = clozed_notes_string.replace("\"", "{{c1::", 1)
                        clozed_notes_string = clozed_notes_string.replace("\"", "}}", 1)
                        to_add_cloze_notes[iterator]["cloze"] = clozed_notes_string
                    else:
                        to_add_cloze_notes[iterator]["cloze_orig"] = ""

                        clozed_notes_string = to_add_cloze_notes[iterator]["cloze"]
                        clozed_notes_string = clozed_notes_string.replace("generate question: ", "")
                        clozed_notes_string = clozed_notes_string.replace("<hl> ", "{{c1::", 1)
                        clozed_notes_string = clozed_notes_string.replace(" <hl>", "}}", 1)
                        clozed_notes_string = clozed_notes_string.replace(" </s>", "")
                        clozed_notes_string = clozed_notes_string.strip()
                        to_add_cloze_notes[iterator]["cloze"] = clozed_notes_string

                    to_add_cloze_notes[iterator]["basic_in_clozed_format"] = ""

    def add_extracted_basic_notes_to_list(self, to_add_basic_notes):
        if to_add_basic_notes:
            for iterator in range(0, len(to_add_basic_notes)):
                if to_add_basic_notes[iterator]["note_type"] == "basic":
                    if self.output_language != "en":
                        to_add_basic_notes[iterator]["question_orig"] = to_add_basic_notes[iterator]["question"]
                        to_add_basic_notes[iterator]["answer_orig"] = to_add_basic_notes[iterator]["answer"]
                        to_add_basic_notes[iterator]["question"] = \
                        self.output_translator(to_add_basic_notes[iterator]["question"])[0]["translation_text"]
                        to_add_basic_notes[iterator]["answer"] = \
                        self.output_translator(to_add_basic_notes[iterator]["answer"])[0]["translation_text"]
                    else:
                        to_add_basic_notes[iterator]["answer_orig"] = ""
                        to_add_basic_notes[iterator]["question_orig"] = ""

                    clozed_fmt = to_add_basic_notes[iterator]['question'] + "<br>{{c1::" \
                                 + to_add_basic_notes[iterator]['answer'] + "}}"
                    to_add_basic_notes[iterator]["basic_in_clozed_format"] = clozed_fmt

    def _sanitize_text(self, text):
        """correct common errors in text"""
        text = text.strip()
        # occurs sometimes in epubs apparently:
        text = text.replace("\xa0", " ")
        # wikipedia style citation:
        text = re.sub(r"\[\d*\]", "", text)
        return text

    def text_to_question_answering_pairs(self, text, title="Title",
                                         process_text_per_paragraph=False):
        """Take text as input and create qa pairs"""
        text = text.replace('\xad ', '')
        text = text.strip()
        self.title = title

        if process_text_per_paragraph:
            print("Consuming text by paragraph:")
            for paragraph in tqdm(text.split('\n\n'),
                                  desc="Processing by paragraph",
                                  unit="paragraph"):
                paragraph = paragraph.replace("\n", " ")
                self._call_question_generation_module(paragraph, title)
        else:
            print("Consuming text:")
            text = re.sub(r"\n\n*", ". ", text)
            text = re.sub(r"\.\.*", ".", text)
            text = self._sanitize_text(text)
            self._call_question_generation_module(text, title)

    def convert_user_input_into_question_answering_pairs(self, title="untitled user input"):
        """Take user input and create qa pairs"""
        user_input = input("Enter your text below then press Enter (press\
 enter twice to validate input):\n>")
        user_input = user_input.strip()

        print("\nFeeding your text to Autocards...")
        user_input = self._sanitize_text(user_input)
        self.text_to_question_answering_pairs(user_input, title, process_text_per_paragraph=False)
        print("Done feeding text.")

    def convert_pdf_into_question_answering_pairs(self, pdf_path, per_paragraph=True):
        """Take pdf pdf_file as input and create qa pairs"""
        if not Path(pdf_path).exists():
            print(f"PDF pdf_file not found at {pdf_path}!")
            return None

        print("Warning: pdf parsing is usually of poor quality because \
there are no good cross platform libraries. Consider using convert_text_file_into_question_answering_pairs() \
after preprocessing the text yourself.")
        title = pdf_path.replace("\\", "").split("/")[-1]
        raw = str(parser.from_file(pdf_path))
        safe_text = raw.encode('utf-8', errors='ignore')
        safe_text = str(safe_text).replace("\\sentence_list_count", "\n").replace("\\t", " ").replace("\\", "")

        text = self._sanitize_text(safe_text)

        self.text_to_question_answering_pairs(text, title, per_paragraph)

    def convert_text_file_into_question_answering_pairs(self, filepath, per_paragraph=True):
        """Take text pdf_file as input and create qa pairs"""
        file_dont_exists = not Path(filepath).exists()
        if file_dont_exists:
            print(f"File not found at {filepath}")
        text = open(filepath).read()
        text = self._sanitize_text(text)
        filename = str(filepath).split("/")[-1]
        if per_paragraph is False and len(text) > 300:
            question = input("The text is more than 300 characters long, \
are you sure you don't want to try to split the text by paragraph?\sentence_list_count(y/sentence_list_count)>")
            if question != "sentence_list_count":
                per_paragraph = True
        self.text_to_question_answering_pairs(text,
                                              filename,
                                              process_text_per_paragraph=per_paragraph)

    def convert_epub_into_question_answering_pairs(self, filepath, title="untitled epub pdf_file"):
        """Take an epub pdf_file as input and create qa pairs"""
        book = open_book(filepath)
        text = " ".join(convert_epub_to_lines(book))
        text = re.sub("<.*?>", "", text)
        text = text.replace("&nbsp;", " ")
        text = text.replace("&dash;", "-")
        text = re.sub("&.*?;", " ", text)
        # make paragraph limitation as expected in self.text_to_question_answering_pairs:
        text = text.replace("\r", "\n\n")
        text = re.sub("\n\n\n*", "\n\n", text)
        text = self._sanitize_text(text)
        self.text_to_question_answering_pairs(text, title, process_text_per_paragraph=True)

    def convert_html_web_page_into_question_answering_pairs(self, source, mode="url", element="p"):
        """Take html pdf_file (local or via url) and create qa pairs"""
        if mode == "local":
            soup = BeautifulSoup(open(source), 'xml')
        elif mode == "url":
            res = requests.get(source, timeout=15)
            html = res.content
            soup = BeautifulSoup(html, 'xml')
        else:
            return "invalid arguments"
        try:
            el = soup.article.body.find_all(element)
        except AttributeError:
            print("Using fallback method to extract page content")
            el = soup.find_all(element)

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
 the 'inspect' functionality in your favorite browser.")
            return None

        for section in tqdm(valid_sections,
                            desc="Processing by section",
                            unit="section"):
            section = self._sanitize_text(section)
            self._call_question_generation_module(section, title)

    def clear_question_answering_pairs(self):
        """Delete currently stored qa pairs"""
        self.question_answering_dictionary_list = []

    def return_question_answering_pairs(self, prefix='', jeopardy=False):
        """Return qa pairs to the user"""
        global string
        if prefix != "" and prefix[-1] != ' ':
            prefix += ' '
        if len(self.question_answering_dictionary_list) == 0:
            print("No qa generated yet!")
            return None

        response = []
        for qa_pair in self.question_answering_dictionary_list:
            if qa_pair['note_type'] == "basic":
                if jeopardy:
                    string = f"\"{prefix}{qa_pair['answer']}\",\" {qa_pair['question']}\""
                else:
                    string = f"\"{prefix}{qa_pair['question']}\",\" {qa_pair['answer']}\""
            elif qa_pair['note_type'] == "cloze":
                string = f"\"{prefix}{qa_pair['cloze']}\""
            response.append(string)
        return response

    def print(self, *args, **kwargs):
        """Print qa pairs to the user"""
        print(self.return_question_answering_pairs(*args, **kwargs))

    def pprint(self, *args, **kwargs):
        """Prettyprint qa pairs to the user"""
        pprint(self.return_question_answering_pairs(*args, **kwargs))

    def _combine_df_columns(self, row, col_names):
        combined = "".join(
            f"{col.upper()}: {dict(row)[col]}<br>\n" for col in col_names
        )

        return "#" * 15 + "Combined columns:<br>\n" + combined + "#" * 15

    def pandas_df(self):
        if len(self.question_answering_dictionary_list) == 0:
            print("No qa generated yet!")
            return None
        "Output a Pandas DataFrame containing qa pairs and metadata"
        df = pd.DataFrame(columns=list(self.question_answering_dictionary_list[0].keys()))
        for qa in self.question_answering_dictionary_list:
            df = df.append(qa, ignore_index=True)
        for i in df.index:
            for c in df.columns:
                if pd.isna(df.loc[i, c]):
                    # otherwise, export functions break:
                    df.loc[i, c] = ""
        if self.original_content_language == "en":
            df = df.drop(columns=["source_text_orig"], axis=1)
        if self.output_language == "en":
            df = df.drop(columns=["cloze_orig", "question_orig", "answer_orig"],
                         axis=1)
        df["combined_columns"] = [self._combine_df_columns(df.loc[x, :], df.columns)
                                  for x in df.index]
        return df

    def to_csv(self, filename="Autocards_export.csv"):
        """Export qa pairs as csv pdf_file"""
        if len(self.question_answering_dictionary_list) == 0:
            print("No qa generated yet!")
            return None

        df = self.pandas_df()

        for i in df.index:
            for c in df.columns:
                df.loc[i, c] = str(df.loc[i, c]).replace(",", r"\,")

        if ".csv" in filename:
            filename = filename.replace(".csv", "")
        df[df["note_type"] == "cloze"].to_csv(f"{filename}_cloze.csv")
        df[df["note_type"] != "cloze"].to_csv(f"{filename}_basic.csv")
        print(f"Done writing qa pairs to {filename}_cloze.csv and {filename}_basic.csv")

    def to_json(self, filename="Autocards_export.json"):
        """Export qa pairs as json pdf_file"""
        if len(self.question_answering_dictionary_list) == 0:
            print("No qa generated yet!")
            return None

        df = self.pandas_df()

        if ".json" in filename:
            filename = filename.replace(".json", "")
        df[df["note_type"] == "cloze"].to_json(f"{filename}_cloze.json")
        df[df["note_type"] != "cloze"].to_json(f"{filename}_basic.json")
        print(f"Done writing qa pairs to {filename}_cloze.json and \
{filename}_basic.json")

    def _ankiconnect_invoke(self, action, **params):
        """send requests to ankiconnect addon"""

        def request_wrapper(action, **params):
            return {'action': action, 'params': params, 'version': 6}

        request_json = json.dumps(request_wrapper(action, **params)
                                 ).encode('utf-8')
        try:
            response = json.load(urllib.request.urlopen(
                urllib.request.Request(
                    'http://localhost:8765',
                    request_json)))
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
        """Export cards to anki using anki-connect addon"""
        df = self.pandas_df()
        df["generation_order"] = [str(int(x) + 1) for x in list(df.index)]
        columns = df.columns.tolist()
        columns.remove("combined_columns")
        tags.append(f"Autocards::{self.title.replace(' ', '_')}")
        with suppress(ValueError):
            tags.remove("")

        # model formatting
        note_list = [{"deckName": deckname,
                      "modelName": "Autocards",
                      "tags": tags,
                      "fields": df.loc[entry, :].to_dict()
                      } for entry in df.index]
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
'change note type' to export the fields you need to your preferred notetype.")
        else:
            print("An error happened: no cards were successfuly sent to anki.")

        return out
