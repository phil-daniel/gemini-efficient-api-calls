import logging
import requests
from requests.exceptions import RequestException, HTTPError

class BaseTextInput():
    content : str = ""

    def __init__(
        self,
        content : str
    ):
        self.content = content

class FileInput(BaseTextInput):

    def __init__(
        self,
        filepath : str,
        filetype : str = 'txt'
    ):
        self.content = ""
        try:
            if filetype == 'txt':
                with open(filepath, 'r', encoding='utf-8') as file:
                    self.content = file.read()
            else:
                logging.error('File reading has currently only been implemented for txt files.')
                raise NotImplementedError("Only text file reading has been implemented")
        except FileNotFoundError:
            logging.error(f"File {filepath} was not found")
            raise FileNotFoundError(f"File {filepath} was not found")
        except Exception as e:
            raise(e)

class WebsiteInput(BaseTextInput):

    def __init__(
        self,
        link : str
    ):
        # TODO: Error handling, website doesnt exist, retry error etc
        try:

            response = requests.get(link)
            response.raise_for_status()

            self.content = response.text
        except:
            pass