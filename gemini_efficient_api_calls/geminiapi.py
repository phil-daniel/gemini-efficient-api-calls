import json
import time
import logging
from typing import Any

from google import genai
from google.genai import types, errors

from .chunker import Chunker
from .mediachunker import MediaChunker

from .input_handler.textinputs import BaseTextInput
from .processor.textchunkandbatch import TextChunkAndBatch

DEFAULT_SYSTEM_PROMPT = """
    You are an AI assistant tasked with answering questions based on the information provided to you, with each answer being a **single** string in the JSON response.
    There should be the **exactly** same number of answers as inputted questions, no more, no less.
    * **Accuracy and Precision:** Provide direct, factual answers. **Do not** create or merge any of the questions.
    * **Source Constraint:** Use *only* information explicitly present in the transcript. Do not infer, speculate, or bring in outside knowledge.
    * **Completeness:** Ensure each answer fully addresses the question, *to the extent possible with the given transcript*.
    * **Missing Information:** If the information required to answer a question is not discussed or cannot be directly derived from the transcript, respond with "N/A".
"""

class Response:
    content : Any
    input_tokens : int
    output_tokens : int

    def __init__(
        self,
        content : Any,
        input_tokens : int,
        output_tokens : int
    ):
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

class GeminiApi:

    def __init__(
        self,
        api_key : str,
        model : str
    ):
        # Error handling - Api key not correct
        self.client = genai.Client(api_key=api_key)

        # Error handling - Model not correct, default to a model
        self.model = model
    
    def parse_json(
        self,
        to_parse : str,
    ):
        # TODO: Error handling - incorrect json formatting.
        parsed = json.loads(to_parse)
        return parsed

    def generate_content(
        self,
        prompt : str,
        file = None,
        system_prompt : str = None,
        max_retries : int = 5,
    ):
        # TODO: Check input and output tokens are below limits.
        # TODO: Improve retry if API failure occurs

        # Adding default system prompt if one is not given.
        if system_prompt == None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        for i in range(max_retries):
            if file:
                prompt = [prompt, file]
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[str],
                        system_instruction=system_prompt,
                    ),
                    contents= prompt
                )

                # TODO: Information about token usage, this can be used to compare performance between different designs
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count

                return Response(
                    content = self.parse_json(response.text),
                    input_tokens = input_tokens,
                    output_tokens = output_tokens
                )
            except errors.APIError as e:
                if e.code == 429:
                    # TODO: Is it possible to identify how long we have to way instead of just doing 10 seconds?
                    logging.info(f'Rate limit exceeded, waiting 20 seconds before retrying API call')
                    logging.debug(f'Gemini API Error Code: {e.code}\nGemini API Error Message: {e.message}')
                    time.sleep(20)
                    continue
                else:
                    logging.info(f'Unknown API Error occured, Error Code: {e.code}\nError Message: {e.message}')
            except Exception as e:
                logging.info(f'Unkown expection occured: {e}')
                logging.info("Retrying API call in 20 seconds.")
                time.sleep(20)
                continue
        
        # TODO: Handle failure better
        return Response(
            content = [],
            input_tokens = 0,
            output_tokens = 0
        )

    def generate_content_fixed(
        self,
        content : BaseTextInput,
        questions : list[str],
        chunk_char_length : int = 100000,
        questions_per_batch : int = 50,
        window_char_length : int = 100,
        system_prompt : str = None
    ):
        # TODO: Currently can only handle a text response i.e. not a code block.

        # # Chunking and Batching the questions
        # chunker = Chunker()
        # if enable_sliding_window:
        #     chunks = chunker.sliding_window_chunking_by_size(content, chunk_char_length, window_char_length)
        # else:
        #     chunks = chunker.fixed_chunking_by_size(content, chunk_char_length)
        # question_batches = chunker.fixed_question_batching(questions, questions_per_batch)

        chunks = TextChunkAndBatch.chunk_sliding_window_by_length(
            text_input = content,
            chunk_char_size = chunk_char_length,
            window_char_size = window_char_length
        )
        question_batches = TextChunkAndBatch.batch_by_number_of_questions(
            questions = questions, 
            questions_per_batch = questions_per_batch
        )

        answers = {}

        total_input_tokens = 0
        total_output_tokens = 0

        for batch in question_batches:
            for chunk in chunks:
                query_contents = f'Content:\n{chunk}\n\nThere are {len(batch)} questions. The questions are:\n' + '\n\t- '.join(batch)
                response = self.generate_content(query_contents, system_prompt=system_prompt)

                total_input_tokens += response.input_tokens
                total_output_tokens += response.output_tokens

                # TODO: If the question has already been answered in a previous chunk the new answer is disregarded, this can be
                # further optimised so the question is not asked again.
                for i in range(len(response.content)):
                    if batch[i] not in answers.keys() and response.content[i] !=  'N/A':
                        answers[batch[i]] = response.content[i]
        
        # TODO: Better way of returning? Tuple?
        return Response(
            content = answers,
            input_tokens = total_input_tokens,
            output_tokens = total_output_tokens
        )
    
    def generate_content_token_aware(
        self,
        content : BaseTextInput,
        questions : list[str],
        system_prompt : str = None
    ):
        # A version of generate_content_fixed() that automatically chunks depending on the token limits of the model being used.
        
        # Adding default system prompt if one is not given.
        if system_prompt == None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        model_info = self.client.models.get(model=self.model)
        input_token_limit = model_info.input_token_limit
        output_token_limit = model_info.output_token_limit

        total_input_tokens = 0
        total_output_tokens = 0

        answers = {}
        queue = [(content.content, questions)]

        while len(queue) > 0:
            curr_content, curr_questions = queue.pop(0)

            input_tokens_used = self.client.models.count_tokens(
                model=self.model, contents = [system_prompt, curr_content, curr_questions]
            )

            # Checking if the content is too large for the input token limit, if so splitting the content in half
            # TODO: Add ability to use sliding window
            if input_tokens_used > input_token_limit:
                chunked_content = [curr_content[0: len(curr_content)//2 + 1], curr_content[len(curr_content)//2 + 1 : len(curr_content)]]

                queue.append((chunked_content[0], curr_questions))
                queue.append((chunked_content[1], curr_questions))

            else:
                query_contents = f'Content:\n{curr_content}\n\nThere are {len(curr_questions)} questions. The questions are:\n' + '\n\t- '.join(curr_questions)
                response = self.generate_content(query_contents, system_prompt=system_prompt)

                # TODO: This doesn't seem to actually occur, need a better way of doing this, checking if the output limit has been reached
                if response.output_tokens > output_token_limit:
                    batched_questions = TextChunkAndBatch.batch_by_number_of_questions(curr_questions, len(curr_questions)//2 + 1)
                    queue.append((curr_content, batched_questions[0]))
                    queue.append((curr_content, batched_questions[1]))
                else:
                    for i in range(len(response.content)):
                        if curr_questions[i] not in answers.keys() and response.content[i] !=  'N/A':
                            answers[curr_questions[i]] = response.content[i]
                    total_input_tokens += response.input_tokens
                    total_output_tokens += response.output_tokens

        return Response(
            content = answers,
            input_tokens = total_input_tokens,
            output_tokens = total_output_tokens
        )

    def generate_content_semantic(
        self,
        content : BaseTextInput,
        questions : list[str],
        system_prompt : str = None
    ):
        content_chunks, question_batches = TextChunkAndBatch.chunk_and_batch_semantically(content, questions)

        total_input_tokens = 0
        total_output_tokens = 0
        answers = {}

        # Adding default system prompt if one is not given.
        if system_prompt == None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        for i in range(len(content_chunks)):
            # If there are no questions in the current chunk's batch, then we don't need to query it.
            if len(question_batches[i]) != 0:
                query_contents = f'Content:\n{content_chunks[i]}\n\nThere are {len(question_batches[i])} questions. The questions are:\n' + '\n\t- '.join(question_batches[i])
                response = self.generate_content(query_contents, system_prompt=system_prompt)

                total_input_tokens += response.input_tokens
                total_output_tokens += response.output_tokens
                
                for j in range(len(response.content)):
                    answers[question_batches[i][j]] = response.content[j]
        
        return Response(
            content = answers,
            input_tokens = total_input_tokens,
            output_tokens = total_output_tokens
        )

    def generate_content_media(
        self,
        media_path : str,
        questions : list[str],
        chunk_duration : int = 100,
        questions_per_batch : int = 50,
        window_duration : int = 0,
        system_prompt : str = None
    ):
        
        # Adding default system prompt if one is not given.
        if system_prompt == None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        chunker = MediaChunker()
        number_of_chunks = chunker.sliding_window_chunking_by_duration(media_path, chunk_duration, window_duration)

        question_batches = TextChunkAndBatch.batch_by_number_of_questions(questions, questions_per_batch)

        answers = {}

        total_input_tokens = 0
        total_output_tokens = 0

        for batch in question_batches:
            for i in range(number_of_chunks):
                query_contents = f'Content:\nThis has been attached as a media file, named {media_path}\n\nThere are {len(batch)} questions. The questions are:\n' + '\n\t- '.join(batch)
                
                file = self.client.files.upload(file=f'./temp_output/chunk_{i}.mp4')
                
                response = self.generate_content(query_contents, system_prompt=system_prompt, file=file)

                total_input_tokens += response.input_tokens
                total_output_tokens += response.output_tokens

                # TODO: If the question has already been answered in a previous chunk the new answer is disregarded, this can be
                # further optimised so the question is not asked again.
                for i in range(len(response.content)):
                    if batch[i] not in answers.keys() and response.content[i] !=  'N/A':
                        answers[batch[i]] = response.content[i]


        return Response(
            content = answers,
            input_tokens = total_input_tokens,
            output_tokens = total_output_tokens
        )
