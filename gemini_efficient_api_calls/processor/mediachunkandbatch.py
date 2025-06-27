import math
import os
import ffmpeg

from .textchunkandbatch import TextChunkAndBatch
from ..input_handler.otherinputs import VideoFileInput

class MediaChunkAndBatch():
    
    def batch_by_number_of_questions(
        questions : list[str],
        questions_per_batch : int = 50
    ) -> list[list[str]]:
        """
        Easy access to the 'batch_by_number_of_questions' function provided by 'TextChunkAndBatch'.
        Groups a list of strings into multiple sublists, based on the maximum number of questions per batch.

        Args:
            - questions : The list of questions to be batched
            - questions_per_batch : The maximum number of questions that can be grouped together.
        
        Returns:
            - A list of a list of questions, where each sublist contains 'questions_per_batch' questions, other than the final
              sublist which may contain less.
        """
        return TextChunkAndBatch.batch_by_number_of_questions(questions, questions_per_batch)
    
    def chunk_sliding_window_by_duration(
        self,
        media_input : VideoFileInput,
        chunk_duration : int = 100,
        window_duration : int = 0,
    ) -> list[str]:
        """
        Chunks an inputted media files into multiple smaller files using the sliding window approach. 

        Args:
            - media_input: The content to be chunked, held in a VideoFileInput class
            - chunk_duration: The maximum duration of returned video chunks (in seconds).
            - window_duration: The duration of the chunk windows (in seconds). This is the overlap between consecutive chunks.
                                This is 0 by default.
        
        Returns:
            - A list of strings, where each string is the file path of a chunk of the inputted video. Each video is of duration
            'chunk_duration', except for the final video, which may be shorter.
        
        Raises:
            TODO
        """
        chunked_files = []
        chunk_count = math.ceil(MediaChunkAndBatch.get_video_duration(media_input.filepath) / (chunk_duration - window_duration))

        # TODO: Better method for creating temporary files?
        os.mkdir('./temp_output/')

        for i in range(chunk_count):
            chunk_start_pos = i * (chunk_duration - window_duration)
            MediaChunkAndBatch.trim_video(media_input.filepath, f'./temp_output/chunk_{i}.mp4', chunk_start_pos, chunk_duration)
            chunked_files.append(f'./temp_output/chunk_{i}.mp4')

        return chunked_files

    def get_video_duration(
        path : str
    ) -> float:
        """
        Returns the duration of the video stored at the inputted file path in seconds.

        Args:
            - path: The filepath of the video.
        
        Returns:
            - The duration of the video in seconds.
        
        Raises:
            TODO
        """
        # TODO: Error checking to ensure that the file path exists
        # TODO: Move this to each individual Input class.
        probe = ffmpeg.probe(path)
        duration = float(probe['format']['duration'])
        return duration
    
    def trim_video(
            in_path : str,
            out_path : str,
            start_time : float,
            duration : float
        ):
        """
        Returns the duration of the video stored at the inputted file path in seconds.

        Args:
            - in_path: The filepath of the video to be trimmed.
            - out_path: The filepath the trimmed video should be stored at.
            - start_time: The timestamp of the original video the trimmed video should start at (in seconds).
            - duration: The duration of the trimmed video (in seconds).
        
        Raises:
            TODO
        """
        # TODO: Move this to each individual Input class.
        # TODO: Error checking to ensure that the file path exists
        ffmpeg.input(in_path, ss=start_time).output(out_path, to=duration, c='copy').run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return