from .text_processing import ProcessingText


def processing(data_type):
    if data_type == 'text':
        return ProcessingText
