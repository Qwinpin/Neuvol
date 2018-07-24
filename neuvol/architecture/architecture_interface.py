from .individ_text import IndividText


def cradle(stage, data_type='text', task_type='classification', parents=None, freeze=None, **kwargs):
    """
    Factory method for different data types
    """
    if data_type == 'text':
        return IndividText(stage, task_type='classification', parents=None, freeze=None, **kwargs)
