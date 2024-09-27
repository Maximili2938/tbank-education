# utils/helpers.py (вспомогательно) 
def validate_input(data):
    if 'question' not in data or 'context' not in data:
        return False
    return True
