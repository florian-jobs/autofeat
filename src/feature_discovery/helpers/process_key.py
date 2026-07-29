import unicodedata

def process_key(key: str) -> str:
    '''
    Processes the foreign key value by normalizing and removing special characters

    Parameters:
    ----------
    key: `str`
        Key to be processed

    Returns:
    -------
    `str`: Processed key
    '''
    if not isinstance(key, str):
        key = str(key)
    key = key.replace('.0', '')
    key = key.lower()
    key = key.replace(u'\xa0', u' ')
    key = key.replace(r'\s+', ' ')
    key = key.replace(' ', '')
    key = unicodedata.normalize('NFKD', key).encode('ascii', 'ignore').decode('utf-8')
    key = ''.join(char for char in key if char.isalnum())

    return key
