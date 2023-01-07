import en_core_web_lg
from spacy.tokens.token import Token

nlp = en_core_web_lg.load()

NEGATIONS = ['no', 'not', 'never']
KEEP_TOKEN_TYPE = 'keep'
REMOVE_TOKEN_TYPE = 'remove'
NEGATION_TOKEN_TYPE = 'negation'


def _get_token_type_and_output_string(token: Token, 
                                      should_ignore_twitter_usernames: bool,
                                      should_ignore_hashtags: bool):
    # Check if negation
    if token.lemma_.lower() in NEGATIONS:
        return (NEGATION_TOKEN_TYPE, 'NOT')

    # Check if hashtag
    if should_ignore_hashtags and token.text[0] == '#':
        return (KEEP_TOKEN_TYPE, token.text)

    # Check if punctuation
    if token.is_punct:
        return (REMOVE_TOKEN_TYPE, 'PUNCT')

    # Check if twitter username (starts with @)
    if should_ignore_twitter_usernames and token.text[0] == '@':
        return (KEEP_TOKEN_TYPE, token.text)

    # Check if number.
    if token.like_num or token.pos == 'NUM' \
        or token.ent_type_ in ['CARDINAL', 'ORDINAL']:
        return (KEEP_TOKEN_TYPE, 'NUMBER')

    # Check if stopword
    if token.is_stop:
        return (REMOVE_TOKEN_TYPE, 'STOP')

    # If this point is reached, the token is one we want to keep
    return (KEEP_TOKEN_TYPE, token.lemma_.lower())


def preprocess_data(raw_data, 
                    should_ignore_twitter_usernames: bool,
                    should_ignore_hashtags: bool):
    """Preprocess the given raw data and returns it as strings.

    This method applies the following transformations:
        - tokenization
        - POS tagging
        - removes stopwords (apart from no and not)
        - removes punctuation
        - clears hashtags (#something -> something) if should_ignore_hashtags is 
          false
        - lemmatisation
        - to lowercase
    """
    all_docs = nlp.pipe(raw_data)

    result = []
    for doc in all_docs:
        processed_tokens = []
        for token in doc:
            token_type, output_string = \
                _get_token_type_and_output_string(token, 
                                                  should_ignore_twitter_usernames=should_ignore_twitter_usernames,
                                                  should_ignore_hashtags=should_ignore_hashtags)
            if token_type != REMOVE_TOKEN_TYPE:
                processed_tokens.append(output_string)
        result.append(''.join(processed_tokens))
            
    return result 
