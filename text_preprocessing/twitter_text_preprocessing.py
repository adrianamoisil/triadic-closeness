from text_preprocessing.text_preprocessing import preprocess_data


def preprocess_twitter_data(raw_data: list) -> list:
    return preprocess_data(raw_data,
                           should_ignore_twitter_usernames=True,
                           should_ignore_hashtags=True)
