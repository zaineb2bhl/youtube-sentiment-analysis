from src.utils.text_cleaning import clean_text


def test_removes_urls_and_mentions():
    assert clean_text("Check https://youtu.be/abc @john #wow great") == "check great"


def test_lowercase_and_special_chars():
    assert clean_text("HELLO!!! *** World") == "hello!! world"


def test_emojis_mapped_to_words():
    assert "love" in clean_text("😍😍")
    assert "sad" in clean_text("so bad 😭")


def test_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(123) == ""
