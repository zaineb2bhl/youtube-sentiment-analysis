"""Nettoyage de texte partagé entre l'entraînement et l'API.

Utiliser la même fonction des deux côtés évite le « training/serving skew ».
"""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"[@#]\w+")
HTML_RE = re.compile(r"<[^>]+>|&\w+;")
# On garde lettres (y compris accentuées), chiffres, apostrophes et quelques ponctuations utiles (! ?)
SPECIAL_RE = re.compile(r"[^\w\s'!?]|_")
REPEAT_RE = re.compile(r"(.)\1{2,}")  # "soooo" -> "soo"
SPACES_RE = re.compile(r"\s+")

# Emojis/émoticônes fréquents convertis en mots porteurs de sentiment
EMOJI_MAP = {
    "😀": " happy ", "😃": " happy ", "😄": " happy ", "😁": " happy ", "😊": " happy ",
    "🙂": " happy ", "😍": " love ", "🥰": " love ", "❤️": " love ", "❤": " love ",
    "👍": " good ", "🔥": " awesome ", "😂": " funny ", "🤣": " funny ", "👏": " great ",
    "😢": " sad ", "😭": " sad ", "😞": " sad ", "☹️": " sad ", "🙁": " sad ",
    "😡": " angry ", "😠": " angry ", "🤬": " angry ", "👎": " bad ", "💩": " bad ",
    ":)": " happy ", ":-)": " happy ", ":d": " happy ", ":(": " sad ", ":-(": " sad ",
}


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for emo, word in EMOJI_MAP.items():
        text = text.replace(emo, word)
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = SPECIAL_RE.sub(" ", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    return SPACES_RE.sub(" ", text).strip()
