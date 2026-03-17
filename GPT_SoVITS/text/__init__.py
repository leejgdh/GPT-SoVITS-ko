from text import symbols2

_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols2.symbols)}


def cleaned_text_to_sequence(cleaned_text, version=None):
    """텍스트 문자열을 심볼 ID 시퀀스로 변환한다."""
    phones = [_symbol_to_id_v2[symbol] for symbol in cleaned_text]
    return phones
