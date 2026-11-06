from src.evaluate import anls, normalize


def test_exact():
    assert anls("HELLO", ["hello"]) == 1.0


def test_multiple_refs():
    assert anls("abc", ["xyz", "abc"]) == 1.0


def test_threshold():
    assert anls("hello", ["zzzzz"]) == 0.0


def test_normalize():
    assert normalize("  Hello   WORLD  ") == "hello world"
