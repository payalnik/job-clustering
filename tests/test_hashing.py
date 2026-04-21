from jobproc.hashing import url_hash


def test_determinism():
    assert url_hash("https://example.com/a") == url_hash("https://example.com/a")


def test_distinct_urls_distinct_hashes():
    assert url_hash("https://example.com/a") != url_hash("https://example.com/b")


def test_fits_in_bigint():
    h = url_hash("https://example.com/a")
    assert 0 <= h < 2 ** 63
