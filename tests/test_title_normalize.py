from jobproc.title_normalize import normalize_title


def test_strips_seniority():
    assert normalize_title("Senior Data Scientist") == "Data Scientist"
    assert normalize_title("Staff AI Engineer") == "AI Engineer"
    assert normalize_title("Principal UX Designer") == "UX Designer"


def test_strips_at_suffix():
    assert normalize_title("Analyst @ Second Dinner") == "Analyst"


def test_strips_contract_suffix():
    assert normalize_title("Product Manager - Remote") == "Product Manager"
    assert normalize_title("Data Scientist - Contract") == "Data Scientist"


def test_strips_trailing_level():
    assert normalize_title("Software Engineer II") == "Software Engineer"
    assert normalize_title("Analyst III") == "Analyst"


def test_preserves_acronyms():
    assert normalize_title("ux researcher") == "UX Researcher"
    assert normalize_title("ai engineer") == "AI Engineer"
    assert normalize_title("ml platform engineer") == "ML Platform Engineer"


def test_rejects_director():
    assert normalize_title("Director of Engineering") is None
    assert normalize_title("VP Product") is None
    assert normalize_title("Vice President of Sales") is None


def test_rejects_combo():
    assert normalize_title("Designer / Developer") is None
    assert normalize_title("Sales & Marketing Manager") is None


def test_rejects_intern_and_placeholders():
    assert normalize_title("Data Science Intern") is None
    assert normalize_title("Recruiter") is None


def test_events_to_event():
    assert normalize_title("Events Manager") == "Event Manager"


def test_short_or_empty():
    assert normalize_title("") is None
    assert normalize_title("  ") is None
    assert normalize_title("X") is None
