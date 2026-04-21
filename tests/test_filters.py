from jobproc.filters import (
    clean_ats_url, clean_description, clean_location,
    is_aggregator_listing, is_garbage_description, is_garbage_url,
)


def test_aggregator_titles():
    assert is_aggregator_listing("139 senior event manager jobs in Mountain View")
    assert is_aggregator_listing("Find Software Engineer Jobs in SF")
    assert is_aggregator_listing("Careers Home")
    assert is_aggregator_listing("")
    assert is_aggregator_listing("unknown")


def test_real_titles_pass():
    assert not is_aggregator_listing("Senior Data Scientist")
    assert not is_aggregator_listing("HIRING NOW! Paintless Dent Repair Tech")
    assert not is_aggregator_listing("Data Scientist - Jobs - Careers at Apple")


def test_garbage_urls():
    assert is_garbage_url("https://careers.example.com/search?role=engineer")
    assert is_garbage_url("https://example.com/job-categories/engineering")
    # Lever board page (no posting UUID)
    assert is_garbage_url("https://jobs.lever.co/acme")


def test_real_urls_pass():
    assert not is_garbage_url("https://boards.greenhouse.io/acme/jobs/12345")
    assert not is_garbage_url("https://jobs.lever.co/acme/uuid-1234-5678")


def test_clean_ats_url_strips_tracking():
    assert clean_ats_url(
        "https://boards.greenhouse.io/acme/jobs/1?gh_jid=1&utm_source=x"
    ) == "https://boards.greenhouse.io/acme/jobs/1"
    assert clean_ats_url(
        "https://jobs.ashbyhq.com/acme/123/application?src=li"
    ) == "https://jobs.ashbyhq.com/acme/123"


def test_garbage_description():
    assert is_garbage_description("")
    assert is_garbage_description("too short")
    assert is_garbage_description("Please enable javascript to continue")


def test_clean_description_strips_html():
    assert clean_description("<p>Hello <b>world</b>&nbsp;!</p>") == "Hello world !"


def test_clean_location():
    assert clean_location("San Francisco, CA") == "San Francisco, CA"
    assert clean_location("2 Locations") == ""
    assert clean_location("Hybrid") == ""
    assert clean_location("") == ""
