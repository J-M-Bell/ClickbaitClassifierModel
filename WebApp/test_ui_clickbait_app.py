import re
from playwright.sync_api import Page, expect

def test_has_title(page: Page):
    """
    Tests that the page has the correct title.
    
    :param page: Description
    :type page: Page
    """
    page.goto("http://localhost:3000/")
    expect(page).to_have_title(re.compile("Clickbait Classifier"))

def test_get_started_link(page: Page):
    """
    Tests that page will display whether the title is clickbait or
    not clickbait depending on the title input from the user.
    """
    page.goto("http://localhost:3000/")
    
    # Locate the input field and fill it with text
    page.locator("input[name='title']").fill("Are You More Walter White Or Heisenberg")


     # Click the get started link.
    page.locator("button[type='submit']").click()
    locator = page.locator("p")

    # Expect a specific URL.
    expect(locator).to_have_text("This title is clickbait")
    # expect(page).to_have_url(re.compile(".*docs/intro"))