import re
from playwright.sync_api import Page, expect

def test_has_title(page: Page):
    page.goto("https://playwright.dev/")
    expect(page).to_have_title(re.compile("Playwright"))

def test_get_started_link(page: Page):
    """Test clicking the 'Get started' link navigates to the Installation heading."""
    page.goto("https://playwright.dev/")
    
     # Click the get started link.
    page.get_by_role("link", name="Get started").click()

    # Expect a specific URL.
    expect(page).to_have_url(re.compile(".*docs/intro"))