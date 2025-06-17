Web Scraping with Playwright in Python
Scrape JavaScript‑heavy sites effortlessly with Playwright.
 (
youtube.com
)
Playwright offers:
JavaScript rendering
: Executes page scripts so you scrape only after content appears. (
playwright.dev
)
Headless & headed modes
: Run without UI or in a real browser for debugging. (
playwright.dev
)
Auto‑waiting & retry
: Built‑in locators reduce flakiness. (
playwright.dev
)
Multi‑browser support
: Chromium, Firefox, WebKit—all from one API. (
playwright.dev
)
Example: Scraping a JS‑Rendered Site
We’ll scrape 
Quotes to Scrape (JS)
—a site that loads quotes via JavaScript, so a simple 
requests
 call gets only an empty shell (
quotes.toscrape.com
). Playwright runs the scripts and gives us the real content:
# /// script


# dependencies = ["playwright"]


# ///



from
 playwright
.
sync_api 
import
 sync_playwright


def
 
scrape_quotes
(
)
:

    
with
 sync_playwright
(
)
 
as
 p
:

        
# Channel can be "chrome", "msedge", "chrome-beta", "msedge-beta" or "msedge-dev".

        browser 
=
 p
.
chromium
.
launch
(
headless
=
True
,
 channel
=
"chrome"
)

        page 
=
 browser
.
new_page
(
)

        page
.
goto
(
"https://quotes.toscrape.com/js/"
)

        quotes 
=
 page
.
query_selector_all
(
".quote"
)

        
for
 q 
in
 quotes
:

            text 
=
 q
.
query_selector
(
".text"
)
.
inner_text
(
)

            author 
=
 q
.
query_selector
(
".author"
)
.
inner_text
(
)

            
print
(
f"
{
text
}
 — 
{
author
}
"
)

        browser
.
close
(
)



if
 __name__ 
==
 
"__main__"
:

    scrape_quotes
(
)
Copy to clipboard
Error
Copied
Save as 
scraper.py
 and run:
uv run scraper.py
Copy to clipboard
Error
Copied
You’ll see each quote plus author printed—fetched only after the JS executes.














Previous




LLM Video Screen-Scraping












Next










Scheduled Scraping with GitHub Actions





