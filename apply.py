import asyncio, json, os, random, re, smtplib, urllib.parse
from email.mime.text import MIMEText
from pathlib import Path

import httpx
import openai
from dotenv import load_dotenv
from playwright.async_api import async_playwright, Browser

load_dotenv()

from main import PROVIDERS, _SKIP_ON, load_resume

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

# ── Humanizer skill ────────────────────────────────────────────────────────
# blader/humanizer is a Markdown prompt file, not a Python library.
# We download SKILL.md once and inject it as the LLM system prompt.
_SKILL_PATH = Path("humanizer/SKILL.md")
_skill_cache: str | None = None


def _skill() -> str:
    global _skill_cache
    if _skill_cache:
        return _skill_cache
    if not _SKILL_PATH.exists():
        print("  Downloading humanizer skill...")
        _SKILL_PATH.parent.mkdir(exist_ok=True)
        r = httpx.get(
            "https://raw.githubusercontent.com/blader/humanizer/main/SKILL.md",
            timeout=15,
            follow_redirects=True,
        )
        r.raise_for_status()
        _SKILL_PATH.write_text(r.text, encoding="utf-8")
    _skill_cache = _SKILL_PATH.read_text(encoding="utf-8")
    return _skill_cache


def humanize(instruction: str, max_tokens: int = 512) -> str:
    available = [(n, u, e, m) for n, u, e, m in PROVIDERS if os.environ.get(e)]
    for name, base_url, env_var, model in available:
        try:
            client = openai.OpenAI(base_url=base_url, api_key=os.environ[env_var])
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": _skill()},
                    {"role": "user", "content": instruction},
                ],
            )
            return resp.choices[0].message.content.strip()
        except _SKIP_ON:
            pass
        except openai.APIStatusError as e:
            if e.status_code not in (402, 429, 503):
                print(f"  humanize/{name} HTTP {e.status_code}")
        except Exception as e:
            print(f"  humanize/{name}: {e}")
    raise RuntimeError("All providers failed in humanize()")


# ── Mailmeteor email finder ────────────────────────────────────────────────
# No API key needed. Cloudflare Turnstile solves itself in a real Chromium
# instance. We intercept the backend API response via page.on("response").
# Using a fresh browser context per request avoids cookie-based rate limits.

async def _mm_call(browser: Browser, tool_url: str, api_path_fragment: str) -> dict:
    """
    Open tool_url in a fresh context, wait for the Mailmeteor backend response
    at api_path_fragment, and return the parsed JSON.
    """
    ctx = await browser.new_context(user_agent=_UA, locale="en-US")
    await ctx.add_init_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
    )
    page = await ctx.new_page()
    result: dict = {}
    done = asyncio.Event()

    async def on_response(resp):
        if api_path_fragment in resp.url:
            try:
                result.update(await resp.json())
            except Exception:
                pass
            done.set()

    page.on("response", on_response)
    try:
        await page.goto(tool_url, wait_until="domcontentloaded", timeout=30_000)
        # Turnstile solves in ~3-8s; give it 30s before giving up
        await asyncio.wait_for(done.wait(), timeout=30)
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        print(f"  mailmeteor: {e}")
    finally:
        await page.close()
        await ctx.close()
    return result


async def mm_by_name(browser: Browser, full_name: str, domain: str) -> tuple[str, str]:
    """Find email by person name + company domain. Returns (email, name)."""
    url = (
        "https://mailmeteor.com/tools/email-finder"
        f"?name={urllib.parse.quote(full_name)}"
        f"&domain={urllib.parse.quote(domain)}"
    )
    r = await _mm_call(browser, url, "email-finder/person")
    if r.get("found") and r.get("email"):
        return r["email"], full_name
    if r.get("error"):
        print(f"  mailmeteor by-name: {r.get('code','error')} — {r.get('message','')}")
    return "", ""


async def mm_by_linkedin(browser: Browser, linkedin_url: str) -> tuple[str, str]:
    """Find email by LinkedIn profile URL. Returns (email, full_name)."""
    url = (
        "https://mailmeteor.com/tools/linkedin-email-finder"
        f"?linkedin-url={urllib.parse.quote(linkedin_url)}"
    )
    r = await _mm_call(browser, url, "email-finder/linkedin")
    if r.get("found") and r.get("email"):
        return r["email"], r.get("full_name", "")
    if r.get("error"):
        print(f"  mailmeteor by-linkedin: {r.get('code','error')}")
    return "", ""


async def find_recruiter(browser: Browser, job: dict, domain: str) -> tuple[str, str]:
    """
    Best-effort recruiter email hunt. Priority:
    1. Recruiter name scraped from the job page  → Mailmeteor name finder
    2. Recruiter LinkedIn URL scraped from page  → Mailmeteor LinkedIn finder
    3. Generic HR titles as fallback             → Mailmeteor name finder
    Returns (email, name).
    """
    # Path 1 — name extracted during apply_to_job
    if job.get("recruiter_name"):
        email, name = await mm_by_name(browser, job["recruiter_name"], domain)
        if email:
            return email, name
        await asyncio.sleep(random.uniform(4, 7))

    # Path 2 — LinkedIn URL extracted during apply_to_job
    if job.get("recruiter_linkedin"):
        email, name = await mm_by_linkedin(browser, job["recruiter_linkedin"])
        if email:
            return email, name
        await asyncio.sleep(random.uniform(4, 7))

    # Path 3 — generic HR role titles as person name (low hit rate, last resort)
    for title in ("HR Manager", "Talent Acquisition", "Recruiting Manager"):
        email, name = await mm_by_name(browser, title, domain)
        if email:
            return email, name
        await asyncio.sleep(random.uniform(5, 8))

    return "", ""


# ── Cold email ─────────────────────────────────────────────────────────────

def send_cold_email(to_email: str, recruiter_name: str, job: dict):
    sender = os.environ["EMAIL_FROM"]
    sender_name = os.environ.get("EMAIL_NAME", sender)
    greeting = f"Hi {recruiter_name.split()[0]}," if recruiter_name else "Hi,"

    body = humanize(
        f"Write a cold email. Strict rules: under 60 words total, no bullet points, "
        f"plain text, no hollow phrases like 'hope this finds you well'.\n\n"
        f"Greeting: '{greeting}'\n"
        f"Context: I just applied for {job['title']} at {job['company']}.\n"
        f"Why I fit: {job['reason']}\n"
        f"Sign off as {sender_name}.\n"
        f"Mention this link once, naturally: {job['url']}"
    )

    msg = MIMEText(body)
    msg["Subject"] = f"Applied – {job['title']} @ {job['company']}"
    msg["From"] = f"{sender_name} <{sender}>"
    msg["To"] = to_email

    with smtplib.SMTP(
        os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        int(os.environ.get("SMTP_PORT", "587")),
    ) as s:
        s.starttls()
        s.login(sender, os.environ["EMAIL_PASSWORD"])
        s.send_message(msg)
    print(f"  Cold email → {recruiter_name or to_email}")


# ── Playwright helpers ─────────────────────────────────────────────────────

async def _click_first(page, selectors: list[str]) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.is_visible(timeout=2000):
                await loc.click()
                return True
        except Exception:
            pass
    return False


async def _human_type(page, locator, text: str):
    try:
        await locator.click()
        for ch in text:
            await page.keyboard.type(ch, delay=random.uniform(45, 130))
    except Exception:
        pass


# ── Recruiter info extraction ──────────────────────────────────────────────

async def extract_recruiter_info(page) -> tuple[str, str]:
    """
    Scrape recruiter name + LinkedIn /in/ URL from the current job page.
    Best results on LinkedIn listings. Returns (name, linkedin_url).
    """
    name, linkedin_url = "", ""

    for sel in (
        ".jobs-poster__name",
        ".hirer-card__hirer-information .app-aware-link",
        "[data-testid='job-poster-card'] strong",
        ".job-details-jobs-unified-top-card__job-poster-name",
    ):
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=1500):
                name = (await el.text_content() or "").strip()
                if name:
                    break
        except Exception:
            pass

    try:
        links = page.locator('a[href*="linkedin.com/in/"]')
        if await links.count() > 0:
            href = (await links.first.get_attribute("href")) or ""
            linkedin_url = href.split("?")[0]
    except Exception:
        pass

    return name, linkedin_url


# ── Job application ────────────────────────────────────────────────────────

async def apply_to_job(page, job: dict, resume_text: str, user_info: dict) -> bool:
    try:
        await page.goto(job["url"], wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(random.uniform(2, 4))

        # Grab recruiter info before navigating away from the listing
        rec_name, rec_linkedin = await extract_recruiter_info(page)
        if rec_name:
            job["recruiter_name"] = rec_name
        if rec_linkedin:
            job["recruiter_linkedin"] = rec_linkedin

        clicked = await _click_first(page, [
            'button:has-text("Easy Apply")',
            'button:has-text("Apply Now")', 'a:has-text("Apply Now")',
            'button:has-text("Apply")',     'a:has-text("Apply")',
            '[data-testid*="apply"]',       '.apply-btn', '#apply-button',
        ])
        if not clicked:
            print(f"  No apply button: {job['title']}")
            return False
        await asyncio.sleep(random.uniform(1.5, 3))

        # Fill free-text textarea fields (cover letter, "why us?", etc.)
        areas = page.locator("textarea")
        for i in range(await areas.count()):
            ta = areas.nth(i)
            hint = " ".join(filter(None, [
                await ta.get_attribute("placeholder") or "",
                await ta.get_attribute("aria-label")  or "",
                await ta.get_attribute("name")         or "",
            ])).lower().strip() or "describe your fit for this role"
            answer = humanize(
                f"Write a 2-sentence professional answer for this job application field: '{hint}'.\n"
                f"Resume excerpt: {resume_text[:1200]}\n"
                f"Job: {job['title']} at {job['company']}."
            )
            await _human_type(page, ta, answer)
            await asyncio.sleep(random.uniform(0.6, 1.5))

        # Standard single-line fields
        for sel, value in {
            'input[autocomplete*="given-name"], input[name*="first_name"], input[id*="first"]':
                user_info["first_name"],
            'input[autocomplete*="family-name"], input[name*="last_name"], input[id*="last"]':
                user_info["last_name"],
            'input[type="email"]':
                user_info["email"],
            'input[type="tel"], input[name*="phone"], input[id*="phone"]':
                user_info["phone"],
            'input[name*="linkedin"], input[placeholder*="LinkedIn"], input[id*="linkedin"]':
                user_info["linkedin"],
        }.items():
            if not value:
                continue
            try:
                loc = page.locator(sel).first
                if await loc.is_visible(timeout=1500):
                    await _human_type(page, loc, value)
                    await asyncio.sleep(random.uniform(0.3, 0.8))
            except Exception:
                pass

        submitted = await _click_first(page, [
            'button[type="submit"]',
            'button:has-text("Submit Application")',
            'button:has-text("Submit")',
            'input[type="submit"]',
        ])
        if submitted:
            await asyncio.sleep(2)
            print(f"  Applied: {job['title']} @ {job['company']}")
            return True

        print(f"  No submit button found: {job['title']}")
        return False

    except Exception as e:
        print(f"  Error ({job['title']}): {e}")
        return False


# ── Domain helper ──────────────────────────────────────────────────────────

_BOARDS = {"linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com", "monster.com"}


def _company_domain(company: str, job_url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", job_url)
    dom = m.group(1) if m else ""
    if any(b in dom for b in _BOARDS):
        return re.sub(r"[^a-z0-9]", "", company.lower()) + ".com"
    return dom


# ── Entry point ────────────────────────────────────────────────────────────

async def _run():
    results_path = Path("results.json")
    if not results_path.exists():
        raise FileNotFoundError("Run main.py first to generate results.json")

    jobs = json.loads(results_path.read_text())
    resume_text = ""
    try:
        resume_text = load_resume()
    except Exception:
        pass

    user_info = {
        "first_name": os.environ.get("USER_FIRST_NAME", ""),
        "last_name":  os.environ.get("USER_LAST_NAME",  ""),
        "email":      os.environ.get("EMAIL_FROM",      ""),
        "phone":      os.environ.get("USER_PHONE",      ""),
        "linkedin":   os.environ.get("USER_LINKEDIN",   ""),
    }

    email_ready = bool(os.environ.get("EMAIL_FROM") and os.environ.get("EMAIL_PASSWORD"))
    profile_dir = str(Path("browser_profile").absolute())

    async with async_playwright() as p:
        # Job application browser: persistent profile so you stay logged in to
        # LinkedIn/Indeed across runs. First run → log in manually, then re-run.
        job_ctx = await p.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            headless=False,
            slow_mo=60,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
            user_agent=_UA,
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        await job_ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        )

        # Separate browser for Mailmeteor: fresh context per search resets
        # session cookies so each lookup is treated as a new visitor.
        mm_browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            slow_mo=30,
        )

        # ponytail: cap at 10 per run; remove slice to process all ranked jobs
        for job in jobs[:10]:
            print(f"\n→ #{job['rank']} [{job['score']}/100]  {job['title']} @ {job['company']}")

            page = await job_ctx.new_page()
            applied = await apply_to_job(page, job, resume_text, user_info)
            await page.close()

            if applied and email_ready:
                dom = _company_domain(job["company"], job["url"])
                email, name = await find_recruiter(mm_browser, job, dom)
                if email:
                    try:
                        send_cold_email(email, name, job)
                    except Exception as e:
                        print(f"  Email send failed: {e}")
                else:
                    print(f"  No recruiter email found for {dom}")
            elif applied:
                print("  Skipping email (EMAIL_FROM / EMAIL_PASSWORD not set)")

            await asyncio.sleep(random.uniform(5, 10))

        await mm_browser.close()
        await job_ctx.close()


if __name__ == "__main__":
    asyncio.run(_run())
