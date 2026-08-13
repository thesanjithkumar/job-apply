import asyncio, json, os, random, re, smtplib, urllib.parse
from datetime import datetime
from email.mime.multipart import MIMEMultipart
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


# ── Report email ──────────────────────────────────────────────────────────

def send_report_email(applied: list[dict], not_applied: list[dict]):
    sender      = os.environ.get("EMAIL_FROM", "")
    password    = os.environ.get("EMAIL_PASSWORD", "")
    sender_name = os.environ.get("EMAIL_NAME", sender)
    if not sender or not password:
        print("  Skipping report email (EMAIL_FROM / EMAIL_PASSWORD not set)")
        return

    date_str = datetime.now().strftime("%B %d, %Y")
    total    = len(applied) + len(not_applied)

    def _job_card(j: dict, show_apply_btn: bool) -> str:
        score   = j.get("score", "—")
        reason  = j.get("reason", "")
        title   = j.get("title", "")
        company = j.get("company", "")
        loc     = j.get("location", "")
        url     = j.get("url", "#")
        rank    = j.get("rank", "—")
        btn = (
            f'<a href="{url}" style="display:inline-block;background:#2563eb;color:#fff;'
            f'padding:8px 18px;border-radius:5px;text-decoration:none;font-size:13px;'
            f'font-weight:600;margin-top:8px;">Apply Now →</a>'
        ) if show_apply_btn else (
            f'<span style="display:inline-block;background:#16a34a;color:#fff;'
            f'padding:5px 14px;border-radius:5px;font-size:12px;font-weight:600;">✓ Applied</span>'
        )
        return f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;
                    padding:16px;margin-bottom:12px;box-shadow:0 1px 3px #0001;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="background:#3b82f6;color:#fff;padding:2px 8px;border-radius:4px;
                         font-size:12px;font-weight:700;">#{rank}</span>
            <span style="background:#f1f5f9;color:#475569;padding:2px 8px;border-radius:4px;
                         font-size:12px;font-weight:700;">Score: {score}/100</span>
          </div>
          <div style="font-size:16px;font-weight:700;color:#1e293b;margin-bottom:2px;">{title}</div>
          <div style="font-size:13px;color:#64748b;margin-bottom:4px;">{company}
            {"&nbsp;·&nbsp;" + loc if loc else ""}</div>
          <div style="font-size:13px;color:#334155;font-style:italic;margin-bottom:8px;">{reason}</div>
          {btn}
        </div>"""

    applied_html = "".join(_job_card(j, False) for j in applied) or \
        '<p style="color:#64748b;">No applications were submitted this run.</p>'

    not_applied_html = "".join(_job_card(j, True) for j in not_applied) or \
        '<p style="color:#64748b;">All jobs were applied to successfully!</p>'

    html = f"""
    <html><body style="font-family:'Segoe UI',sans-serif;background:#f8fafc;margin:0;padding:0;">
    <div style="max-width:640px;margin:auto;">
      <div style="background:#0f172a;color:#fff;padding:28px 24px;border-radius:8px 8px 0 0;">
        <h1 style="margin:0;font-size:22px;">Daily Job Apply Report</h1>
        <p style="margin:4px 0 0;opacity:.7;font-size:14px;">{date_str}</p>
      </div>
      <div style="background:#fff;padding:20px 24px;border-left:4px solid #3b82f6;margin:0;">
        <div style="display:flex;gap:24px;">
          <div style="text-align:center;">
            <div style="font-size:32px;font-weight:800;color:#16a34a;">{len(applied)}</div>
            <div style="font-size:12px;color:#64748b;text-transform:uppercase;">Applied</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:32px;font-weight:800;color:#2563eb;">{len(not_applied)}</div>
            <div style="font-size:12px;color:#64748b;text-transform:uppercase;">Needs Manual Apply</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:32px;font-weight:800;color:#334155;">{total}</div>
            <div style="font-size:12px;color:#64748b;text-transform:uppercase;">Total Reviewed</div>
          </div>
        </div>
      </div>

      <div style="padding:20px 24px 8px;">
        <h2 style="font-size:16px;color:#16a34a;margin:0 0 12px;">✓ Successfully Applied ({len(applied)})</h2>
        {applied_html}
      </div>

      <div style="padding:8px 24px 24px;">
        <h2 style="font-size:16px;color:#2563eb;margin:0 0 12px;">
          📋 Apply Manually — No Apply Button Found ({len(not_applied)})</h2>
        {not_applied_html}
      </div>

      <div style="text-align:center;padding:16px;color:#94a3b8;font-size:11px;
                  border-top:1px solid #e2e8f0;">
        Generated by Job Apply Automation
      </div>
    </div>
    </body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Job Apply Report – {len(applied)} applied, {len(not_applied)} pending – {date_str}"
    msg["From"]    = f"{sender_name} <{sender}>"
    msg["To"]      = sender  # report goes to yourself

    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(
            os.environ.get("SMTP_HOST", "smtp.gmail.com"),
            int(os.environ.get("SMTP_PORT", "587")),
        ) as s:
            s.starttls()
            s.login(sender, password)
            s.send_message(msg)
        print(f"\n  Report email sent → {sender}")
        print(f"  Applied: {len(applied)}  |  Needs manual apply: {len(not_applied)}")
    except Exception as e:
        print(f"  Report email failed: {e}")


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

        # Inject LinkedIn session cookie if provided — avoids the login wall in CI.
        # Get it from: linkedin.com → F12 → Application → Cookies → li_at (Value)
        # Store as GitHub Secret: LINKEDIN_LI_AT
        li_at = os.environ.get("LINKEDIN_LI_AT", "")
        if li_at:
            await job_ctx.add_cookies([{
                "name":     "li_at",
                "value":    li_at,
                "domain":   ".linkedin.com",
                "path":     "/",
                "httpOnly": True,
                "secure":   True,
                "sameSite": "None",
            }])
            print("  LinkedIn cookie injected")
        else:
            print("  LINKEDIN_LI_AT not set — LinkedIn jobs will be skipped (add as GitHub Secret)")

        # Separate browser for Mailmeteor: fresh context per search resets
        # session cookies so each lookup is treated as a new visitor.
        mm_browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            slow_mo=30,
        )

        applied_jobs     = []
        not_applied_jobs = []

        eligible = [j for j in jobs[:100] if j.get("score", 0) >= 75]
        print(f"\n  {len(eligible)} jobs with score ≥ 75 (out of {len(jobs[:100])} ranked)")

        for job in eligible:
            print(f"\n→ #{job['rank']} [{job['score']}/100]  {job['title']} @ {job['company']}")

            # LinkedIn requires login — skip if no cookie was injected
            if "linkedin.com" in job.get("url", "") and not li_at:
                print(f"  LinkedIn job — no cookie set, skipping (will appear in report email)")
                not_applied_jobs.append(job)
                continue

            page = await job_ctx.new_page()
            success = await apply_to_job(page, job, resume_text, user_info)
            await page.close()

            if success:
                applied_jobs.append(job)
                if email_ready:
                    dom = _company_domain(job["company"], job["url"])
                    email, name = await find_recruiter(mm_browser, job, dom)
                    if email:
                        try:
                            send_cold_email(email, name, job)
                        except Exception as e:
                            print(f"  Email send failed: {e}")
                    else:
                        print(f"  No recruiter email found for {dom}")
                else:
                    print("  Skipping email (EMAIL_FROM / EMAIL_PASSWORD not set)")
            else:
                not_applied_jobs.append(job)

            await asyncio.sleep(random.uniform(5, 10))

        await mm_browser.close()
        await job_ctx.close()

    send_report_email(applied_jobs, not_applied_jobs)


if __name__ == "__main__":
    asyncio.run(_run())
