import asyncio, colorsys, email.policy, html as _html, json, os, random, re, smtplib, traceback, urllib.parse
from datetime import datetime
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def _hdr(s: str) -> str:
    """Sanitize a string for use in an email header — removes non-ASCII."""
    return s.replace("\xa0", " ").replace("–", "-").replace("—", "-").encode("ascii", "replace").decode("ascii")

import db
from pathlib import Path

import httpx
import openai
from dotenv import load_dotenv
from playwright.async_api import async_playwright, Browser

load_dotenv()

from main import PROVIDERS, SEARCH_TERMS, _SKIP_ON, _role_match, _india_loc, load_resume

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

    body_safe = body.encode("ascii", "replace").decode("ascii")
    msg = MIMEText(body_safe, "plain", "utf-8")
    msg["Subject"] = _hdr(f"Applied - {job['title']} @ {job['company']}")
    msg["From"] = _hdr(f"{sender_name} <{sender}>")
    msg["To"] = to_email

    with smtplib.SMTP(
        os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        int(os.environ.get("SMTP_PORT", "587")),
    ) as s:
        s.starttls()
        s.login(sender, os.environ["EMAIL_PASSWORD"].replace("\xa0", "").replace(" ", "").strip())
        s.send_message(msg)
    print(f"  Cold email → {recruiter_name or to_email}")


# ── Himalayas Playwright scraper ───────────────────────────────────────────

async def scrape_himalayas_playwright(ctx, seen: set) -> list[dict]:
    """
    Scrape Himalayas job search using a real browser (Playwright).
    The public API ignores search terms; the website's frontend calls a real
    search endpoint — we intercept that response to get accurate results.
    Returns a list of job dicts ready to score and apply to.
    """
    import urllib.parse as _up
    collected: list[dict] = []

    async def _search_term(term: str):
        page = await ctx.new_page()
        caught: list[dict] = []
        done = asyncio.Event()

        async def on_response(resp):
            # Himalayas frontend calls its own /jobs/api (or similar) with real search
            if "himalayas.app" in resp.url and resp.status == 200:
                try:
                    data = await resp.json()
                    raw = data.get("jobs", [])
                    if raw:
                        caught.extend(raw)
                        done.set()
                except Exception:
                    pass

        page.on("response", on_response)
        url = f"https://himalayas.app/jobs?q={_up.quote(term)}"
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            # Wait for the API response to arrive (up to 15s)
            try:
                await asyncio.wait_for(done.wait(), timeout=15)
            except asyncio.TimeoutError:
                pass
        except Exception as e:
            print(f"  [Himalayas PW] '{term}' page error: {e}")
        finally:
            await page.close()

        for j in caught:
            title = j.get("title", "")
            if not _role_match(title):
                continue
            locs = j.get("locationRestrictions") or []
            if locs and not any(
                _india_loc(l) or l.lower() in ("worldwide", "anywhere", "global", "remote")
                for l in locs
            ):
                continue
            job_url = j.get("applicationLink", "")
            if not job_url or job_url in seen:
                continue
            seen.add(job_url)
            loc_str = ", ".join(locs) if locs else "Worldwide Remote"
            collected.append({
                "title": title,
                "company": j.get("companyName", ""),
                "location": loc_str,
                "url": job_url,
                "description": str(j.get("excerpt", "") or "")[:600],
                "source": "Himalayas",
                "date_posted": j.get("pubDate", ""),
                "score": 80,  # auto-qualify: targeted search already filtered by role
                "reason": f"ML/AI role at {j.get('companyName', 'company')} matching your search",
            })

    print("  [Himalayas] Playwright search...")
    for term in SEARCH_TERMS:
        await _search_term(term)
        await asyncio.sleep(random.uniform(2, 4))

    print(f"  [Himalayas] Found {len(collected)} matching jobs")
    return collected


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

        # Himalayas job pages link out to the company ATS — follow that link first
        if "himalayas.app" in job.get("url", ""):
            followed = await _click_first(page, [
                'a:has-text("Apply Now")',
                'a:has-text("Apply for this job")',
                'a:has-text("Apply for job")',
                'a[href*="lever.co"]',
                'a[href*="greenhouse.io"]',
                'a[href*="workable.com"]',
                'a[href*="ashbyhq.com"]',
                'a[href*="jobs."]',
                '[data-testid*="apply"] a',
            ])
            if not followed:
                print(f"  No external apply link on Himalayas page: {job['title']}")
                return False
            await asyncio.sleep(random.uniform(2, 3))

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

        async def _fill_current_step():
            # Textareas (cover letter, "why us?", additional questions)
            areas = page.locator("textarea:visible")
            for i in range(await areas.count()):
                ta = areas.nth(i)
                if await ta.input_value() != "":
                    continue
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

            # Standard single-line text/tel/email inputs
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
                    if await loc.is_visible(timeout=1500) and await loc.input_value() == "":
                        await _human_type(page, loc, value)
                        await asyncio.sleep(random.uniform(0.3, 0.8))
                except Exception:
                    pass

            # Number inputs (years of experience, salary expectations, etc.)
            num_inputs = page.locator('input[type="number"]:visible, input[type="text"][id*="year"]:visible')
            for i in range(await num_inputs.count()):
                ni = num_inputs.nth(i)
                try:
                    if await ni.is_visible(timeout=1000) and await ni.input_value() == "":
                        label = (await ni.get_attribute("aria-label") or
                                 await ni.get_attribute("placeholder") or "").lower()
                        # Salary → 0 (prefer not to say), years → 3, default → 3
                        value = "0" if "salary" in label or "ctc" in label or "compensation" in label else "3"
                        await ni.fill(value)
                        await asyncio.sleep(random.uniform(0.3, 0.6))
                except Exception:
                    pass

            # Select dropdowns — pick first non-placeholder option
            selects = page.locator("select:visible")
            for i in range(await selects.count()):
                sel_el = selects.nth(i)
                try:
                    if not await sel_el.is_visible(timeout=1000):
                        continue
                    opts = sel_el.locator("option")
                    count = await opts.count()
                    for j in range(count):
                        opt = opts.nth(j)
                        val = await opt.get_attribute("value") or ""
                        txt = (await opt.inner_text()).strip().lower()
                        if val and val not in ("", "select", "choose") and txt not in ("select", "choose", "please select", "-- select --"):
                            await sel_el.select_option(value=val)
                            break
                    await asyncio.sleep(random.uniform(0.2, 0.5))
                except Exception:
                    pass

            # Radio buttons — pick first option in each group (usually "Yes" / first choice)
            radio_groups: dict[str, object] = {}
            radios = page.locator('input[type="radio"]:visible')
            for i in range(await radios.count()):
                r = radios.nth(i)
                try:
                    name = await r.get_attribute("name") or str(i)
                    if name not in radio_groups:
                        radio_groups[name] = r
                except Exception:
                    pass
            for r in radio_groups.values():
                try:
                    if not await r.is_checked():
                        await r.check()
                        await asyncio.sleep(random.uniform(0.2, 0.5))
                except Exception:
                    pass

        # LinkedIn Easy Apply is multi-step: loop through Next → … → Submit
        for step in range(10):
            await _fill_current_step()
            await asyncio.sleep(random.uniform(0.8, 1.5))

            # Try submit first (last step)
            submitted = await _click_first(page, [
                'button:has-text("Submit application")',
                'button:has-text("Submit Application")',
                'button[aria-label*="Submit"]',
                'button[type="submit"]:has-text("Submit")',
                'input[type="submit"]',
            ])
            if submitted:
                await asyncio.sleep(2)
                print(f"  Applied: {job['title']} @ {job['company']}")
                return True

            # Advance to next step
            advanced = await _click_first(page, [
                'button:has-text("Next")',
                'button:has-text("Review")',
                'button:has-text("Continue")',
                'button[aria-label*="Next"]',
                'button[aria-label*="Review"]',
            ])
            if not advanced:
                break
            await asyncio.sleep(random.uniform(1.5, 2.5))

        print(f"  No submit button found: {job['title']}")
        return False

    except Exception as e:
        print(f"  Error ({job['title']}): {e}")
        return False


# ── Domain helper ──────────────────────────────────────────────────────────

_BOARDS = {"linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com", "monster.com", "himalayas.app"}


def _company_domain(company: str, job_url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", job_url)
    dom = m.group(1) if m else ""
    if any(b in dom for b in _BOARDS):
        return re.sub(r"[^a-z0-9]", "", company.lower()) + ".com"
    return dom


# ── Report email ──────────────────────────────────────────────────────────

def send_report_email(applied: list[dict], not_applied: list[dict]):
    sender      = os.environ.get("EMAIL_FROM", "")
    # Strip non-breaking spaces / whitespace — copied App Passwords often have \xa0
    password    = os.environ.get("EMAIL_PASSWORD", "").replace("\xa0", "").replace(" ", "").strip()
    sender_name = os.environ.get("EMAIL_NAME", sender)
    if not sender or not password:
        print("  Skipping report email (EMAIL_FROM / EMAIL_PASSWORD not set)")
        return

    date_str = datetime.now().strftime("%B %d, %Y")

    _MONO = "'Fira Code','Courier New',Courier,monospace"
    _SANS = "Manrope,Arial,Helvetica,sans-serif"
    _SERIF = "Spectral,Georgia,'Times New Roman',serif"
    _INK, _SUB, _FAINT, _LINE = "#15161a", "#52565e", "#9ba0aa", "#e5e7eb"
    _ACCENT, _CANVAS = "#4b3df5", "#eef0f3"
    _GOOD = "#0f7a56"

    def _clean(s: str) -> str:
        return _html.escape(str(s).replace("\xa0", " ").strip())

    def _avatar(name: str) -> tuple[str, str, str]:
        """Deterministic bg / fg / initials for a company name."""
        name = name or "?"
        hue = (sum(ord(c) for c in name) % 360) / 360.0
        r, g, b = colorsys.hls_to_rgb(hue, 0.88, 0.42)
        bg = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
        r, g, b = colorsys.hls_to_rgb(hue, 0.28, 0.48)
        fg = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
        return bg, fg, name[:2].upper()

    def _score_tier(score) -> str:
        try:
            return _GOOD if float(score) >= 85 else _ACCENT
        except (TypeError, ValueError):
            return _ACCENT

    def _headline(applied_n: int, pending_n: int) -> str:
        if applied_n and pending_n:
            return f"{applied_n} applied, {pending_n} to review"
        if applied_n:
            return f"{applied_n} applied &mdash; you're all caught up"
        if pending_n:
            return f"{pending_n} job{'s' if pending_n != 1 else ''} to review"
        return "No qualifying matches today"

    def _ledger_bar(applied_n: int, pending_n: int) -> str:
        total = applied_n + pending_n
        if total == 0:
            return f'<div style="height:8px;background:{_CANVAS};border-radius:2px;"></div>'
        applied_w = round(520 * applied_n / total)
        pending_w = 520 - applied_w
        segs = ""
        if applied_w:
            segs += f'<td width="{applied_w}" height="8" style="background:{_INK};font-size:0;line-height:8px;">&nbsp;</td>'
        if pending_w:
            segs += f'<td width="{pending_w}" height="8" style="background:{_ACCENT};font-size:0;line-height:8px;">&nbsp;</td>'
        return f"""
        <table role="presentation" width="520" cellpadding="0" cellspacing="0"><tr>{segs}</tr></table>
        <table role="presentation" width="520" cellpadding="0" cellspacing="0" style="margin-top:10px;"><tr>
          <td style="font-family:{_MONO};font-size:10px;letter-spacing:1px;color:{_INK};font-weight:700;">APPLIED</td>
          <td align="right" style="font-family:{_MONO};font-size:10px;letter-spacing:1px;color:{_ACCENT};font-weight:700;">TO REVIEW</td>
        </tr></table>"""

    def _job_row(j: dict, show_apply_btn: bool) -> str:
        score      = j.get("score", "—")
        reason     = _clean(j.get("reason", ""))
        title      = _clean(j.get("title", ""))
        company    = _clean(j.get("company", ""))
        loc        = _clean(j.get("location", ""))
        url        = j.get("url", "#")
        rank       = j.get("rank")
        rank_label = f"{rank:02d}" if isinstance(rank, int) else "--"
        tier       = _score_tier(score)
        avatar_bg, avatar_fg, initials = _avatar(j.get("company", ""))

        try:
            score_label = f"{int(float(score)):03d}"
        except (TypeError, ValueError):
            score_label = "---"

        stamp = (
            f'<span style="display:inline-block;font-family:{_MONO};font-size:12px;font-weight:700;'
            f'color:{tier};border:1px solid {tier};padding:3px 7px;border-radius:4px;white-space:nowrap;">{score_label}</span>'
        )

        if show_apply_btn:
            right_cell = (
                f'<a href="{url}" style="display:inline-block;background:{_ACCENT};color:#ffffff;'
                f'font-family:{_SANS};padding:9px 16px;border-radius:20px;text-decoration:none;'
                f'font-size:12.5px;font-weight:700;white-space:nowrap;">Apply &rarr;</a>'
            )
            meta = f'<div style="margin-top:6px;">{stamp}</div>'
        else:
            right_cell = stamp
            meta = ""

        return f"""
        <tr><td style="padding:18px 40px;border-top:1px solid {_LINE};" valign="top">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
          <td width="26" valign="top" style="font-family:{_MONO};font-size:11px;font-weight:700;color:{_FAINT};padding-top:3px;">{rank_label}</td>
          <td width="34" valign="top" style="padding:0 12px;">
            <table role="presentation" cellpadding="0" cellspacing="0"><tr>
              <td width="34" height="34" align="center" valign="middle"
                  style="width:34px;height:34px;background:{avatar_bg};color:{avatar_fg};
                         border-radius:6px;font-family:{_SANS};font-size:12px;font-weight:700;text-align:center;">{initials}</td>
            </tr></table>
          </td>
          <td valign="top" style="font-family:{_SANS};">
            <div style="font-size:15px;font-weight:700;color:{_INK};margin-bottom:2px;">{title}</div>
            <div style="font-size:13px;color:{_SUB};margin-bottom:6px;">{company}{"&nbsp;&middot;&nbsp;" + loc if loc else ""}</div>
            <div style="font-size:12.5px;color:{_FAINT};font-style:italic;">{reason}</div>
            {meta}
          </td>
          <td width="92" valign="top" align="right" style="padding-top:2px;">{right_cell}</td>
        </tr></table>
        </td></tr>"""

    applied_html = "".join(_job_row(j, False) for j in applied) or \
        f'<tr><td style="padding:18px 40px;font-family:{_SANS};color:{_FAINT};font-size:13px;">Nothing applied automatically this run.</td></tr>'

    not_applied_html = "".join(_job_row(j, True) for j in not_applied) or \
        f'<tr><td style="padding:18px 40px;font-family:{_SANS};color:{_FAINT};font-size:13px;">Nothing left for you — everything auto-applied.</td></tr>'

    html = f"""
    <html><head><meta charset="utf-8">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;0,700;1,400&family=Manrope:wght@400;600;800&family=Fira+Code:wght@500;700&display=swap">
    </head><body style="margin:0;padding:0;background:{_CANVAS};font-family:{_SANS};">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{_CANVAS};">
    <tr><td align="center" style="padding:48px 16px;">
    <table role="presentation" width="600" cellpadding="0" cellspacing="0"
           style="width:600px;max-width:600px;background:#ffffff;border:1px solid {_LINE};border-radius:16px;overflow:hidden;">
      <tr><td height="3" style="height:3px;line-height:3px;font-size:0;background:{_INK};">&nbsp;</td></tr>

      <tr><td style="padding:34px 40px 4px;">
        <table role="presentation" cellpadding="0" cellspacing="0" style="margin-bottom:16px;"><tr>
          <td width="6" height="6" style="background:{_ACCENT};font-size:0;line-height:0;">&nbsp;</td>
          <td style="padding-left:8px;font-family:{_MONO};font-size:11px;font-weight:700;letter-spacing:2px;color:{_ACCENT};text-transform:uppercase;">Overnight Dispatch</td>
        </tr></table>
        <div style="font-family:{_SERIF};font-size:27px;font-weight:700;color:{_INK};letter-spacing:-.2px;margin-bottom:6px;">{_headline(len(applied), len(not_applied))}</div>
        <div style="font-family:{_SANS};font-size:13px;color:{_FAINT};">{date_str}</div>
      </td></tr>

      <tr><td style="padding:22px 40px 4px;">
        {_ledger_bar(len(applied), len(not_applied))}
      </td></tr>

      <tr><td style="padding-top:22px;border-top:1px solid {_LINE};font-size:0;line-height:0;">&nbsp;</td></tr>

      <tr><td style="padding:24px 40px 6px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
          <td style="font-family:{_MONO};font-size:12px;font-weight:700;letter-spacing:1px;color:{_SUB};text-transform:uppercase;">Applied Automatically</td>
          <td align="right"><span style="font-family:{_MONO};font-size:11px;font-weight:700;color:{_INK};background:{_CANVAS};padding:3px 8px;border-radius:4px;">{len(applied):02d}</span></td>
        </tr></table>
      </td></tr>
      <tr><td><table role="presentation" width="100%" cellpadding="0" cellspacing="0">{applied_html}</table></td></tr>

      <tr><td style="padding:28px 40px 6px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
          <td style="font-family:{_MONO};font-size:12px;font-weight:700;letter-spacing:1px;color:{_SUB};text-transform:uppercase;">Needs Your Review</td>
          <td align="right"><span style="font-family:{_MONO};font-size:11px;font-weight:700;color:{_ACCENT};background:{_CANVAS};padding:3px 8px;border-radius:4px;">{len(not_applied):02d}</span></td>
        </tr></table>
      </td></tr>
      <tr><td><table role="presentation" width="100%" cellpadding="0" cellspacing="0">{not_applied_html}</table></td></tr>

      <tr><td style="padding:26px 40px 30px;border-top:1px solid {_LINE};text-align:center;">
        <div style="font-family:{_MONO};font-size:10px;letter-spacing:1.5px;color:{_FAINT};text-transform:uppercase;">&mdash; End of Dispatch &mdash;</div>
        <div style="font-family:{_SANS};font-size:11px;color:{_FAINT};margin-top:8px;">Generated automatically by your job search agent.</div>
      </td></tr>
    </table>
    </td></tr>
    </table>
    </body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = _hdr(f"Job Apply Report - {len(applied)} applied, {len(not_applied)} pending - {date_str}")
    msg["From"]    = _hdr(f"{sender_name} <{sender}>")
    msg["To"]      = _hdr(sender)

    # Encode all non-ASCII as HTML entities so smtplib never sees non-ASCII bytes
    html_safe = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
    msg.attach(MIMEText(html_safe, "html", "utf-8"))

    try:
        with smtplib.SMTP(
            os.environ.get("SMTP_HOST", "smtp.gmail.com"),
            int(os.environ.get("SMTP_PORT", "587")),
        ) as s:
            s.starttls()
            s.login(sender, password)
            # Use sendmail with explicit bytes to bypass compat32 encoding quirks
            raw = msg.as_bytes(policy=email.policy.SMTP)
            s.sendmail(sender, [sender], raw)
        print(f"\n  Report email sent → {sender}")
        print(f"  Applied: {len(applied)}  |  Needs manual apply: {len(not_applied)}")
    except Exception as e:
        print(f"  Report email failed: {e}")
        traceback.print_exc()


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
            headless=True,
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
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            slow_mo=30,
        )

        applied_jobs     = []
        not_applied_jobs = []

        # Skip jobs already applied to or emailed in previous runs
        db.init_db()
        already_seen = db.get_seen_urls()

        # Supplement scored jobs with Himalayas Playwright search results
        seen_urls = {j.get("url") for j in jobs}
        hw_jobs = await scrape_himalayas_playwright(job_ctx, seen_urls | already_seen)
        jobs = jobs + hw_jobs

        eligible = [
            j for j in jobs[:100]
            if j.get("score", 0) >= 70 and j.get("url") not in already_seen
        ]
        skipped = len([j for j in jobs[:100] if j.get("url") in already_seen])
        print(f"\n  {len(eligible)} jobs with score ≥ 70 (out of {len(jobs[:100])} ranked, {skipped} already seen skipped)")

        for job in eligible:
            print(f"\n→ #{job.get('rank', '—')} [{job['score']}/100]  {job['title']} @ {job['company']}")

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
                db.mark_applied(job)
                if email_ready:
                    dom = _company_domain(job["company"], job["url"])
                    email, name = await find_recruiter(mm_browser, job, dom)
                    if email:
                        try:
                            send_cold_email(email, name, job)
                        except Exception as e:
                            print(f"  Cold email failed: {e}")
                    else:
                        print(f"  No recruiter email found for {dom}")
                else:
                    print("  Skipping cold email (EMAIL_FROM / EMAIL_PASSWORD not set)")
            else:
                not_applied_jobs.append(job)

            await asyncio.sleep(random.uniform(5, 10))

        await mm_browser.close()
        await job_ctx.close()

    send_report_email(applied_jobs, not_applied_jobs)
    db.mark_seen(not_applied_jobs)


if __name__ == "__main__":
    asyncio.run(_run())
