import os, json, re
from pathlib import Path

import anthropic
import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from jobspy import scrape_jobs

load_dotenv()

SEARCH_TERMS = ["AI Engineer", "AI Full Stack Engineer", "Full Stack Engineer"]
SITES = ["linkedin", "indeed", "glassdoor"]
RESULTS_PER_SEARCH = 15
LOCATIONS = ["Bengaluru, India", "Hyderabad, India", "Bangalore, India"]

# (name, base_url, api_key_env, model)
PROVIDERS = [
    ("Groq",        "https://api.groq.com/openai/v1",                           "GROQ_API_KEY",       "llama-3.3-70b-versatile"),
    ("Cerebras",    "https://api.cerebras.ai/v1",                                "CEREBRAS_API_KEY",   "llama3.3-70b"),
    ("OpenRouter",  "https://openrouter.ai/api/v1",                              "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    ("Gemini",      "https://generativelanguage.googleapis.com/v1beta/openai/",  "GEMINI_API_KEY",     "gemini-2.5-flash-lite"),
    ("Mistral",     "https://api.mistral.ai/v1",                                 "MISTRAL_API_KEY",    "mistral-small-latest"),
    ("SambaNova",   "https://api.sambanova.ai/v1",                               "SAMBANOVA_API_KEY",  "Meta-Llama-3.3-70B-Instruct"),
    ("NVIDIA NIM",  "https://integrate.api.nvidia.com/v1",                       "NVIDIA_API_KEY",     "meta/llama-3.3-70b-instruct"),
    ("Kilo",        "https://api.kilo.ai/v1",                                    "KILO_API_KEY",       "nvidia/nemotron-3-ultra-550b-a55b"),
    ("Scaleway",    "https://api.scaleway.ai/v1",                                "SCALEWAY_API_KEY",   "llama-3.3-70b-instruct-fp8"),
    ("Hyperbolic",  "https://api.hyperbolic.xyz/v1",                             "HYPERBOLIC_API_KEY", "meta-llama/Llama-3.3-70B-Instruct"),
    ("Nebius",      "https://api.studio.nebius.ai/v1",                           "NEBIUS_API_KEY",     "meta-llama/Llama-3.3-70B-Instruct"),
    ("Fireworks",   "https://api.fireworks.ai/inference/v1",                     "FIREWORKS_API_KEY",  "accounts/fireworks/models/llama-v3p3-70b-instruct"),
    ("Novita",      "https://api.novita.ai/v3/openai",                           "NOVITA_API_KEY",     "meta-llama/llama-3.3-70b-instruct"),
    ("Inference",   "https://api.inference.net/v1",                              "INFERENCE_API_KEY",  "meta-llama/llama-3.3-70b-instruct:free"),
    ("Lambda",      "https://api.lambdalabs.com/v1",                             "LAMBDA_API_KEY",     "llama3.3-70b-instruct-fp8"),
    ("Chutes",      "https://llm.chutes.ai/v1",                                  "CHUTES_API_KEY",     "chutesai/Llama-3.3-70B-Instruct"),
    ("OVH",         "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",         "OVH_API_KEY",        "Meta-Llama-3_3-70B-Instruct"),
    ("HuggingFace", "https://api-inference.huggingface.co/v1",                   "HF_TOKEN",           "meta-llama/Llama-3.3-70B-Instruct"),
    ("Upstage",     "https://api.upstage.ai/v1",                                 "UPSTAGE_API_KEY",    "solar-pro"),
    ("Kluster",     "https://api.kluster.ai/v1",                                 "KLUSTER_API_KEY",    "meta-llama/Llama-3.3-70B-Instruct"),
]

_SKIP_ON = (openai.RateLimitError, openai.AuthenticationError, openai.PermissionDeniedError)

# Greenhouse and Lever board slugs for India tech companies
GREENHOUSE_BOARDS = [
    # India tech companies
    "thoughtworks", "twilio", "truecaller", "payoneer",
    "circleslife", "productiv", "purestorage", "memryx",
    # Global companies with India offices
    "stripe", "coinbase", "hubspot", "zendesk",
    "figma", "notion", "reddit", "squarespace",
    "postman", "browserstack", "freshworks",
]
LEVER_BOARDS = [
    "meesho", "fampay", "stable-money1",
    "razorpay", "swiggy",
]

# Workday ATS: (tenant, wd_version, board_name, display_name)
# URL pattern: https://{tenant}.wd{n}.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs
# Only boards confirmed to accept unauthenticated POST requests are listed.
WORKDAY_BOARDS = [
    ("visa",   "5", "Visa",  "Visa"),
    ("paypal", "1", "jobs",  "PayPal"),
]

# Role keywords for Greenhouse/Lever/API source filtering
_TARGET_ROLES = [
    "ai engineer", "ai full stack", "full stack", "fullstack",
    "software engineer", "software developer", "sde",
    "machine learning", "ml engineer",
    "backend engineer", "backend developer",
    "frontend engineer", "frontend developer",
    "python developer", "python engineer",
    "data engineer", "data scientist",
]
_INDIA_KEYWORDS = [
    "india", "bangalore", "bengaluru", "hyderabad", "mumbai",
    "delhi", "noida", "gurgaon", "gurugram", "chennai",
    "remote india", "worldwide", "anywhere",
]
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/html, */*",
}


def _role_match(title: str) -> bool:
    t = title.lower()
    return any(role in t for role in _TARGET_ROLES)


def _india_loc(location: str) -> bool:
    loc = (location or "").lower()
    return any(kw in loc for kw in _INDIA_KEYWORDS)


def load_resume(resume_dir: str = "resume") -> str:
    folder = Path(resume_dir)
    if not folder.exists():
        raise FileNotFoundError("Create a resume/ folder and drop your resume file in it.")
    for f in sorted(folder.iterdir()):
        if f.suffix == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(str(f))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        if f.suffix in (".txt", ".md"):
            return f.read_text(encoding="utf-8")
    raise FileNotFoundError("Put your resume (PDF or TXT/MD) inside the resume/ folder.")


# ── Scraper helpers ────────────────────────────────────────────────────────────

def _scrape_jobspy(seen: set, jobs: list):
    """LinkedIn / Indeed / Glassdoor via python-jobspy, targeting Bengaluru and Hyderabad."""
    for term in SEARCH_TERMS:
        for location in LOCATIONS:
            print(f"  [JobSpy] {term} @ {location} ...")
            try:
                df = scrape_jobs(
                    site_name=SITES,
                    search_term=term,
                    location=location,
                    results_wanted=RESULTS_PER_SEARCH,
                    country_indeed="India",
                    linkedin_fetch_description=True,
                    verbose=0,
                )
            except Exception as e:
                print(f"  Warning: JobSpy failed for '{term}' @ {location}: {e}")
                continue
            for _, row in df.iterrows():
                url = str(row.get("job_url", ""))
                if not url or url in seen:
                    continue
                seen.add(url)
                desc = row.get("description") or ""
                jobs.append({
                    "title": str(row.get("title", "")),
                    "company": str(row.get("company", "")),
                    "location": str(row.get("location", location)),
                    "url": url,
                    "description": str(desc)[:600],
                    "source": "JobSpy",
                })


def _scrape_greenhouse(seen: set, jobs: list):
    """Greenhouse public ATS boards for India tech companies (free, no auth)."""
    print("  [Greenhouse] Scraping India ATS boards...")
    for board in GREENHOUSE_BOARDS:
        try:
            url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"
            res = requests.get(url, headers=_HEADERS, timeout=15)
            if res.status_code != 200:
                continue
            for j in res.json().get("jobs", []):
                title = j.get("title", "")
                location = (j.get("location") or {}).get("name", "")
                if not _role_match(title) or not _india_loc(location):
                    continue
                job_url = j.get("absolute_url", "")
                if not job_url or job_url in seen:
                    continue
                seen.add(job_url)
                desc = j.get("content", "") or ""
                jobs.append({
                    "title": title,
                    "company": board.capitalize(),
                    "location": location or "India",
                    "url": job_url,
                    "description": str(desc)[:600],
                    "source": "Greenhouse",
                })
        except Exception as e:
            print(f"  Warning: Greenhouse '{board}' failed: {e}")


def _scrape_lever(seen: set, jobs: list):
    """Lever public ATS boards for India tech companies (free, no auth)."""
    print("  [Lever] Scraping India ATS boards...")
    for board in LEVER_BOARDS:
        try:
            url = f"https://api.lever.co/v0/postings/{board}?mode=json"
            res = requests.get(url, headers=_HEADERS, timeout=15)
            if res.status_code != 200:
                continue
            for j in res.json():
                title = j.get("text", "")
                location = (j.get("categories") or {}).get("location", "")
                if not _role_match(title) or not _india_loc(location):
                    continue
                job_url = j.get("hostedUrl", "")
                if not job_url or job_url in seen:
                    continue
                seen.add(job_url)
                desc = (j.get("descriptionPlain") or j.get("description") or "")
                jobs.append({
                    "title": title,
                    "company": board.capitalize(),
                    "location": location or "India",
                    "url": job_url,
                    "description": str(desc)[:600],
                    "source": "Lever",
                })
        except Exception as e:
            print(f"  Warning: Lever '{board}' failed: {e}")


def _scrape_remoteok(seen: set, jobs: list):
    """RemoteOK public API — worldwide remote + India-eligible jobs."""
    print("  [RemoteOK] Scraping...")
    try:
        res = requests.get("https://remoteok.com/api", headers=_HEADERS, timeout=15)
        for j in res.json()[1:]:  # index 0 is metadata
            if not _role_match(j.get("position", "")):
                continue
            loc = j.get("location", "") or ""
            is_open = not loc.strip() or "worldwide" in loc.lower() or "anywhere" in loc.lower()
            if not (_india_loc(loc) or is_open):
                continue
            job_url = j.get("url", "")
            if not job_url or job_url in seen:
                continue
            seen.add(job_url)
            tags = j.get("tags", [])
            jobs.append({
                "title": j.get("position", ""),
                "company": j.get("company", ""),
                "location": loc or "Worldwide Remote",
                "url": job_url,
                "description": str(j.get("description", ""))[:600],
                "source": "RemoteOK",
            })
    except Exception as e:
        print(f"  Warning: RemoteOK failed: {e}")


def _scrape_arbeitnow(seen: set, jobs: list):
    """Arbeitnow public API — remote and India-eligible tech jobs."""
    print("  [Arbeitnow] Scraping...")
    try:
        res = requests.get("https://www.arbeitnow.com/api/job-board-api", headers=_HEADERS, timeout=15)
        for j in res.json().get("data", []):
            if not _role_match(j.get("title", "")):
                continue
            loc = j.get("location", "") or "Remote"
            is_remote = j.get("remote", False)
            if not (_india_loc(loc) or is_remote):
                continue
            job_url = j.get("url", "")
            if not job_url or job_url in seen:
                continue
            seen.add(job_url)
            company = j.get("company", {})
            company_name = company.get("name", "Unknown") if isinstance(company, dict) else j.get("company_name", "Unknown")
            jobs.append({
                "title": j.get("title", ""),
                "company": company_name,
                "location": loc,
                "url": job_url,
                "description": str(j.get("description", ""))[:600],
                "source": "Arbeitnow",
            })
    except Exception as e:
        print(f"  Warning: Arbeitnow failed: {e}")



def _scrape_jobviareferral(seen: set, jobs: list):
    """Job Via Referral — fresher referral postings for India."""
    print("  [JobViaReferral] Scraping...")
    try:
        res = requests.get(
            "https://jobviareferral.com/category/fresher-referral-jobs/",
            headers=_HEADERS, timeout=15,
        )
        if res.status_code != 200:
            return
        soup = BeautifulSoup(res.text, "html.parser")
        for heading in soup.find_all(["h1", "h2", "h3"]):
            link = heading.find("a")
            if not link:
                continue
            raw_title = link.get_text(strip=True)
            title = re.sub(r'^[^\w]+', '', raw_title).strip()  # strip leading emoji
            if not title or not _role_match(title):
                continue
            href = link.get("href", "")
            if not href or href in seen:
                continue
            seen.add(href)
            t_lower = title.lower()
            loc = "India"
            if "bengaluru" in t_lower or "bangalore" in t_lower:
                loc = "Bengaluru, India"
            elif "hyderabad" in t_lower:
                loc = "Hyderabad, India"
            jobs.append({
                "title": title,
                "company": "See listing",
                "location": loc,
                "url": href,
                "description": "",
                "source": "JobViaReferral",
            })
    except Exception as e:
        print(f"  Warning: JobViaReferral failed: {e}")


def _scrape_workday(seen: set, jobs: list):
    """Workday ATS public job search API — Visa, Mastercard, PayPal, AmEx, Citi etc."""
    print("  [Workday] Scraping career pages...")
    headers = {**_HEADERS, "Content-Type": "application/json"}
    for tenant, wd_ver, board, company in WORKDAY_BOARDS:
        url = f"https://{tenant}.wd{wd_ver}.myworkdayjobs.com/wday/cxs/{tenant}/{board}/jobs"
        base_url = f"https://{tenant}.wd{wd_ver}.myworkdayjobs.com"
        found = 0
        for term in SEARCH_TERMS:
            try:
                res = requests.post(
                    url,
                    json={"limit": 20, "offset": 0, "searchText": term, "appliedFacets": {}},
                    headers=headers,
                    timeout=15,
                )
                if res.status_code != 200:
                    break  # wrong tenant/board — skip this company entirely
                for posting in res.json().get("jobPostings", []):
                    title = posting.get("title", "")
                    location = posting.get("locationsText", "")
                    if not _role_match(title) or not _india_loc(location):
                        continue
                    ext_path = posting.get("externalPath", "")
                    job_url = f"{base_url}{ext_path}" if ext_path else base_url
                    if not job_url or job_url in seen:
                        continue
                    seen.add(job_url)
                    jobs.append({
                        "title": title,
                        "company": company,
                        "location": location or "India",
                        "url": job_url,
                        "description": "",
                        "source": "Workday",
                    })
                    found += 1
            except Exception as e:
                print(f"  Warning: Workday '{company}' / '{term}' failed: {e}")
        if found:
            print(f"    {company}: {found} matched")


def _scrape_apna(seen: set, jobs: list):
    """Apna.co — jobs in Bengaluru and Hyderabad (paginated, up to 5 pages each)."""
    MAX_PAGES = 5
    city_slugs = [("bengaluru", "Bengaluru, India"), ("hyderabad", "Hyderabad, India"), ("bangalore", "Bangalore, India")]
    for city_slug, city_label in city_slugs:
        print(f"  [Apna.co] Scraping {city_label}...")
        for page in range(1, MAX_PAGES + 1):
            try:
                base = f"https://apna.co/jobs/jobs-in-{city_slug}"
                url = base if page == 1 else f"{base}?page={page}"
                res = requests.get(url, headers=_HEADERS, timeout=15)
                if res.status_code != 200:
                    break
                soup = BeautifulSoup(res.text, "html.parser")
                links = soup.find_all("a", href=lambda h: h and f"/job/{city_slug}/" in h)
                if not links:
                    break
                for link in links:
                    href = link.get("href", "")
                    slug = href.rstrip("/").split("/")[-1]
                    slug = re.sub(r'-\d+$', '', slug)
                    title = slug.replace("-", " ").strip().title()
                    if not title or not _role_match(title):
                        continue
                    full_href = href if href.startswith("http") else "https://apna.co" + href
                    if full_href in seen:
                        continue
                    seen.add(full_href)
                    jobs.append({
                        "title": title,
                        "company": "See listing",
                        "location": city_label,
                        "url": full_href,
                        "description": "",
                        "source": "Apna",
                    })
            except Exception as e:
                print(f"  Warning: Apna.co page {page} for {city_slug} failed: {e}")
                break


def scrape_all() -> list[dict]:
    seen, jobs = set(), []

    _scrape_jobspy(seen, jobs)
    _scrape_greenhouse(seen, jobs)
    _scrape_lever(seen, jobs)
    _scrape_workday(seen, jobs)
    _scrape_remoteok(seen, jobs)
    _scrape_arbeitnow(seen, jobs)
    _scrape_jobviareferral(seen, jobs)
    _scrape_apna(seen, jobs)

    # Ensure every job has a description field
    for j in jobs:
        if "description" not in j:
            j["description"] = ""

    return jobs


def _parse_rankings(raw: str, jobs: list[dict]) -> list[dict]:
    raw = raw.strip()
    start, end = raw.index("["), raw.rindex("]") + 1
    rankings = json.loads(raw[start:end])
    return [
        {**jobs[r["index"] - 1], "rank": r["rank"], "score": r["score"], "reason": r["reason"]}
        for r in rankings
        if isinstance(r.get("index"), int) and 1 <= r["index"] <= len(jobs)
    ]


def rank_jobs(resume: str, jobs: list[dict]) -> list[dict]:
    jobs_blob = "\n\n".join(
        f"[{i+1}] {j['title']} @ {j['company']} ({j['location']})\n{j['description']}"
        for i, j in enumerate(jobs)
    )
    prompt = (
        "You are a career advisor. Rank the top 20 best-matching jobs for this candidate.\n\n"
        f"RESUME:\n{resume}\n\n"
        f"JOB LISTINGS ({len(jobs)} total):\n{jobs_blob}\n\n"
        "Return ONLY a JSON array ranked best to worst (top 20 or fewer):\n"
        '[{"rank":1,"index":<1-based job index>,"score":<0-100>,"reason":"<one sentence>"},...]\n'
        "No other text."
    )

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("  Trying Anthropic (claude-opus-5)...")
        try:
            client = anthropic.Anthropic()
            full_text = ""
            with client.messages.stream(
                model="claude-opus-5",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    full_text += chunk
            print("  Ranked by Anthropic")
            return _parse_rankings(full_text, jobs)
        except Exception as e:
            print(f"  Anthropic failed: {e}")

    available = [(n, u, e, m) for n, u, e, m in PROVIDERS if os.environ.get(e)]
    if not available:
        raise RuntimeError(
            "No provider API keys found. Copy .env.example to .env and add at least one key."
        )

    for name, base_url, env_var, model in available:
        print(f"  Trying {name} ({model})...")
        try:
            client = openai.OpenAI(base_url=base_url, api_key=os.environ[env_var])
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.1,
                timeout=120,
            )
            raw = resp.choices[0].message.content.strip()
            print(f"  Ranked by {name}")
            return _parse_rankings(raw, jobs)
        except _SKIP_ON as e:
            print(f"  {name}: exhausted/unauthorized — {e}")
        except openai.APIStatusError as e:
            if e.status_code in (402, 429, 503):
                print(f"  {name}: quota/unavailable (HTTP {e.status_code})")
            else:
                print(f"  {name}: API error (HTTP {e.status_code}) — {e.message}")
        except Exception as e:
            print(f"  {name}: failed — {e}")

    raise RuntimeError("All configured providers failed or exhausted. Add more keys or try again later.")


if __name__ == "__main__":
    print("→ Loading resume...")
    resume = load_resume()
    print(f"  Loaded ({len(resume):,} chars)")

    print("→ Scraping jobs...")
    jobs = scrape_all()
    print(f"  Found {len(jobs)} unique listings")

    if not jobs:
        print("No jobs found. Check your internet connection and try again.")
        raise SystemExit(1)

    print("→ Ranking with AI (trying providers in order)...")
    ranked = rank_jobs(resume, jobs)

    bar = "=" * 60
    print(f"\n{bar}\nTOP {len(ranked)} MATCHES FOR YOUR RESUME\n{bar}\n")
    for j in ranked:
        print(f"#{j['rank']}  [{j['score']}/100]  {j['title']} @ {j['company']}")
        print(f"     {j['location']}")
        print(f"     {j['reason']}")
        print(f"     {j['url']}\n")

    with open("results.json", "w") as f:
        json.dump(ranked, f, indent=2)
    print("Results saved to results.json")
