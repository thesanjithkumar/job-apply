import os, json
from pathlib import Path

import openai
from dotenv import load_dotenv
from jobspy import scrape_jobs

load_dotenv()

SEARCH_TERMS = ["AI Engineer", "AI Full Stack Engineer", "Full Stack Engineer"]
SITES = ["linkedin", "indeed", "glassdoor"]
RESULTS_PER_SEARCH = 15  # per site per term

# (name, base_url, api_key_env, model)
# Ordered by free-tier generosity. Script skips entries with no key set.
# Model IDs may drift as providers update their catalogues — check provider docs if one fails.
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

# ponytail: catch-all — every provider gets one attempt; any API failure means try next
_SKIP_ON = (openai.RateLimitError, openai.AuthenticationError, openai.PermissionDeniedError)


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


def scrape_all() -> list[dict]:
    seen, jobs = set(), []
    for term in SEARCH_TERMS:
        print(f"  Scraping: {term} ...")
        try:
            df = scrape_jobs(
                site_name=SITES,
                search_term=term,
                results_wanted=RESULTS_PER_SEARCH,
                country_indeed="USA",
                linkedin_fetch_description=True,
                verbose=1,
            )
        except Exception as e:
            print(f"  Warning: scrape failed for '{term}': {e}")
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
                "location": str(row.get("location", "")),
                "url": url,
                "description": str(desc)[:600],
            })
    return jobs


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
            start, end = raw.index("["), raw.rindex("]") + 1
            rankings = json.loads(raw[start:end])
            print(f"  Ranked by {name}")
            return [
                {
                    **jobs[r["index"] - 1],
                    "rank": r["rank"],
                    "score": r["score"],
                    "reason": r["reason"],
                }
                for r in rankings
                if isinstance(r.get("index"), int) and 1 <= r["index"] <= len(jobs)
            ]
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
