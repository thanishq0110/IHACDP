"""Download the reference papers that are legitimately redistributable.

Only open-access articles and author/arXiv preprints are fetched. The rest of
the bibliography sits behind publisher paywalls; those are listed in the
generated papers/README.md with their DOI rather than downloaded, because
putting a paywalled PDF in a public repository is redistribution.

    python scripts/fetch_papers.py
"""
from __future__ import annotations
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "papers"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# (reference number, filename stem, short citation, licence note, url)
OPEN = [
    (3, "fansi-tchango-2022-ddxplus",
     'Fansi Tchango et al., "DDXPlus: A New Dataset For Automatic Medical Diagnosis", NeurIPS 2022',
     "arXiv:2205.09148", "https://arxiv.org/pdf/2205.09148"),
    (7, "ramana-2011-liver-classifiers",
     'Ramana, Babu & Venkateswarlu, "A critical study of selected classification algorithms for liver disease diagnosis", IJDMS 2011',
     "AIRCC open access", "https://airccse.org/journal/ijdms/papers/3211ijdms07.pdf"),
    (11, "breiman-2001-random-forests",
     'Breiman, "Random Forests", Machine Learning 2001',
     "author copy, Berkeley", "https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf"),
    (12, "chen-2016-xgboost",
     'Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016',
     "arXiv:1603.02754", "https://arxiv.org/pdf/1603.02754"),
    (13, "ribeiro-2016-lime",
     'Ribeiro, Singh & Guestrin, "Why Should I Trust You?", KDD 2016',
     "arXiv:1602.04938", "https://arxiv.org/pdf/1602.04938"),
    (14, "lundberg-2017-shap",
     'Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NIPS 2017',
     "arXiv:1705.07874", "https://arxiv.org/pdf/1705.07874"),
    (15, "lundberg-2020-tree-explainer",
     'Lundberg et al., "From local explanations to global understanding with explainable AI for trees", Nat Mach Intell 2020',
     "arXiv:1905.04610", "https://arxiv.org/pdf/1905.04610"),
    (16, "rudin-2019-stop-explaining",
     'Rudin, "Stop explaining black box machine learning models for high stakes decisions", Nat Mach Intell 2019',
     "arXiv:1811.10154", "https://arxiv.org/pdf/1811.10154"),
    (18, "singhal-2023-med-palm",
     'Singhal et al., "Large language models encode clinical knowledge", Nature 2023',
     "arXiv:2212.13138", "https://arxiv.org/pdf/2212.13138"),
    (21, "ji-2023-hallucination-survey",
     'Ji et al., "Survey of Hallucination in Natural Language Generation", ACM Comput Surv 2023',
     "arXiv:2202.03629", "https://arxiv.org/pdf/2202.03629"),
    (22, "wei-2022-chain-of-thought",
     'Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", NeurIPS 2022',
     "arXiv:2201.11903", "https://arxiv.org/pdf/2201.11903"),
    (23, "gemma-team-2025-gemma3",
     'Gemma Team, "Gemma 3 Technical Report", 2025',
     "arXiv:2503.19786", "https://arxiv.org/pdf/2503.19786"),
]

# Open access, but the host serves a bot check to any scripted request (BMJ is
# behind Cloudflare, PMC and Europe PMC both return an interstitial). Free to
# download by hand from the link below.
MANUAL = [
    (1, 'Semigran et al., "Evaluation of symptom checkers for self diagnosis and triage", BMJ 2015',
     "https://www.bmj.com/content/351/bmj.h3480"),
    (9, 'Xie et al., "Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques", Prev Chronic Dis 2019',
     "https://www.cdc.gov/pcd/issues/2019/19_0109.htm"),
]

# Behind a publisher paywall - cited by DOI, not redistributed.
CLOSED = [
    (2, 'Fraser, Coiera & Wong, "Safety of patient-facing digital symptom checkers", The Lancet 2018',
     "10.1016/S0140-6736(18)32819-8"),
    (4, 'Sparck Jones, "A statistical interpretation of term specificity and its application in retrieval", J Doc 1972',
     "10.1108/eb026526"),
    (5, 'Detrano et al., "International application of a new probability algorithm for the diagnosis of coronary artery disease", Am J Cardiol 1989',
     "10.1016/0002-9149(89)90524-9"),
    (6, 'Street, Wolberg & Mangasarian, "Nuclear feature extraction for breast tumor diagnosis", Proc SPIE 1993',
     "10.1117/12.148698"),
    (17, 'Ghassemi, Oakden-Rayner & Beam, "The false hope of current approaches to explainable AI in health care", Lancet Digit Health 2021',
     "10.1016/S2589-7500(21)00208-9"),
    (19, 'Thirunavukarasu et al., "Large language models in medicine", Nature Medicine 2023',
     "10.1038/s41591-023-02448-8"),
    (20, 'Ayers et al., "Comparing physician and artificial intelligence chatbot responses to patient questions", JAMA Intern Med 2023',
     "10.1001/jamainternmed.2023.1838"),
]

# Dataset donations rather than papers.
DATASETS = [
    (8, "Rubini, Soundarapandian & Eswaran, Chronic Kidney Disease, UCI ML Repository 2015",
     "10.24432/C5G020"),
    (10, "CDC Diabetes Health Indicators, UCI ML Repository (dataset 891)",
     "10.24432/C53919"),
]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf,*/*"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    got, failed = [], []
    for ref, stem, cite, licence, url in OPEN:
        dest = OUT / f"ref{ref:02d}-{stem}.pdf"
        if dest.exists() and dest.stat().st_size > 40_000:
            got.append((ref, dest, cite, licence)); print(f"  have  {dest.name}"); continue
        try:
            data = get(url)
            if not data.startswith(b"%PDF"):
                raise ValueError(f"not a PDF ({data[:24]!r})")
            dest.write_bytes(data)
            got.append((ref, dest, cite, licence))
            print(f"  ok    {dest.name}  ({len(data) // 1024} KB)")
        except Exception as e:
            failed.append((ref, cite, url, str(e)[:90]))
            print(f"  FAIL  [{ref}] {type(e).__name__}: {str(e)[:70]}")

    lines = ["# Reference papers", "",
             "Sources for the works cited in the capstone report. Downloaded by",
             "`python scripts/fetch_papers.py`.", "",
             f"## Included ({len(got)})", "",
             "| Ref | File | Citation | Source |", "|---|---|---|---|"]
    for ref, dest, cite, licence in sorted(got):
        lines.append(f"| [{ref}] | `{dest.name}` | {cite} | {licence} |")

    lines += ["", f"## Open access, download by hand ({len(MANUAL)})", "",
              "Both are free to read. Their hosts serve a bot check to scripted requests,",
              "so they are not fetched automatically - open the link and save the PDF here.",
              "", "| Ref | Citation | Link |", "|---|---|---|"]
    for ref, cite, url in MANUAL:
        lines.append(f"| [{ref}] | {cite} | [{url}]({url}) |")

    lines += ["", f"## Not included - publisher paywall ({len(CLOSED)})", "",
              "These are cited in the report but are not redistributable, so only the",
              "DOI is recorded here.", "",
              "| Ref | Citation | DOI |", "|---|---|---|"]
    for ref, cite, doi in CLOSED:
        lines.append(f"| [{ref}] | {cite} | [{doi}](https://doi.org/{doi}) |")

    lines += ["", "## Dataset donations, not papers", "",
              "| Ref | Citation | DOI |", "|---|---|---|"]
    for ref, cite, doi in DATASETS:
        lines.append(f"| [{ref}] | {cite} | [{doi}](https://doi.org/{doi}) |")

    if failed:
        lines += ["", "## Download failed", "",
                  "| Ref | Citation | URL | Error |", "|---|---|---|---|"]
        for ref, cite, url, err in failed:
            lines.append(f"| [{ref}] | {cite} | {url} | {err} |")

    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    total = sum(d.stat().st_size for _, d, _, _ in got)
    print(f"\n  {len(got)} papers, {total // 1024 // 1024} MB -> papers/")
    if failed:
        print(f"  {len(failed)} failed, recorded in papers/README.md")


if __name__ == "__main__":
    main()
