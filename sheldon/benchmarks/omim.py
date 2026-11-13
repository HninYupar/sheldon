"""Literature search on rare diseases listed in OMIM.

Each task is one OMIM phenotype entry. The solver is given the disease name,
symbol and phenotype MIM number (and, only if `benchmark_opts.gene_hint=true`, the gene)
and must find the published literature on it. The gold answer is the entry's
own reference list; the headline score is recall of those references by PMID.

Data layout, the same as ARC-AGI (one task per file):

    data/omim/evaluation/<MIM>.json
    data/omim/training/<MIM>.json

    {"train": [],
     "test":  [{"input":  {"disease", "symbol", "phenotype_mim_number", "gene"},
                "output": [{"title", "pmid", ...}, ...]}],
     "meta":  {"omim_url"}}

Tools: three PubMed probes over NCBI E-utilities (search, fetch, related). No
Python sandbox and no OMIM access -- the answer key is OMIM's own list, so a
tool that can read omim.org would let the solver copy it. For the same reason,
run with `-p probes=[]` so the generic `run_shell` probe is not offered either.

Set NCBI_API_KEY (free) to raise PubMed's rate limit from 3 to 10 requests/s.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, ClassVar

from sheldon.benchmarks.base import Benchmark
from sheldon.core.probes import Probe, ProbeResult
from sheldon.core.registry import benchmarks
from sheldon.core.types import Prediction, RunContext, Score, Task

_SPLIT_DIRS = {"train": "training", "eval": "evaluation", "evaluation": "evaluation"}

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/" # base URL for NCBI's E-utilities, the offical free API for PubMed
_TOOL = "sheldon-omim" # a name sent with each request; NCBI asks API users to identify themselves

# Speed limiter for requests to PudMed
class SpeedLimiter:
    """NCBI allows 3 requests without a key, 10 with one. Shared by every
    probe in the process, so concurrent tasks cannot trip the limit together."""

    def __init__(self) -> None:
        self.gap = 0.11 if os.environ.get("NCBI_API_KEY") else 0.35 # minimum wait betwee requests
        self.lock = asyncio.Lock()
        self.last = 0.0

    async def wait(self) -> None:
        async with self.lock:
            delay = self.last + self.gap - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.last = time.monotonic()

_speed_limiter: SpeedLimiter | None = None

async def eutils(endpoint: str, **params: Any) -> bytes:
    """GET one E-utilities endpoint, throttled, with two retries."""
    global _speed_limiter
    if _speed_limiter is None:
        _speed_limiter = SpeedLimiter()
    params = {k: v for k, v in params.items() if v is not None}
    params.setdefault("tool", _TOOL)
    if os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    url = EUTILS + endpoint + "?" + urllib.parse.urlencode(params, doseq=True)

    def get() -> bytes:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed NCBI host
            return resp.read()

    err: Exception | None = None
    for attempt in range(3):
        await _speed_limiter.wait()
        try:
            return await asyncio.to_thread(get)
        except Exception as e:  # noqa: BLE001 - network faults are retried, then surfaced
            err = e
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"PubMed {endpoint} failed: {err}")

async def summaries(pmids: list[str]) -> list[dict[str, Any]]:
    """PMIDs -> short records (title, first author, year, journal), in order."""
    if not pmids:
        return []
    raw = json.loads(await eutils("esummary.fcgi", db="pubmed", id=",".join(pmids), retmode="json"))
    result = raw.get("result", {})
    out = []
    for pmid in pmids:
        r = result.get(pmid)
        if not r or "error" in r:
            continue
        authors = r.get("authors") or []
        doi = next((a["value"] for a in r.get("articleids", []) if a.get("idtype") == "doi"), "")
        out.append(
            {
                "pmid": pmid,
                "title": r.get("title", ""),
                "first_author": authors[0]["name"] if authors else "",
                "year": (r.get("pubdate") or "")[:4],
                "journal": r.get("source", ""),
                "doi": doi,
                "pubtype": r.get("pubtype", []),
            }
        )
    return out

def _text(node: ET.Element | None) -> str:
    return "".join(node.itertext()).strip() if node is not None else ""

def parse_efetch(xml: bytes) -> list[dict[str, Any]]:
    """PubMed XML -> one record per article, with the abstract and MeSH terms."""
    out = []
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        a = cit.find("Article") if cit is not None else None
        if cit is None or a is None:
            continue
        abstract = "\n".join(
            (f"{p.get('Label')}: " if p.get("Label") else "") + _text(p)
            for p in a.findall("Abstract/AbstractText")
        )
        authors = [
            f"{_text(au.find('LastName'))} {_text(au.find('Initials'))}".strip()
            or _text(au.find("CollectiveName"))
            for au in a.findall("AuthorList/Author")
        ]
        doi = next(
            (_text(i) for i in art.findall("PubmedData/ArticleIdList/ArticleId") if i.get("IdType") == "doi"),
            "",
        )
        year = _text(a.find("Journal/JournalIssue/PubDate/Year")) or _text(
            a.find("Journal/JournalIssue/PubDate/MedlineDate")
        )[:4]
        out.append(
            {
                "pmid": _text(cit.find("PMID")),
                "title": _text(a.find("ArticleTitle")),
                "authors": authors[:6] + (["et al."] if len(authors) > 6 else []),
                "year": year,
                "journal": _text(a.find("Journal/Title")),
                "doi": doi,
                "pubtype": [_text(p) for p in a.findall("PublicationTypeList/PublicationType")],
                "mesh": [_text(m.find("DescriptorName")) for m in cit.findall("MeshHeadingList/MeshHeading")],
                "abstract": abstract,
            }
        )
    return out


def _pmid_list(value: Any, cap: int) -> list[str]:
    ids = re.findall(r"\d+", value if isinstance(value, str) else " ".join(map(str, value or [])))
    return list(dict.fromkeys(ids))[:cap]


# -- probes -------------------------------------------------------------------


class PubMedSearchProbe(Probe):
    name = "pubmed_search"
    description = (
        "Search PubMed and get back matching papers (PMID, title, first author, year, journal). "
        "Accepts full PubMed query syntax: field tags such as [gene], [tiab], [mh], [au], [dp], "
        "and AND / OR / NOT. One query per call; vary the angle (disease name, synonyms, gene "
        "symbol, phenotype terms, variant notation) to find reports that do not use the current name."
    )
    max_strength = 0.6

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "PubMed query, e.g. 'SAT1[gene] AND lupus'."},
                "max_results": {
                    "type": "integer",
                    "description": "How many papers to return, 1-100 (default 20).",
                },
                "sort": {
                    "type": "string",
                    "enum": ["relevance", "pub_date"],
                    "description": "Order of results (default relevance).",
                },
            },
            "required": ["query", "max_results", "sort"],
            "additionalProperties": False,
        }

    async def run(self, **kwargs: Any) -> ProbeResult:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return ProbeResult(summary="empty query", ok=False)
        n = max(1, min(100, int(kwargs.get("max_results") or 20)))
        sort = "pub_date" if kwargs.get("sort") == "pub_date" else "relevance"
        try:
            raw = json.loads(
                await eutils("esearch.fcgi", db="pubmed", term=query, retmax=n, sort=sort, retmode="json")
            )
            res = raw.get("esearchresult", {})
            papers = await summaries(res.get("idlist", []))
        except Exception as e:  # noqa: BLE001
            return ProbeResult(summary=f"PubMed search failed: {e}", ok=False)
        total = res.get("count", "?")
        return ProbeResult(
            summary=f"{total} PubMed hits for {query!r}; showing {len(papers)}"
            + (f" (PubMed read it as: {res['querytranslation']})" if res.get("querytranslation") else ""),
            evidence=papers,
            limit=12000,
        )


class PubMedFetchProbe(Probe):
    name = "pubmed_fetch"
    description = (
        "Read papers by PMID: full author list, abstract, publication type and MeSH terms. "
        "Use it to confirm that a paper really reports the disease (patients, gene, phenotype) "
        "before listing it, and to pick up names, synonyms and variants for further searches."
    )
    max_strength = 0.7

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pmids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Up to 10 PMIDs.",
                }
            },
            "required": ["pmids"],
            "additionalProperties": False,
        }

    async def run(self, **kwargs: Any) -> ProbeResult:
        pmids = _pmid_list(kwargs.get("pmids"), 10)
        if not pmids:
            return ProbeResult(summary="no PMIDs given", ok=False)
        try:
            papers = parse_efetch(
                await eutils("efetch.fcgi", db="pubmed", id=",".join(pmids), retmode="xml")
            )
        except Exception as e:  # noqa: BLE001
            return ProbeResult(summary=f"PubMed fetch failed: {e}", ok=False)
        for p in papers:
            p["abstract"] = p["abstract"][:2500]
        return ProbeResult(
            summary=f"fetched {len(papers)} of {len(pmids)} PMIDs",
            evidence=papers,
            limit=30000,
        )


class RelatedPapersProbe(Probe):
    name = "related_papers"
    description = (
        "Follow the citation graph from one paper: 'cited_by' (later papers citing it, e.g. "
        "follow-up case reports), 'references' (what it cites, e.g. earlier reports), or "
        "'similar' (PubMed's similar-articles list). Returns PMID, title, year, journal."
    )
    max_strength = 0.6
    _LINK: ClassVar[dict[str, str]] = {
        "cited_by": "pubmed_pubmed_citedin",
        "references": "pubmed_pubmed_refs",
        "similar": "pubmed_pubmed",
    }

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pmid": {"type": "string", "description": "The paper to start from."},
                "relation": {"type": "string", "enum": list(self._LINK)},
                "max_results": {"type": "integer", "description": "1-50 (default 20)."},
            },
            "required": ["pmid", "relation", "max_results"],
            "additionalProperties": False,
        }

    async def run(self, **kwargs: Any) -> ProbeResult:
        pmid = "".join(re.findall(r"\d", str(kwargs.get("pmid", ""))))
        relation = kwargs.get("relation", "similar")
        if not pmid or relation not in self._LINK:
            return ProbeResult(summary="need a PMID and a relation", ok=False)
        n = max(1, min(50, int(kwargs.get("max_results") or 20)))
        try:
            raw = json.loads(
                await eutils(
                    "elink.fcgi",
                    dbfrom="pubmed",
                    db="pubmed",
                    id=pmid,
                    linkname=self._LINK[relation],
                    retmode="json",
                )
            )
            ids: list[str] = []
            for ls in raw.get("linksets", []):
                for db in ls.get("linksetdbs", []):
                    ids += [str(i) for i in db.get("links", [])]
            ids = [i for i in dict.fromkeys(ids) if i != pmid]
            papers = await summaries(ids[:n])
        except Exception as e:  # noqa: BLE001
            return ProbeResult(summary=f"PubMed link lookup failed: {e}", ok=False)
        return ProbeResult(
            summary=f"{len(ids)} '{relation}' links for PMID {pmid}; showing {len(papers)}",
            evidence=papers,
            limit=12000,
        )


# -- benchmark ----------------------------------------------------------------


def parse_pmids(text: str) -> list[str]:
    """PMIDs out of model text, in order, deduplicated.

    Takes every number written as 'PMID 123', 'PMID: 123' or a pubmed URL; if
    there are none, every line that is just a number. Never a bare number inside
    other text, which would catch years, page numbers and DOI fragments.
    """
    tagged = re.findall(
        r"(?:PMID[:\s#]*|pubmed\.ncbi\.nlm\.nih\.gov/|pubmed/)(\d{1,9})", text, flags=re.IGNORECASE
    )
    if not tagged:
        tagged = re.findall(r"^\s*(?:[-*\d.)]+\s+)?(\d{5,9})\s*$", text, flags=re.MULTILINE)
    return list(dict.fromkeys(tagged))


def _field(d: dict[str, Any], *names: str) -> Any:
    for n in names:
        if d.get(n) not in (None, ""):
            return d[n]
    return None


@benchmarks.register("omim")
class Omim(Benchmark):
    name = "omim"
    splits = ("train", "eval")
    n_attempts = 1
    answer_kind = "pmid_list"
    answer_format = (
        "List every paper you found that reports this disease, one per line, each starting with "
        "its PMID, as 'PMID 12345678 - first author, year, title'. List papers you verified, "
        "not ones you only suspect."
    )
    blurb = "OMIM rare-disease literature search over PubMed, recall of the entry's references"

    def load(self, split: str = "eval", limit: int | None = None, seed: int = 0) -> list[Task]:
        d = self.root / _SPLIT_DIRS.get(split, split)
        if not d.exists():
            raise FileNotFoundError(f"{self.name}: no such split directory: {d}")
        tasks: list[Task] = []
        for path in sorted(d.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            for i, test in enumerate(raw["test"]):
                inp = test["input"]
                tid = path.stem if len(raw["test"]) == 1 else f"{path.stem}#{i}"
                tasks.append(
                    Task(
                        id=tid,
                        benchmark=self.name,
                        question=self._render(inp),
                        payload={"input": inp},
                        answer=test.get("output") or [],
                        meta={"file": path.name, "test_index": i},
                    )
                )
        return self.subsample(tasks, limit, seed)

    def _render(self, inp: dict[str, Any]) -> str:
        mim = _field(inp, "phenotype_mim_number", "mim_number", "mim")
        lines = [
            "Find the published literature on this rare disease.",
            "",
            f"Disease: {inp.get('disease', '')}",
            f"Symbol: {inp.get('symbol', '')}",
            f"Phenotype MIM number: {mim}",
        ]
        gene = _field(inp, "gene", "gene_hint")
        if self.opts.get("gene_hint") and gene:
            lines.append(f"Gene: {gene}")
        lines += [
            "",
            "Find every paper that reports patients with this disease or establishes its genetic "
            "cause: the original report(s), later case reports and cohorts, and functional "
            "studies of the disease variants. Papers may describe the disease under an older or "
            "different name, or only by gene and phenotype. Use PubMed; do not consult OMIM.",
        ]
        return "\n".join(lines)

    def parse_answer(self, text: str) -> Any:
        return parse_pmids(text if isinstance(text, str) else "\n".join(map(str, text or [])))

    def probes(self, task: Task) -> list[Probe]:
        """PubMed only. No Python sandbox: there is nothing to compute, and a
        sandbox has network access, which reaches OMIM."""
        return [PubMedSearchProbe(), PubMedFetchProbe(), RelatedPapersProbe()]

    async def score(self, task: Task, pred: Prediction, ctx: RunContext) -> Score:
        gold = [g for g in (task.answer or []) if isinstance(g, dict)]
        found = pred.answer if isinstance(pred.answer, list) else self.parse_answer(pred.answer or "")
        found_set = {str(p) for p in found}
        gold_ids = [str(g.get("pmid") or "").strip() for g in gold]
        matchable = [p for p in gold_ids if p]
        hit = [p for p in matchable if p in found_set]
        recall = len(hit) / len(matchable) if matchable else 0.0
        precision = len(hit) / len(found_set) if found_set else 0.0
        return Score(
            task_id=task.id,
            score=recall,
            metrics={
                "recall": recall,
                "precision": precision,
                "n_found": float(len(found_set)),
                "n_gold": float(len(gold)),
                "n_gold_with_pmid": float(len(matchable)),
                "scored": float(bool(matchable)),
            },
            detail={
                "hits": hit,
                "missed": [p for p in matchable if p not in found_set],
                "gold_without_pmid": [g.get("title", "")[:120] for g in gold if not g.get("pmid")],
                "new": sorted(found_set - set(matchable)),
            },
        )

    def check(self) -> list[str]:
        problems = super().check()
        if problems:
            return problems
        if not any((self.root / _SPLIT_DIRS[s]).exists() for s in self.splits):
            return [f"no evaluation/ or training/ directory under {self.root}"]
        return []
