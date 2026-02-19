"""
fiercepharma_ingestor.py — Automatischer FiercePharma News Scraper

Läuft täglich via Cron/Docker und importiert:
- FDA Approvals
- Phase III Ergebnisse
- Pipeline Updates
- Klinische Trial-Stopps

Verwendung:
  python fiercepharma_ingestor.py --dry-run      # Preview ohne Import
  python fiercepharma_ingestor.py --ingest       # Importiert in KB
"""

import re, hashlib, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import argparse

try:
    import httpx
    from bs4 import BeautifulSoup
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("⚠ httpx oder beautifulsoup4 nicht installiert:")
    print("  pip install httpx beautifulsoup4 lxml --break-system-packages")

# ═══════════════════════════════════════════════════════════════════
# FIERCEPHARMA RSS FEEDS
# ═══════════════════════════════════════════════════════════════════
RSS_FEEDS = {
    "oncology": "https://www.fiercepharma.com/tag/oncology/feed",
    "fda":      "https://www.fiercepharma.com/tag/fda-approvals/feed",
    "pipeline": "https://www.fiercepharma.com/tag/pipeline-watch/feed",
    "clinical": "https://www.fiercepharma.com/tag/clinical-trials/feed",
}

# Relevanz-Keywords für Onkologie (case-insensitive)
ONCOLOGY_KEYWORDS = [
    "cancer", "oncology", "tumor", "carcinoma", "melanoma", "leukemia", "lymphoma",
    "myeloma", "sarcoma", "glioblastoma", "mesothelioma", "breast cancer", 
    "lung cancer", "NSCLC", "colorectal", "prostate", "ovarian", "pancreatic",
    "checkpoint inhibitor", "ADC", "antibody-drug conjugate", "CAR-T", "TKI",
    "PARP inhibitor", "BTK inhibitor", "CDK4/6", "HER2", "KRAS", "EGFR", "PD-1", "PD-L1",
    "phase 3", "phase III", "FDA approval", "accelerated approval", "breakthrough therapy",
]

# Negativfall-Keywords
NEGATIVE_KEYWORDS = [
    "discontinued", "stopped", "failed", "futility", "miss", "setback",
    "halted", "withdrawn", "rejected", "CRL", "complete response letter",
]


# ═══════════════════════════════════════════════════════════════════
# ARTICLE PARSER
# ═══════════════════════════════════════════════════════════════════
def parse_rss_feed(feed_url: str) -> list[dict]:
    """Parse RSS feed and return list of articles."""
    if not HAS_DEPS:
        return []
    
    try:
        r = httpx.get(feed_url, timeout=15, follow_redirects=True)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        
        articles = []
        for item in soup.find_all("item"):
            title = item.find("title")
            link = item.find("link")
            pub_date = item.find("pubDate")
            desc = item.find("description")
            
            if not all([title, link, pub_date]):
                continue
                
            # Parse date
            date_str = pub_date.text.strip()
            try:
                dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
                date = dt.strftime("%Y-%m-%d")
            except:
                date = datetime.now().strftime("%Y-%m-%d")
            
            articles.append({
                "title": title.text.strip(),
                "url": link.text.strip(),
                "date": date,
                "description": desc.text.strip() if desc else "",
            })
        
        return articles
    except Exception as e:
        print(f"✗ Fehler beim Parsen von {feed_url}: {e}")
        return []


def fetch_article_content(url: str) -> Optional[str]:
    """Fetch full article content from URL."""
    if not HAS_DEPS:
        return None
    
    try:
        r = httpx.get(url, timeout=15, follow_redirects=True)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        # FiercePharma article content is typically in <div class="article-body">
        article = soup.find("div", class_="article-body")
        if not article:
            article = soup.find("article")
        
        if article:
            # Remove scripts, styles, ads
            for tag in article.find_all(["script", "style", "aside", "nav"]):
                tag.decompose()
            
            # Extract text
            paragraphs = [p.get_text(strip=True) for p in article.find_all("p")]
            content = " ".join(paragraphs)
            return content[:2000]  # Limit to 2000 chars
        
        return None
    except Exception as e:
        print(f"  ⚠ Artikel-Content nicht abrufbar: {e}")
        return None


def is_oncology_relevant(text: str) -> bool:
    """Check if article is oncology-relevant."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in ONCOLOGY_KEYWORDS)


def detect_article_type(title: str, content: str) -> str:
    """Detect article type: approval, phase3_failure, phase3_data, or pipeline."""
    text = (title + " " + content).lower()
    
    if any(neg in text for neg in NEGATIVE_KEYWORDS):
        if "phase 3" in text or "phase iii" in text:
            return "phase3_failure"
        return "negative_case"
    
    if "fda" in text and ("approval" in text or "cleared" in text or "authorized" in text):
        return "approval"
    
    if "phase 3" in text or "phase iii" in text:
        if "data" in text or "results" in text or "trial" in text:
            return "phase3_data"
    
    return "pipeline"


def extract_tags(title: str, content: str) -> list[str]:
    """Extract relevant tags from article."""
    text = (title + " " + content).lower()
    tags = set()
    
    # Modalitäten
    modalitäten = {
        "ADC": ["adc", "antibody-drug conjugate", "antibody drug conjugate"],
        "TKI": ["tki", "kinase inhibitor", "tyrosine kinase"],
        "checkpoint": ["checkpoint", "pd-1", "pd-l1", "ctla-4", "lag-3", "tigit"],
        "CAR_T": ["car-t", "car t", "cellular therapy"],
        "BiTE": ["bite", "bispecific"],
        "RLT": ["radioligand", "radiopharmaceutical", "lutetium"],
        "mRNA_vaccine": ["mrna vaccine", "personalized vaccine", "neoantigen"],
    }
    
    for tag, patterns in modalitäten.items():
        if any(p in text for p in patterns):
            tags.add(tag)
    
    # Indikationen
    indikationen = {
        "NSCLC": ["nsclc", "non-small cell lung", "lung cancer"],
        "breast_cancer": ["breast cancer", "mammakarzinom"],
        "colorectal": ["colorectal", "colon cancer", "crc"],
        "melanoma": ["melanoma"],
        "prostate": ["prostate cancer"],
        "pancreatic": ["pancreatic cancer"],
    }
    
    for tag, patterns in indikationen.items():
        if any(p in text for p in patterns):
            tags.add(tag)
    
    # Targets
    targets = ["HER2", "EGFR", "KRAS", "BRAF", "PD-L1", "TROP2", "cMET", "BTK", "CDK46"]
    for target in targets:
        if target.lower() in text:
            tags.add(target)
    
    return sorted(list(tags))


# ═══════════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════════
def ingest_to_kb(article: dict, content: str, dry_run: bool = False):
    """Add article to knowledge base."""
    import json
    
    kb_path = Path("knowledge_base.json")
    kb = json.loads(kb_path.read_text()) if kb_path.exists() else []
    
    # Generate unique ID
    doc_id = "kb_" + hashlib.md5(
        f"{article['title']}{article['date']}".encode()
    ).hexdigest()[:8]
    
    # Check if already exists
    if any(d["id"] == doc_id for d in kb):
        print(f"  ⊘ Bereits in KB: {article['title'][:60]}")
        return False
    
    # Detect type and tags
    article_type = detect_article_type(article["title"], content)
    tags = extract_tags(article["title"], content)
    
    # Extract source (company name if present)
    source_match = re.search(r"(Pfizer|Roche|Novartis|AstraZeneca|Merck|BMS|J&J|Lilly|GSK|Gilead|AbbVie|Amgen|Daiichi|Moderna)", 
                            article["title"] + " " + content, re.IGNORECASE)
    source = source_match.group(1) if source_match else "FiercePharma"
    
    doc = {
        "id": doc_id,
        "title": article["title"],
        "type": article_type,
        "date": article["date"],
        "source": source,
        "tags": tags,
        "content": content,
        "url": article["url"],
    }
    
    if dry_run:
        print(f"  [DRY RUN] Würde hinzufügen:")
        print(f"    Typ: {article_type} | Tags: {', '.join(tags[:5])}")
        return True
    
    kb.append(doc)
    kb_path.write_text(json.dumps(kb, ensure_ascii=False, indent=2))
    
    # Rebuild embeddings
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        print(f"  ↻ Erstelle Embedding...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [f"{d['title']} {d['content']}" for d in kb]
        matrix = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        np.save("embeddings.npy", matrix)
        Path("embed_meta.json").write_text(json.dumps([d["id"] for d in kb]))
        print(f"  ✓ Embedding erstellt")
    except ImportError:
        print(f"  ⚠ sentence-transformers nicht installiert — nur Text-KB aktualisiert")
    
    return True


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="FiercePharma Onkologie-News Scraper")
    parser.add_argument("--dry-run", action="store_true", help="Preview ohne Import")
    parser.add_argument("--days", type=int, default=7, help="Artikel der letzten N Tage (default: 7)")
    parser.add_argument("--feeds", nargs="+", choices=list(RSS_FEEDS.keys()), 
                       default=["oncology", "fda"], help="Welche Feeds scrapen")
    args = parser.parse_args()
    
    if not HAS_DEPS:
        print("✗ Dependencies fehlen. Installation:")
        print("  pip install httpx beautifulsoup4 lxml --break-system-packages")
        return
    
    print(f"{'='*70}")
    print(f"FiercePharma Onkologie-Scraper")
    print(f"{'='*70}")
    print(f"Feeds: {', '.join(args.feeds)}")
    print(f"Zeitraum: Letzte {args.days} Tage")
    print(f"Modus: {'DRY RUN (kein Import)' if args.dry_run else 'IMPORT AKTIV'}")
    print()
    
    cutoff_date = datetime.now() - timedelta(days=args.days)
    total_added = 0
    total_processed = 0
    
    for feed_name in args.feeds:
        feed_url = RSS_FEEDS[feed_name]
        print(f"🔄 Verarbeite Feed: {feed_name}")
        
        articles = parse_rss_feed(feed_url)
        print(f"  Gefunden: {len(articles)} Artikel")
        
        for article in articles:
            # Filter by date
            try:
                article_date = datetime.strptime(article["date"], "%Y-%m-%d")
                if article_date < cutoff_date:
                    continue
            except:
                pass
            
            # Check oncology relevance
            if not is_oncology_relevant(article["title"] + " " + article["description"]):
                continue
            
            total_processed += 1
            print(f"\n  📄 {article['title'][:70]}...")
            print(f"     {article['date']} | {article['url']}")
            
            # Fetch full content
            content = fetch_article_content(article["url"])
            if not content:
                content = article["description"]
            
            # Ingest
            if ingest_to_kb(article, content, dry_run=args.dry_run):
                total_added += 1
                print(f"  ✓ {'(Dry-Run)' if args.dry_run else 'Hinzugefügt'}")
            
            time.sleep(1)  # Rate limiting
    
    print(f"\n{'='*70}")
    print(f"✓ Fertig")
    print(f"  Verarbeitet: {total_processed} onkologie-relevante Artikel")
    print(f"  Neu hinzugefügt: {total_added}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
