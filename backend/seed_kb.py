"""
seed_kb.py – Initiales Befüllen der Wissensdatenbank
Führe einmalig aus: python seed_kb.py
"""
import json, hashlib
from pathlib import Path

DOCS = [
  {
    "title": "Datroway (Dato-DXd) FDA Approval – HR+/HER2- Breast Cancer",
    "type": "approval", "date": "2025-01-17", "source": "FDA",
    "tags": ["ADC","TROP2","breast_cancer","DXd","HR_positive","bystander_effect"],
    "content": "Datopotamab deruxtecan (Datroway) FDA-Zulassung 17. Jan 2025 für metastatisches HR+/HER2- Mammakarzinom. TROP2-ADC mit DXd-Payload. KEIN IHC-Cutoff erforderlich – Bystander-Effekt macht breite Anwendung möglich. TROPION-Breast01: PFS 6.9 vs 4.9 Mo (HR 0.63), OS 18.6 vs 18.3 Mo (kein OS-Benefit). ORR 36% vs 23%. Algorithmus-Lektion: DXd-Klasse ermöglicht Zulassung ohne IHC-Schwelle. Priority Review für TNBC 2026 erteilt."
  },
  {
    "title": "Imlunestrant (Inluriyo) FDA Approval – ESR1-mutiertes Brustkrebs",
    "type": "approval", "date": "2025-09-25", "source": "FDA",
    "tags": ["SERD","ESR1","breast_cancer","acquired_resistance","ctDNA","companion_diagnostic"],
    "content": "Imlunestrant (Inluriyo, Eli Lilly) FDA-Zulassung 25. Sep 2025 für ER+/HER2-/ESR1-mutiertes MBC. Oraler brain-penetranter SERD. EMBER-3: PFS 5.5 vs 3.8 Mo (HR 0.62), OS 34.5 vs 23.1 Mo (HR 0.60). Guardant360 CDx als Companion Diagnostic (ctDNA). Algorithmus-Lektion: Acquired Resistance Mutation als Biomarker + Companion Diagnostic = PoS +25-35%. EU-Zulassung 2025 ebenfalls erteilt."
  },
  {
    "title": "Camizestrant SERENA-6 Phase III – Breakthrough Therapy ctDNA-guided Switch",
    "type": "phase3_data", "date": "2025-06-01", "source": "ASCO 2025",
    "tags": ["SERD","ESR1","CDK46","ctDNA","preemptive_switch","breakthrough","AstraZeneca"],
    "content": "Camizestrant (AstraZeneca) SERENA-6: PFS 16.0 vs 9.2 Mo (HR 0.44, p<0.001) bei HR+/HER2- Brustkrebs. Erstes Trial mit ctDNA-guided Therapiewechsel VOR klinischer Progression. FDA Breakthrough Designation Jun 2025. Kompatibel mit Palbociclib, Ribociclib UND Abemaciclib. Neues Paradigma: Sequenz-Therapie gesteuert durch molekulares Monitoring. PoS-Boost für ctDNA+preemptiver Switch: +30-45%."
  },
  {
    "title": "STAR-221 Abbruch – Domvanalimab/Zimberelimab TIGIT Phase III Futilität",
    "type": "phase3_failure", "date": "2025-12-12", "source": "Arcus/Gilead",
    "tags": ["TIGIT","checkpoint","gastric","esophageal","failure","futility","class_failure"],
    "content": "Phase-III STAR-221 (Domvanalimab+Zimberelimab+Chemo vs Nivolumab+Chemo) wegen Futilität abgebrochen Dez 2025. Kein OS-Vorteil in 1040 Patienten. Teil des TIGIT-Klassen-Versagens: Roche (Tiragolumab), Merck (Vibostolimab), GSK (Belrestotug), BMS alle ebenfalls gescheitert. Algorithmus Typ 5: Klassen-Versagen. PoS-Reduktion TIGIT: -30 bis -50% automatisch. AstraZeneca (Rilvegostomig) einziger verbliebener Spieler."
  },
  {
    "title": "Telisotuzumab vedotin (Emrelis) FDA Accelerated Approval – cMET ADC NSCLC",
    "type": "approval", "date": "2025-05-01", "source": "FDA",
    "tags": ["ADC","cMET","NSCLC","IHC","overexpression","MMAE","AbbVie"],
    "content": "Telisotuzumab vedotin (Emrelis) FDA Accelerated Approval Mai 2025: cMET IHC3+ ≥50% NSCLC (EGFR-WT). LUMINOSITY Phase II: ORR 35.3%, DoR 7.2 Mo. Fundamentale Algorithmus-Lektion: IHC-Überexpression die für mAb (Onartuzumab) scheiterte, reicht für ADC aus. ADC benötigt Dichte (Payload-Delivery), kein onkogenes Signal. TeliMET NSCLC-01 Phase III läuft als Confirmatory Trial."
  },
  {
    "title": "Patritumab Deruxtecan (HER3-DXd) FDA Accelerated Approval – EGFR NSCLC",
    "type": "approval", "date": "2025-01-01", "source": "FDA",
    "tags": ["ADC","HER3","NSCLC","EGFR","DXd","resistance","Daiichi_Sankyo","AstraZeneca"],
    "content": "Patritumab deruxtecan FDA Accelerated Approval Jan 2025. HERTHENA-Lung01: ORR 29.8%, DoR 6.4 Mo, OS 11.9 Mo in EGFR-mut. NSCLC nach Platin+EGFR-TKI. HER3 hochreguliert als Resistenzmechanismus nach EGFR-TKI. Kein IHC-Cutoff (Bystander-Effekt DXd). Phase III HERTHENA-Lung02 läuft. Demo-Case PoS 76%. Kreuz-Modalitäts-Validierung: HER3 als ADC-Target etabliert."
  },
  {
    "title": "V940/mRNA-4157 + Pembrolizumab – Personalisierte mRNA-Vakzine 5-Jahres-Daten",
    "type": "phase3_data", "date": "2025-06-01", "source": "ASCO 2025",
    "tags": ["mRNA_vaccine","neoantigen","melanoma","pembrolizumab","TMB","personalized","Moderna","MSD"],
    "content": "V940 (mRNA-4157, Moderna/MSD) KEYNOTE-942 5-Jahres-Update: HR 0.510 (49% Risikoreduktion) hochrisiko Melanom. INTerpath Phase-III-Programm: Melanom (001), NSCLC (002/009), RCC (004). FDA Breakthrough Designation. Algorithmus-Prädiktoren für mRNA-Vakzine: TMB/Neoantigen-Last, HLA-Diversität, CPI-Kombination. Neue Modalitätskategorie mit +20-35% PoS bei erfüllten Kriterien."
  },
  {
    "title": "Onartuzumab Phase III METLung – Aktiver Schaden durch falschen Biomarker",
    "type": "negative_case", "date": "2014-01-01", "source": "Roche/Genentech",
    "tags": ["mAb","cMET","NSCLC","wrong_biomarker","overexpression","failure","type_1_error"],
    "content": "Onartuzumab (Anti-cMET mAb) Phase III METLung: OS 6.8 vs 9.1 Mo (HR 1.27) – AKTIVER SCHADEN. Fehler: MET IHC (Überexpression) als Biomarker für Signal-blockierenden mAb. METex14 Skipping (echter Treiber) nicht getestet. Algorithmus Typ 1+2. PoS retrospektiv: 18%. Kernlektion: Überexpression ≠ onkogene Abhängigkeit für TKI/mAb. Direkter Kontrast zu Telisotuzumab vedotin (ADC, selber Target, richtige Biomarker-Logik)."
  },
  {
    "title": "Bintrafusp alfa Phase III Scheitern – Pleiotropes TGF-beta Target",
    "type": "negative_case", "date": "2021-01-01", "source": "Merck KGaA",
    "tags": ["bifunctional","PD-L1","TGF_beta","NSCLC","pleiotropy","failure","type_3_error"],
    "content": "Bintrafusp alfa (PD-L1+TGF-β Bifunktional) scheiterte Phase III trotz Phase-II-ORR 85.7% bei PD-L1-high. TGF-β hat kontextabhängige Doppelrolle (Tumorsuppressor früh, Tumorförderer spät). Kein prospektiver Biomarker für klinischen Kontext. Algorithmus Typ 3: Pleiotropes Target. PoS-Reduktion: -35 bis -50%. Vergleich: Pumitamig (PD-L1xVEGF) erfüllt Kriterien – beide Targets einzeln validiert, komplementäre Mechanismen."
  },
  {
    "title": "Zongertinib FDA Approval – HER2-mutiertes NSCLC 2025",
    "type": "approval", "date": "2025-01-01", "source": "FDA",
    "tags": ["TKI","HER2","NSCLC","mutation","not_amplification","Boehringer_Ingelheim"],
    "content": "Zongertinib FDA-Zulassung 2025 für HER2-mutiertes NSCLC. Selektiver kovalenter HER2-TKI. Algorithmus-Lektion: HER2-Mutation (Exon 20 Insertion) = onkogene Abhängigkeit → TKI korrekt. HER2-Amplifikation = Überexpression → ADC/mAb korrekt. Dasselbe Target, fundamental unterschiedliche Biomarker je nach Modalität und Tumorentität. PoS-Warnung: HER2-Amplifikation als TKI-Biomarker in NSCLC = ROT -30-40%."
  },
  {
    "title": "Durvalumab + FLOT FDA Approval – Perioperatives Magen-/Ösophaguskarzinom",
    "type": "approval", "date": "2025-11-01", "source": "FDA",
    "tags": ["checkpoint","PD-L1","FLOT","gastric","esophageal","perioperative","neoadjuvant","AstraZeneca"],
    "content": "Durvalumab (Imfinzi) + FLOT FDA Nov 2025 als neoadjuvante+adjuvante Therapie für resektables Magenkarzinom. Neues perioperatives CPI-Paradigma. Algorithmus-Lektion: CPI neoadjuvant+adjuvant verlängert Immunaktivierung, eliminiert MRD. pCR-Rate als Surrogatendpunkt validiert. Kontrastiert direkt mit STAR-221-Scheitern im metastatischen Setting derselben Tumorentität."
  },
  {
    "title": "Nirogacestat (Ogsiveo) EU Approval – Gamma-Secretase-Inhibitor Desmoid",
    "type": "approval", "date": "2025-08-01", "source": "EMA",
    "tags": ["gamma_secretase","Notch","desmoid","rare_disease","drug_repurposing","orphan"],
    "content": "Nirogacestat EU-Zulassung Aug 2025 (FDA Nov 2023). Erste Therapie spezifisch für Desmoid-Tumoren. Drug Repurposing: Ursprünglich Alzheimer-Kandidat. DeFi Phase III: 71% Progressionsrisiko-Reduktion (HR 0.29), ORR 41%. Algorithmus-Lektion: Failed Candidate in einer Indikation = Success in anderer mit richtiger Pathway-Abhängigkeit. Notch-Signalweg als Treiber in Desmoid identifiziert. Breakthrough + Orphan Drug = beschleunigter Zulassungsweg."
  },
  {
    "title": "Sotorasib + Panitumumab FDA Approval – KRAS G12C mCRC Kombinationsstrategie",
    "type": "approval", "date": "2025-01-01", "source": "FDA",
    "tags": ["KRAS_G12C","TKI","anti-EGFR","colorectal","combination","feedback_resistance","Amgen"],
    "content": "Sotorasib + Panitumumab FDA-Zulassung 2025 für KRAS G12C-mutiertes mCRC. Algorithmus-Lektion: KRAS G12C-Inhibitor zeigt in CRC schlechtere Mono-Daten als in NSCLC wegen EGFR-Feedback-Aktivierung als Resistenzmechanismus. Lösung: EGFR-Blockade mit Panitumumab eliminiert Feedback. Prädiktor 12: Bekannter Feedback-Resistenzmechanismus + validierter Kombinationspartner = PoS +15-25%."
  },
  {
    "title": "FAPi Radioligand Therapie – Pan-Tumor Stroma-Targeting Pipeline 2026",
    "type": "pipeline", "date": "2026-01-01", "source": "Multiple",
    "tags": ["RLT","FAPi","FAP","stroma","CAF","pan_tumor","theranostics","heterogeneity"],
    "content": "FAPi RLT: FAP exprimiert auf CAFs in >90% epithelialer Malignome. Kandidaten: FAPI-46, EB-FAPI, SA.FAPi, FAP-2286, FAPI-C16 (Albumin-Binder für verlängerte Retention). FAPi-PET klinisch etabliert. Hauptrisiko: FAP-Heterogenität inter- und intraläsional. Algorithmus: FAPi-PET als Pflicht-Selektor. Ohne PET: PoS -25-35%. Kurze Tumor-Retention bei frühen FAPi-04/46 limitiert therapeutisches Fenster → Albumin-Binder-Strategie."
  },
  {
    "title": "Revumenib (Komzifti) FDA Approval – Menin-Inhibitor AML",
    "type": "approval", "date": "2024-01-01", "source": "FDA",
    "tags": ["menin","epigenetic","AML","NPM1","KMT2A","Syndax"],
    "content": "Revumenib (Komzifti, Syndax) FDA-Zulassung für R/R AML mit NPM1-Mutation oder KMT2A-Fusion. Neuer Biomarker-Typ: Epigenetische Abhängigkeit – upstream Mutation (NPM1/KMT2Ar) erzeugt downstream Vulnerabilität des Menin-Komplexes. Nicht Überexpression, nicht Treibermutation des Targets selbst. Algorithmus Prädiktor 11: Upstream-Mutation/Fusion als Biomarker für epigenetischen Regulator = PoS +20-30%."
  },
  {
    "title": "Pirtobrutinib (Jaypirca) Regular Approval – Non-kovalenter BTK nach C481S-Resistenz",
    "type": "approval", "date": "2025-12-01", "source": "FDA",
    "tags": ["BTK","TKI","CLL","SLL","resistance_mutation","C481S","non_covalent","Lilly"],
    "content": "Pirtobrutinib (Jaypirca) Reguläre FDA-Zulassung Dez 2025 für R/R CLL/SLL. C481S-Mutation macht kovalente BTK-Inhibitoren (Ibrutinib, Acalabrutinib) wirkungslos. Pirtobrutinib bindet nicht-kovalent und ist C481S-unabhängig. Algorithmus Prädiktor 13: Bekannte Resistenzmutation als Selektionsbiomarker für Next-Gen = PoS +20-30%. Marktpotenzial: Gesamte kovalente BTK-resistente Population ist Zielpopulation."
  }
]

def seed():
    kb_path = Path("knowledge_base.json")
    existing = json.loads(kb_path.read_text()) if kb_path.exists() else []
    existing_ids = {d["id"] for d in existing}

    added = 0
    for doc in DOCS:
        doc_id = "kb_" + hashlib.md5(f"{doc['title']}{doc['date']}".encode()).hexdigest()[:8]
        if doc_id not in existing_ids:
            existing.append({"id": doc_id, **doc})
            added += 1

    kb_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"✅ Wissensdatenbank: {added} neue Einträge hinzugefügt. Gesamt: {len(existing)}")

    # Optional: rebuild embeddings if sentence-transformers available
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        print("🔄 Erstelle Embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [f"{d['title']} {d['content']}" for d in existing]
        matrix = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        np.save("embeddings.npy", matrix)
        Path("embed_meta.json").write_text(json.dumps([d["id"] for d in existing]))
        print(f"✅ {len(existing)} Embeddings erstellt.")
    except ImportError:
        print("ℹ️ sentence-transformers nicht installiert → Keyword-Fallback aktiv")

if __name__ == "__main__":
    seed()
