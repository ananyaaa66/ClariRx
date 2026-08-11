"""
Medical Knowledge Base for ClariRx
=====================================

Provides curated, verified medical information for grounding LLM
explanations. Prevents hallucination by supplying factual drug info
and lab test reference data.

Two data stores:
  - Drug KB: ~80 common Indian prescription drugs with generic names,
    drug class, uses, side effects, and warnings
  - Lab KB:  ~30 common lab tests with reference ranges, what they
    measure, and what abnormal values mean

All data is hardcoded (not scraped) for accuracy and auditability.

Usage:
    # Build and export JSON files
    python build_kb.py

    # As a module
    from knowledge_base.build_kb import MedicalKB
    kb = MedicalKB()
    info = kb.lookup_drug("Amoxicillin")
    info = kb.lookup_lab_test("Haemoglobin")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DRUG_KB_PATH = SCRIPT_DIR / "drug_kb.json"
LAB_KB_PATH = SCRIPT_DIR / "lab_kb.json"


# ─── Drug Knowledge Base ──────────────────────────────────────────────────────

DRUG_DATABASE: List[Dict] = [
    # ── Antibiotics ────────────────────────────────────────────────────────
    {
        "generic_name": "Amoxicillin",
        "brand_names": ["Amoxil", "Mox", "Novamox", "Amoxicillin"],
        "drug_class": "Antibiotic (Penicillin)",
        "use": "Treats bacterial infections of the ear, nose, throat, urinary tract, and skin.",
        "common_dosages": ["250mg", "500mg"],
        "side_effects": ["Nausea", "Diarrhea", "Skin rash", "Stomach upset"],
        "warnings": ["Tell your doctor if you are allergic to penicillin.", "Complete the full course even if you feel better."],
    },
    {
        "generic_name": "Azithromycin",
        "brand_names": ["Azithral", "Zithromax", "Azee", "Azithrocin", "Azyth", "Zithrin", "Rozith"],
        "drug_class": "Antibiotic (Macrolide)",
        "use": "Treats bacterial infections including respiratory infections, ear infections, and skin infections.",
        "common_dosages": ["250mg", "500mg"],
        "side_effects": ["Nausea", "Diarrhea", "Stomach pain", "Headache"],
        "warnings": ["Take on an empty stomach or 1 hour before meals.", "Do not take antacids within 2 hours."],
    },
    {
        "generic_name": "Ciprofloxacin",
        "brand_names": ["Cipro", "Ciplox", "Cifran"],
        "drug_class": "Antibiotic (Fluoroquinolone)",
        "use": "Treats urinary tract infections, respiratory infections, and gastrointestinal infections.",
        "common_dosages": ["250mg", "500mg"],
        "side_effects": ["Nausea", "Diarrhea", "Dizziness", "Tendon pain"],
        "warnings": ["Avoid excessive sun exposure.", "Drink plenty of water.", "May cause tendon problems in rare cases."],
    },
    {
        "generic_name": "Levofloxacin",
        "brand_names": ["Levaquin", "Levoflox", "Glevo"],
        "drug_class": "Antibiotic (Fluoroquinolone)",
        "use": "Treats pneumonia, sinusitis, urinary tract infections, and skin infections.",
        "common_dosages": ["250mg", "500mg", "750mg"],
        "side_effects": ["Nausea", "Headache", "Diarrhea", "Dizziness"],
        "warnings": ["Avoid sun exposure.", "May cause tendon problems.", "Take with plenty of fluids."],
    },
    {
        "generic_name": "Doxycycline",
        "brand_names": ["Doxy", "Doxt", "Doxycycline"],
        "drug_class": "Antibiotic (Tetracycline)",
        "use": "Treats acne, respiratory infections, malaria prevention, and tick-borne diseases.",
        "common_dosages": ["100mg"],
        "side_effects": ["Sun sensitivity", "Nausea", "Stomach upset"],
        "warnings": ["Avoid sun exposure.", "Do not lie down for 30 min after taking.", "Do not take with milk or antacids."],
    },
    {
        "generic_name": "Cefixime",
        "brand_names": ["Suprax", "Cefix", "Taxim-O"],
        "drug_class": "Antibiotic (Cephalosporin)",
        "use": "Treats ear infections, urinary tract infections, throat infections, and gonorrhea.",
        "common_dosages": ["200mg", "400mg"],
        "side_effects": ["Diarrhea", "Stomach pain", "Nausea"],
        "warnings": ["Complete the full course.", "Tell doctor if allergic to penicillin."],
    },
    {
        "generic_name": "Cephalexin",
        "brand_names": ["Keflex", "Cephalexin", "Sporidex"],
        "drug_class": "Antibiotic (Cephalosporin)",
        "use": "Treats skin infections, bone infections, respiratory tract infections, and urinary tract infections.",
        "common_dosages": ["250mg", "500mg"],
        "side_effects": ["Diarrhea", "Nausea", "Stomach upset", "Rash"],
        "warnings": ["Complete the full course.", "Inform doctor if allergic to penicillin."],
    },
    {
        "generic_name": "Metronidazole",
        "brand_names": ["Flagyl", "Metro", "Metrogyl", "Nidazyl"],
        "drug_class": "Antibiotic / Antiprotozoal",
        "use": "Treats infections caused by bacteria and parasites, including dental, stomach, and vaginal infections.",
        "common_dosages": ["200mg", "400mg"],
        "side_effects": ["Metallic taste", "Nausea", "Dark urine", "Headache"],
        "warnings": ["Do NOT drink alcohol while taking this medicine — causes severe nausea and vomiting.", "Complete the full course."],
    },
    {
        "generic_name": "Ofloxacin",
        "brand_names": ["Oflox", "Zanocin", "Ofloxacin"],
        "drug_class": "Antibiotic (Fluoroquinolone)",
        "use": "Treats urinary tract infections, respiratory infections, and skin infections.",
        "common_dosages": ["200mg", "400mg"],
        "side_effects": ["Nausea", "Headache", "Dizziness", "Diarrhea"],
        "warnings": ["Avoid sun exposure.", "Drink plenty of water."],
    },
    {
        "generic_name": "Clindamycin",
        "brand_names": ["Cleocin", "Dalacin", "Clindamycin"],
        "drug_class": "Antibiotic (Lincosamide)",
        "use": "Treats serious bacterial infections of the skin, lungs, blood, and internal organs.",
        "common_dosages": ["150mg", "300mg"],
        "side_effects": ["Diarrhea", "Nausea", "Stomach pain", "Rash"],
        "warnings": ["Stop and call doctor if you develop severe diarrhea.", "Complete the full course."],
    },
    {
        "generic_name": "Fluconazole",
        "brand_names": ["Diflucan", "Flucos", "Diflu", "Fluconazole"],
        "drug_class": "Antifungal",
        "use": "Treats fungal infections including yeast infections, oral thrush, and skin fungal infections.",
        "common_dosages": ["150mg", "200mg"],
        "side_effects": ["Headache", "Nausea", "Stomach pain", "Diarrhea"],
        "warnings": ["May interact with many other medicines.", "Tell doctor about all medicines you take."],
    },

    # ── Pain & Fever ───────────────────────────────────────────────────────
    {
        "generic_name": "Paracetamol",
        "brand_names": ["Crocin", "Dolo", "Napa", "Napa Extend", "Paracetamol", "Ace", "Tylenol"],
        "drug_class": "Analgesic / Antipyretic",
        "use": "Relieves mild to moderate pain and reduces fever. Used for headaches, body aches, toothaches, and cold/flu symptoms.",
        "common_dosages": ["500mg", "650mg"],
        "side_effects": ["Rarely causes side effects at normal doses"],
        "warnings": ["Do not exceed 4g (4000mg) per day.", "Avoid alcohol while taking this.", "Overdose can cause serious liver damage."],
    },
    {
        "generic_name": "Ibuprofen",
        "brand_names": ["Brufen", "Ibugesic", "Advil", "Ibuprofen"],
        "drug_class": "NSAID (Anti-inflammatory Painkiller)",
        "use": "Reduces pain, inflammation, and fever. Used for headaches, muscle pain, arthritis, and menstrual cramps.",
        "common_dosages": ["200mg", "400mg"],
        "side_effects": ["Stomach upset", "Heartburn", "Nausea", "Dizziness"],
        "warnings": ["Take with food to protect stomach.", "Not suitable for people with stomach ulcers.", "Avoid in last 3 months of pregnancy."],
    },
    {
        "generic_name": "Diclofenac",
        "brand_names": ["Voveran", "Voltaren", "Diclofenac"],
        "drug_class": "NSAID (Anti-inflammatory Painkiller)",
        "use": "Treats pain, swelling, and inflammation from arthritis, sprains, and injuries.",
        "common_dosages": ["50mg"],
        "side_effects": ["Stomach pain", "Nausea", "Headache", "Dizziness"],
        "warnings": ["Take with food.", "Not for stomach ulcer patients.", "Use for shortest duration needed."],
    },
    {
        "generic_name": "Aceclofenac",
        "brand_names": ["Hifenac", "Zerodol", "Aceclofenac"],
        "drug_class": "NSAID (Anti-inflammatory Painkiller)",
        "use": "Treats pain and inflammation from arthritis, dental pain, and musculoskeletal injuries.",
        "common_dosages": ["100mg"],
        "side_effects": ["Stomach upset", "Nausea", "Diarrhea", "Headache"],
        "warnings": ["Take with food.", "Avoid if you have stomach ulcers."],
    },
    {
        "generic_name": "Tramadol",
        "brand_names": ["Ultram", "Contramal", "Tramadol"],
        "drug_class": "Opioid Analgesic",
        "use": "Treats moderate to moderately severe pain.",
        "common_dosages": ["50mg", "100mg"],
        "side_effects": ["Drowsiness", "Nausea", "Constipation", "Dizziness"],
        "warnings": ["May cause drowsiness — do not drive.", "Can be habit-forming.", "Do not take with alcohol."],
    },

    # ── Stomach / Acid ─────────────────────────────────────────────────────
    {
        "generic_name": "Omeprazole",
        "brand_names": ["Prilosec", "Omez", "Omastin", "Omeprazole"],
        "drug_class": "Proton Pump Inhibitor (PPI)",
        "use": "Reduces stomach acid. Treats heartburn, acid reflux (GERD), and stomach ulcers.",
        "common_dosages": ["20mg", "40mg"],
        "side_effects": ["Headache", "Stomach pain", "Nausea", "Gas"],
        "warnings": ["Take 30 minutes before a meal.", "Long-term use may affect calcium absorption."],
    },
    {
        "generic_name": "Pantoprazole",
        "brand_names": ["Pantop", "Pan", "Pantocid", "Pantoprazole"],
        "drug_class": "Proton Pump Inhibitor (PPI)",
        "use": "Reduces stomach acid. Treats GERD, acid reflux, and stomach ulcers.",
        "common_dosages": ["20mg", "40mg"],
        "side_effects": ["Headache", "Diarrhea", "Stomach pain"],
        "warnings": ["Take before breakfast.", "Do not crush or chew the tablet."],
    },
    {
        "generic_name": "Rabeprazole",
        "brand_names": ["Razo", "Aciphex", "Rabeprazole"],
        "drug_class": "Proton Pump Inhibitor (PPI)",
        "use": "Reduces stomach acid production. Treats GERD, peptic ulcers, and Zollinger-Ellison syndrome.",
        "common_dosages": ["20mg"],
        "side_effects": ["Headache", "Diarrhea", "Nausea"],
        "warnings": ["Take before meals.", "Not for long-term use without doctor supervision."],
    },
    {
        "generic_name": "Esomeprazole",
        "brand_names": ["Nexium", "Esoral", "Nexcap", "Sergel", "Esonix", "Esomeprazole"],
        "drug_class": "Proton Pump Inhibitor (PPI)",
        "use": "Reduces stomach acid. Treats GERD, erosive esophagitis, and stomach ulcers.",
        "common_dosages": ["20mg", "40mg"],
        "side_effects": ["Headache", "Nausea", "Stomach pain", "Flatulence"],
        "warnings": ["Take 1 hour before a meal.", "Swallow whole, do not chew."],
    },
    {
        "generic_name": "Ranitidine",
        "brand_names": ["Zantac", "Rantac", "Ranitidine"],
        "drug_class": "H2 Receptor Antagonist",
        "use": "Reduces stomach acid. Treats heartburn, acid reflux, and stomach ulcers.",
        "common_dosages": ["150mg", "300mg"],
        "side_effects": ["Headache", "Constipation", "Diarrhea"],
        "warnings": ["Some formulations were recalled due to impurities — use only prescribed brands."],
    },
    {
        "generic_name": "Domperidone",
        "brand_names": ["Motilium", "Domstal", "Domperidone"],
        "drug_class": "Antiemetic / Prokinetic",
        "use": "Relieves nausea, vomiting, and bloating. Helps food move through the stomach.",
        "common_dosages": ["10mg"],
        "side_effects": ["Headache", "Dry mouth", "Stomach cramps"],
        "warnings": ["Take before meals.", "Avoid in patients with heart rhythm problems."],
    },
    {
        "generic_name": "Ondansetron",
        "brand_names": ["Zofran", "Emeset", "Ondansetron"],
        "drug_class": "Antiemetic",
        "use": "Prevents nausea and vomiting caused by surgery, chemotherapy, or medications.",
        "common_dosages": ["4mg", "8mg"],
        "side_effects": ["Headache", "Constipation", "Dizziness"],
        "warnings": ["May cause QT prolongation in rare cases.", "Inform doctor of heart conditions."],
    },

    # ── Allergy ────────────────────────────────────────────────────────────
    {
        "generic_name": "Cetirizine",
        "brand_names": ["Zyrtec", "Cetzine", "Cetisoft", "Alatrol", "Cetirizine"],
        "drug_class": "Antihistamine",
        "use": "Treats allergies — runny nose, sneezing, itchy/watery eyes, and hives.",
        "common_dosages": ["5mg", "10mg"],
        "side_effects": ["Drowsiness", "Dry mouth", "Headache"],
        "warnings": ["May cause drowsiness — be careful while driving.", "Avoid alcohol."],
    },
    {
        "generic_name": "Levocetirizine",
        "brand_names": ["Xyzal", "Levocet", "Levocetirizine"],
        "drug_class": "Antihistamine",
        "use": "Treats allergies — sneezing, runny nose, itching, and hives. Less drowsy than cetirizine.",
        "common_dosages": ["5mg"],
        "side_effects": ["Mild drowsiness", "Dry mouth", "Headache"],
        "warnings": ["Take in the evening.", "Less drowsy but still be careful driving."],
    },
    {
        "generic_name": "Fexofenadine",
        "brand_names": ["Allegra", "Telfast", "Fexofast", "Fenadin", "Fexo", "Fexofenadine"],
        "drug_class": "Antihistamine (Non-drowsy)",
        "use": "Treats seasonal allergies and chronic hives without causing drowsiness.",
        "common_dosages": ["60mg", "120mg", "180mg"],
        "side_effects": ["Headache", "Nausea", "Dizziness"],
        "warnings": ["Do not take with fruit juices (reduces absorption).", "Non-drowsy — safe for daytime use."],
    },
    {
        "generic_name": "Chlorpheniramine",
        "brand_names": ["Piriton", "CTM", "Chlorpheniramine"],
        "drug_class": "Antihistamine (Sedating)",
        "use": "Treats allergy symptoms and common cold — sneezing, runny nose, watery eyes.",
        "common_dosages": ["4mg"],
        "side_effects": ["Drowsiness", "Dry mouth", "Blurred vision"],
        "warnings": ["Causes strong drowsiness — do not drive.", "Avoid alcohol."],
    },
    {
        "generic_name": "Montelukast",
        "brand_names": ["Singulair", "Montair", "Montene", "Montex", "M-Kast", "Odmon", "Montelukast"],
        "drug_class": "Leukotriene Receptor Antagonist",
        "use": "Prevents asthma attacks and treats seasonal allergies. Reduces airway inflammation.",
        "common_dosages": ["4mg", "5mg", "10mg"],
        "side_effects": ["Headache", "Stomach pain", "Fatigue"],
        "warnings": ["Take in the evening.", "Not for acute asthma attacks — use inhaler for emergencies."],
    },

    # ── Diabetes ───────────────────────────────────────────────────────────
    {
        "generic_name": "Metformin",
        "brand_names": ["Glucophage", "Glycomet", "Metformin"],
        "drug_class": "Antidiabetic (Biguanide)",
        "use": "Controls blood sugar levels in Type 2 diabetes. Helps body use insulin more effectively.",
        "common_dosages": ["500mg", "850mg", "1000mg"],
        "side_effects": ["Nausea", "Diarrhea", "Stomach upset", "Metallic taste"],
        "warnings": ["Take with food to reduce stomach upset.", "Do not drink alcohol excessively.", "Inform doctor before any surgery or CT scan."],
    },

    # ── Heart / Blood Pressure ─────────────────────────────────────────────
    {
        "generic_name": "Amlodipine",
        "brand_names": ["Norvasc", "Amlong", "Amlodipine"],
        "drug_class": "Calcium Channel Blocker",
        "use": "Lowers blood pressure and treats chest pain (angina). Relaxes blood vessels.",
        "common_dosages": ["2.5mg", "5mg", "10mg"],
        "side_effects": ["Swollen ankles", "Headache", "Flushing", "Dizziness"],
        "warnings": ["Do not stop suddenly.", "Avoid grapefruit juice.", "Take at the same time each day."],
    },
    {
        "generic_name": "Telmisartan",
        "brand_names": ["Micardis", "Telma", "Telmisartan"],
        "drug_class": "ARB (Angiotensin II Receptor Blocker)",
        "use": "Lowers blood pressure and protects kidneys in diabetic patients.",
        "common_dosages": ["20mg", "40mg", "80mg"],
        "side_effects": ["Dizziness", "Back pain", "Diarrhea"],
        "warnings": ["Do not use during pregnancy.", "May cause dizziness — stand up slowly."],
    },
    {
        "generic_name": "Losartan",
        "brand_names": ["Cozaar", "Losacar", "Losartan"],
        "drug_class": "ARB (Angiotensin II Receptor Blocker)",
        "use": "Lowers blood pressure and protects kidneys. Also used after heart attacks.",
        "common_dosages": ["25mg", "50mg", "100mg"],
        "side_effects": ["Dizziness", "Fatigue", "Low blood pressure"],
        "warnings": ["Do not use during pregnancy.", "Stay hydrated."],
    },
    {
        "generic_name": "Olmesartan",
        "brand_names": ["Benicar", "Olmy", "Olmesartan"],
        "drug_class": "ARB (Angiotensin II Receptor Blocker)",
        "use": "Lowers high blood pressure by relaxing blood vessels.",
        "common_dosages": ["20mg", "40mg"],
        "side_effects": ["Dizziness", "Diarrhea", "Joint pain"],
        "warnings": ["Do not use during pregnancy.", "Take at the same time daily."],
    },
    {
        "generic_name": "Atorvastatin",
        "brand_names": ["Lipitor", "Atorva", "Atorvastatin"],
        "drug_class": "Statin (Cholesterol Lowering)",
        "use": "Lowers bad cholesterol (LDL) and triglycerides. Reduces risk of heart attack and stroke.",
        "common_dosages": ["10mg", "20mg", "40mg"],
        "side_effects": ["Muscle pain", "Headache", "Nausea", "Joint pain"],
        "warnings": ["Take at bedtime.", "Report unexplained muscle pain to your doctor.", "Avoid grapefruit."],
    },
    {
        "generic_name": "Rosuvastatin",
        "brand_names": ["Crestor", "Rosuvas", "Rosuvastatin"],
        "drug_class": "Statin (Cholesterol Lowering)",
        "use": "Lowers cholesterol and reduces risk of heart disease and stroke.",
        "common_dosages": ["5mg", "10mg", "20mg"],
        "side_effects": ["Headache", "Muscle pain", "Nausea", "Weakness"],
        "warnings": ["Can be taken at any time of day.", "Report muscle pain or dark urine immediately."],
    },
    {
        "generic_name": "Clopidogrel",
        "brand_names": ["Plavix", "Clopilet", "Clopidogrel"],
        "drug_class": "Antiplatelet",
        "use": "Prevents blood clots. Used after heart attacks, strokes, and stent placement.",
        "common_dosages": ["75mg"],
        "side_effects": ["Easy bruising", "Bleeding", "Stomach upset"],
        "warnings": ["Do not stop without doctor's advice.", "Tell dentist/surgeon you are taking this.", "Avoid aspirin unless prescribed."],
    },
    {
        "generic_name": "Aspirin",
        "brand_names": ["Disprin", "Ecosprin", "Aspirin"],
        "drug_class": "Antiplatelet / NSAID",
        "use": "Low-dose: prevents blood clots and heart attacks. High-dose: pain relief and anti-inflammatory.",
        "common_dosages": ["75mg", "150mg", "325mg"],
        "side_effects": ["Stomach irritation", "Easy bleeding", "Heartburn"],
        "warnings": ["Take with food.", "Not for children under 16.", "Increases bleeding risk."],
    },

    # ── Respiratory / Asthma ───────────────────────────────────────────────
    {
        "generic_name": "Salbutamol",
        "brand_names": ["Ventolin", "Asthalin", "Salbutamol"],
        "drug_class": "Bronchodilator (Beta-2 Agonist)",
        "use": "Opens airways during asthma attacks and breathing difficulty. Quick-relief inhaler.",
        "common_dosages": ["100mcg inhaler", "2mg tablet", "4mg tablet"],
        "side_effects": ["Tremor", "Fast heartbeat", "Headache", "Nervousness"],
        "warnings": ["For quick relief only — not for daily prevention.", "Rinse mouth after using inhaler."],
    },

    # ── Neurological / Psychiatric ─────────────────────────────────────────
    {
        "generic_name": "Gabapentin",
        "brand_names": ["Neurontin", "Gabantin", "Gabapentin"],
        "drug_class": "Anticonvulsant / Nerve Pain Medication",
        "use": "Treats nerve pain (neuropathy), seizures, and restless leg syndrome.",
        "common_dosages": ["100mg", "300mg", "400mg"],
        "side_effects": ["Drowsiness", "Dizziness", "Fatigue", "Swelling"],
        "warnings": ["Do not stop suddenly — must be tapered.", "May cause drowsiness — avoid driving."],
    },
    {
        "generic_name": "Pregabalin",
        "brand_names": ["Lyrica", "Pregastar", "Pregabalin"],
        "drug_class": "Anticonvulsant / Nerve Pain Medication",
        "use": "Treats nerve pain from diabetes, shingles, fibromyalgia, and anxiety.",
        "common_dosages": ["50mg", "75mg", "150mg"],
        "side_effects": ["Drowsiness", "Dizziness", "Weight gain", "Blurred vision"],
        "warnings": ["Do not stop suddenly.", "May cause drowsiness.", "Avoid alcohol."],
    },
    {
        "generic_name": "Sertraline",
        "brand_names": ["Zoloft", "Serta", "Sertraline"],
        "drug_class": "SSRI Antidepressant",
        "use": "Treats depression, anxiety, panic attacks, OCD, and PTSD.",
        "common_dosages": ["25mg", "50mg", "100mg"],
        "side_effects": ["Nausea", "Headache", "Drowsiness", "Dry mouth", "Insomnia"],
        "warnings": ["Takes 2-4 weeks to show full effect.", "Do not stop suddenly.", "Avoid alcohol."],
    },
    {
        "generic_name": "Escitalopram",
        "brand_names": ["Lexapro", "Cipralex", "Escitalopram"],
        "drug_class": "SSRI Antidepressant",
        "use": "Treats depression and generalized anxiety disorder.",
        "common_dosages": ["5mg", "10mg", "20mg"],
        "side_effects": ["Nausea", "Drowsiness", "Insomnia", "Sexual dysfunction"],
        "warnings": ["Takes 2-4 weeks for full effect.", "Do not stop abruptly.", "Avoid alcohol."],
    },
    {
        "generic_name": "Alprazolam",
        "brand_names": ["Xanax", "Alprax", "Alprazolam"],
        "drug_class": "Benzodiazepine (Anti-anxiety)",
        "use": "Treats anxiety and panic disorders. Provides quick relief from severe anxiety.",
        "common_dosages": ["0.25mg", "0.5mg", "1mg"],
        "side_effects": ["Drowsiness", "Fatigue", "Memory problems", "Dizziness"],
        "warnings": ["Can be habit-forming — use only as prescribed.", "Do NOT stop suddenly.", "Avoid alcohol.", "Do not drive."],
    },
    {
        "generic_name": "Clonazepam",
        "brand_names": ["Klonopin", "Rivotril", "Clonazepam"],
        "drug_class": "Benzodiazepine (Anticonvulsant / Anti-anxiety)",
        "use": "Treats seizures, panic disorder, and anxiety.",
        "common_dosages": ["0.25mg", "0.5mg", "1mg", "2mg"],
        "side_effects": ["Drowsiness", "Dizziness", "Coordination problems"],
        "warnings": ["Can be habit-forming.", "Do not stop suddenly — must be tapered.", "Avoid alcohol."],
    },

    # ── Muscle Relaxants ───────────────────────────────────────────────────
    {
        "generic_name": "Baclofen",
        "brand_names": ["Lioresal", "Baclofen", "Baclon", "Bacmax", "Flexibac"],
        "drug_class": "Muscle Relaxant",
        "use": "Relieves muscle stiffness and spasms from conditions like multiple sclerosis and spinal cord injuries.",
        "common_dosages": ["10mg", "25mg"],
        "side_effects": ["Drowsiness", "Dizziness", "Weakness", "Nausea"],
        "warnings": ["Do not stop suddenly — must be tapered.", "May cause drowsiness.", "Avoid alcohol."],
    },

    # ── Antifungal / Skin ──────────────────────────────────────────────────
    {
        "generic_name": "Ketoconazole",
        "brand_names": ["Nizoral", "Ketocon", "Ketoral", "Ketotab", "Ketozol", "Nizoder"],
        "drug_class": "Antifungal",
        "use": "Treats fungal infections of the skin, scalp (dandruff), and nails.",
        "common_dosages": ["200mg tablet", "2% cream", "2% shampoo"],
        "side_effects": ["Nausea", "Stomach pain", "Headache", "Skin irritation (topical)"],
        "warnings": ["Oral form may affect liver — needs monitoring.", "Topical forms are generally safe."],
    },

    # ── Prokinetic / Anti-diarrheal ────────────────────────────────────────
    {
        "generic_name": "Loperamide",
        "brand_names": ["Imodium", "Lopamide", "Loperamide"],
        "drug_class": "Antidiarrheal",
        "use": "Treats acute diarrhea by slowing bowel movement.",
        "common_dosages": ["2mg"],
        "side_effects": ["Constipation", "Stomach cramps", "Nausea"],
        "warnings": ["Do not use for more than 2 days without doctor advice.", "Stop if fever develops.", "Stay hydrated."],
    },

    # ── Supplements / Vitamins ─────────────────────────────────────────────
    {
        "generic_name": "Vitamin D3 (Cholecalciferol)",
        "brand_names": ["D-Rise", "Calcirol", "Uprise D3"],
        "drug_class": "Vitamin Supplement",
        "use": "Treats and prevents Vitamin D deficiency. Important for bone health and immunity.",
        "common_dosages": ["1000 IU", "60000 IU (weekly)"],
        "side_effects": ["Rarely causes side effects at normal doses"],
        "warnings": ["Take with fatty food for better absorption.", "Excessive doses can cause calcium buildup."],
    },
    {
        "generic_name": "Calcium + Vitamin D",
        "brand_names": ["Shelcal", "CCM", "Gemcal"],
        "drug_class": "Mineral Supplement",
        "use": "Prevents and treats calcium deficiency. Strengthens bones, especially in women.",
        "common_dosages": ["500mg + 250IU"],
        "side_effects": ["Constipation", "Gas", "Stomach upset"],
        "warnings": ["Take after meals.", "Do not take with iron supplements at the same time."],
    },
    {
        "generic_name": "Iron (Ferrous Sulfate/Fumarate)",
        "brand_names": ["Fefol", "Autrin", "Orofer"],
        "drug_class": "Mineral Supplement",
        "use": "Treats iron-deficiency anemia. Increases red blood cell production.",
        "common_dosages": ["200mg", "325mg"],
        "side_effects": ["Constipation", "Black stool", "Nausea", "Stomach upset"],
        "warnings": ["Take on empty stomach with vitamin C for best absorption.", "Black stool is normal."],
    },
]


# ─── Lab Test Knowledge Base ──────────────────────────────────────────────────

LAB_DATABASE: List[Dict] = [
    # ── CBC ────────────────────────────────────────────────────────────────
    {
        "test_name": "Haemoglobin (Hb)",
        "aliases": ["Hemoglobin", "Hb", "HGB"],
        "category": "CBC",
        "what_it_measures": "The protein in red blood cells that carries oxygen throughout your body.",
        "unit": "g/dL",
        "normal_range_male": [13.0, 17.0],
        "normal_range_female": [12.0, 15.5],
        "high_means": "Dehydration, lung disease, or living at high altitude. Rarely, a blood disorder.",
        "low_means": "Anemia — your body isn't getting enough oxygen. Could be from iron deficiency, blood loss, or chronic disease.",
    },
    {
        "test_name": "RBC Count",
        "aliases": ["Red Blood Cell Count", "RBC", "Erythrocyte Count"],
        "category": "CBC",
        "what_it_measures": "The number of red blood cells in your blood.",
        "unit": "million/µL",
        "normal_range_male": [4.5, 5.5],
        "normal_range_female": [3.8, 4.8],
        "high_means": "Dehydration, smoking, or living at high altitude.",
        "low_means": "Anemia, blood loss, or bone marrow problems.",
    },
    {
        "test_name": "WBC Count (TLC)",
        "aliases": ["White Blood Cell Count", "WBC", "TLC", "Leukocyte Count"],
        "category": "CBC",
        "what_it_measures": "The number of white blood cells that fight infections.",
        "unit": "cells/µL",
        "normal_range_male": [4000, 11000],
        "normal_range_female": [4000, 11000],
        "high_means": "Your body is fighting an infection or inflammation. Could also indicate stress or medication effects.",
        "low_means": "Weak immune system. Could be from a viral infection, bone marrow problem, or certain medications.",
    },
    {
        "test_name": "Platelet Count",
        "aliases": ["Platelets", "PLT", "Thrombocyte Count"],
        "category": "CBC",
        "what_it_measures": "The cells that help your blood clot and stop bleeding.",
        "unit": "lakhs/µL",
        "normal_range_male": [1.5, 4.0],
        "normal_range_female": [1.5, 4.0],
        "high_means": "Infection, inflammation, iron deficiency, or in rare cases a bone marrow disorder.",
        "low_means": "Risk of excessive bleeding. Could be from dengue, viral infections, medications, or liver disease.",
    },
    {
        "test_name": "PCV / Hematocrit",
        "aliases": ["PCV", "Hematocrit", "HCT", "Packed Cell Volume"],
        "category": "CBC",
        "what_it_measures": "The percentage of your blood that is made up of red blood cells.",
        "unit": "%",
        "normal_range_male": [40, 50],
        "normal_range_female": [36, 44],
        "high_means": "Dehydration or polycythemia (too many red blood cells).",
        "low_means": "Anemia or overhydration.",
    },
    {
        "test_name": "MCV",
        "aliases": ["Mean Corpuscular Volume"],
        "category": "CBC",
        "what_it_measures": "The average size of your red blood cells.",
        "unit": "fL",
        "normal_range_male": [80, 100],
        "normal_range_female": [80, 100],
        "high_means": "Red blood cells are larger than normal — could be from vitamin B12 or folate deficiency.",
        "low_means": "Red blood cells are smaller than normal — often from iron deficiency.",
    },
    {
        "test_name": "MCH",
        "aliases": ["Mean Corpuscular Hemoglobin"],
        "category": "CBC",
        "what_it_measures": "The average amount of hemoglobin in each red blood cell.",
        "unit": "pg",
        "normal_range_male": [27, 32],
        "normal_range_female": [27, 32],
        "high_means": "May indicate vitamin B12 or folate deficiency.",
        "low_means": "May indicate iron deficiency anemia.",
    },
    {
        "test_name": "MCHC",
        "aliases": ["Mean Corpuscular Hemoglobin Concentration"],
        "category": "CBC",
        "what_it_measures": "The average concentration of hemoglobin in your red blood cells.",
        "unit": "g/dL",
        "normal_range_male": [32, 36],
        "normal_range_female": [32, 36],
        "high_means": "Hereditary spherocytosis or autoimmune hemolytic anemia (rare).",
        "low_means": "Iron deficiency anemia or thalassemia.",
    },
    {
        "test_name": "RDW-CV",
        "aliases": ["RDW", "Red Cell Distribution Width"],
        "category": "CBC",
        "what_it_measures": "How much variation there is in the size of your red blood cells.",
        "unit": "%",
        "normal_range_male": [11.5, 14.5],
        "normal_range_female": [11.5, 14.5],
        "high_means": "Red blood cells vary widely in size — often seen in iron deficiency or mixed anemias.",
        "low_means": "Usually normal and not clinically significant.",
    },
    {
        "test_name": "ESR",
        "aliases": ["Erythrocyte Sedimentation Rate", "Sed Rate"],
        "category": "CBC",
        "what_it_measures": "How quickly red blood cells settle — a marker of inflammation in the body.",
        "unit": "mm/hr",
        "normal_range_male": [0, 15],
        "normal_range_female": [0, 20],
        "high_means": "Inflammation somewhere in the body — could be from infection, autoimmune disease, or cancer.",
        "low_means": "Usually normal. Very low values are rarely clinically significant.",
    },
    {
        "test_name": "Neutrophils",
        "aliases": ["Neutrophil %", "Neutrophil Count"],
        "category": "CBC - Differential",
        "what_it_measures": "The main type of white blood cell that fights bacterial infections.",
        "unit": "%",
        "normal_range_male": [40, 70],
        "normal_range_female": [40, 70],
        "high_means": "Bacterial infection, inflammation, or stress.",
        "low_means": "Viral infection, certain medications, or bone marrow problems.",
    },
    {
        "test_name": "Lymphocytes",
        "aliases": ["Lymphocyte %", "Lymphocyte Count"],
        "category": "CBC - Differential",
        "what_it_measures": "White blood cells important for fighting viral infections and producing antibodies.",
        "unit": "%",
        "normal_range_male": [20, 40],
        "normal_range_female": [20, 40],
        "high_means": "Viral infection (e.g., dengue, COVID), chronic infection, or lymphoma.",
        "low_means": "HIV/AIDS, steroid use, or autoimmune diseases.",
    },
    {
        "test_name": "Monocytes",
        "aliases": ["Monocyte %"],
        "category": "CBC - Differential",
        "what_it_measures": "White blood cells that help fight chronic infections and remove dead cells.",
        "unit": "%",
        "normal_range_male": [2, 8],
        "normal_range_female": [2, 8],
        "high_means": "Chronic infection (e.g., TB), autoimmune disease, or recovery from infection.",
        "low_means": "Rarely significant on its own.",
    },
    {
        "test_name": "Eosinophils",
        "aliases": ["Eosinophil %"],
        "category": "CBC - Differential",
        "what_it_measures": "White blood cells involved in fighting parasites and allergic reactions.",
        "unit": "%",
        "normal_range_male": [1, 4],
        "normal_range_female": [1, 4],
        "high_means": "Allergies, asthma, parasitic infection (worms), or drug reactions.",
        "low_means": "Usually normal.",
    },
    {
        "test_name": "Basophils",
        "aliases": ["Basophil %"],
        "category": "CBC - Differential",
        "what_it_measures": "The rarest white blood cells, involved in allergic and inflammatory responses.",
        "unit": "%",
        "normal_range_male": [0, 1],
        "normal_range_female": [0, 1],
        "high_means": "Allergic reactions, chronic inflammation, or very rarely blood disorders.",
        "low_means": "Usually normal.",
    },

    # ── Lipid Profile ──────────────────────────────────────────────────────
    {
        "test_name": "Total Cholesterol",
        "aliases": ["Cholesterol", "TC"],
        "category": "LIPID",
        "what_it_measures": "The total amount of cholesterol in your blood (good + bad combined).",
        "unit": "mg/dL",
        "normal_range_male": [0, 200],
        "normal_range_female": [0, 200],
        "high_means": "Increased risk of heart disease and stroke. Diet and lifestyle changes may help.",
        "low_means": "Rarely a concern. Very low levels may be linked to malnutrition or liver problems.",
    },
    {
        "test_name": "Triglycerides",
        "aliases": ["TG", "Triglyceride"],
        "category": "LIPID",
        "what_it_measures": "A type of fat in your blood. High levels increase heart disease risk.",
        "unit": "mg/dL",
        "normal_range_male": [0, 150],
        "normal_range_female": [0, 150],
        "high_means": "Increased heart disease risk. Often linked to diet, obesity, diabetes, or excessive alcohol.",
        "low_means": "Usually not a concern.",
    },
    {
        "test_name": "HDL Cholesterol",
        "aliases": ["HDL", "Good Cholesterol"],
        "category": "LIPID",
        "what_it_measures": "The 'good' cholesterol that removes bad cholesterol from your arteries.",
        "unit": "mg/dL",
        "normal_range_male": [40, 60],
        "normal_range_female": [50, 70],
        "high_means": "Good! Higher HDL is protective against heart disease.",
        "low_means": "Increased heart disease risk. Exercise and healthy fats can help raise it.",
    },
    {
        "test_name": "LDL Cholesterol",
        "aliases": ["LDL", "Bad Cholesterol"],
        "category": "LIPID",
        "what_it_measures": "The 'bad' cholesterol that builds up in artery walls and causes blockages.",
        "unit": "mg/dL",
        "normal_range_male": [0, 100],
        "normal_range_female": [0, 100],
        "high_means": "Increased risk of heart attack and stroke. May need statins and diet changes.",
        "low_means": "Generally good. Very low levels are rarely concerning.",
    },
    {
        "test_name": "VLDL Cholesterol",
        "aliases": ["VLDL"],
        "category": "LIPID",
        "what_it_measures": "Very low density lipoprotein — carries triglycerides in the blood.",
        "unit": "mg/dL",
        "normal_range_male": [5, 40],
        "normal_range_female": [5, 40],
        "high_means": "High triglyceride levels. Linked to heart disease risk.",
        "low_means": "Usually normal.",
    },

    # ── Blood Sugar ────────────────────────────────────────────────────────
    {
        "test_name": "Fasting Blood Sugar",
        "aliases": ["FBS", "Fasting Glucose", "Fasting Blood Glucose"],
        "category": "DIABETES",
        "what_it_measures": "Your blood sugar level after not eating for 8-12 hours. Used to screen for diabetes.",
        "unit": "mg/dL",
        "normal_range_male": [70, 100],
        "normal_range_female": [70, 100],
        "high_means": "Pre-diabetes (100-125) or diabetes (>125). Needs diet control and possibly medication.",
        "low_means": "Hypoglycemia — you may feel dizzy, shaky, or sweaty. Eat something sugary immediately.",
    },
    {
        "test_name": "HbA1c",
        "aliases": ["Glycated Hemoglobin", "A1C", "Glycosylated Hemoglobin"],
        "category": "DIABETES",
        "what_it_measures": "Your average blood sugar over the last 2-3 months. The gold standard for diabetes monitoring.",
        "unit": "%",
        "normal_range_male": [4.0, 5.6],
        "normal_range_female": [4.0, 5.6],
        "high_means": "Pre-diabetes (5.7-6.4%) or diabetes (>6.5%). Indicates poor blood sugar control over time.",
        "low_means": "Blood sugar has been well controlled. Very low values may indicate frequent hypoglycemia.",
    },

    # ── Kidney ─────────────────────────────────────────────────────────────
    {
        "test_name": "Creatinine",
        "aliases": ["Serum Creatinine", "Creat"],
        "category": "KIDNEY",
        "what_it_measures": "A waste product filtered by your kidneys. Shows how well your kidneys are working.",
        "unit": "mg/dL",
        "normal_range_male": [0.7, 1.3],
        "normal_range_female": [0.6, 1.1],
        "high_means": "Your kidneys may not be filtering properly. Could indicate kidney disease or dehydration.",
        "low_means": "Usually normal. Very low levels may be seen in low muscle mass.",
    },
    {
        "test_name": "Blood Urea Nitrogen",
        "aliases": ["BUN", "Urea", "Blood Urea"],
        "category": "KIDNEY",
        "what_it_measures": "A waste product from protein breakdown, filtered by the kidneys.",
        "unit": "mg/dL",
        "normal_range_male": [7, 20],
        "normal_range_female": [7, 20],
        "high_means": "Kidney problems, dehydration, or high protein diet.",
        "low_means": "Liver disease or malnutrition (rare).",
    },

    # ── Liver ──────────────────────────────────────────────────────────────
    {
        "test_name": "SGPT (ALT)",
        "aliases": ["ALT", "SGPT", "Alanine Aminotransferase"],
        "category": "LIVER",
        "what_it_measures": "An enzyme found mainly in the liver. High levels suggest liver damage.",
        "unit": "U/L",
        "normal_range_male": [7, 56],
        "normal_range_female": [7, 45],
        "high_means": "Liver inflammation or damage — from hepatitis, fatty liver, alcohol, or medications.",
        "low_means": "Usually normal.",
    },
    {
        "test_name": "SGOT (AST)",
        "aliases": ["AST", "SGOT", "Aspartate Aminotransferase"],
        "category": "LIVER",
        "what_it_measures": "An enzyme found in liver, heart, and muscles. High levels may indicate tissue damage.",
        "unit": "U/L",
        "normal_range_male": [10, 40],
        "normal_range_female": [10, 35],
        "high_means": "Liver disease, heart problems, or muscle injury.",
        "low_means": "Usually normal.",
    },

    # ── Thyroid ─────────────────────────────────────────────────────────────
    {
        "test_name": "TSH",
        "aliases": ["Thyroid Stimulating Hormone", "Thyrotropin"],
        "category": "THYROID",
        "what_it_measures": "Controls your thyroid gland. Shows if your thyroid is overactive or underactive.",
        "unit": "mIU/L",
        "normal_range_male": [0.4, 4.0],
        "normal_range_female": [0.4, 4.0],
        "high_means": "Hypothyroidism (underactive thyroid) — may cause weight gain, fatigue, and feeling cold.",
        "low_means": "Hyperthyroidism (overactive thyroid) — may cause weight loss, anxiety, and rapid heartbeat.",
    },
]


# ─── Knowledge Base Class ─────────────────────────────────────────────────────

class MedicalKB:
    """
    Medical Knowledge Base with fuzzy lookup for drug and lab test information.

    Supports both exact and fuzzy string matching to tolerate OCR errors
    in drug names and lab test names.
    """

    def __init__(self):
        self.drug_db = DRUG_DATABASE
        self.lab_db = LAB_DATABASE

        # Build lookup indices
        self._drug_index: Dict[str, Dict] = {}
        for drug in self.drug_db:
            # Index by generic name
            key = drug["generic_name"].lower()
            self._drug_index[key] = drug
            # Index by brand names
            for brand in drug.get("brand_names", []):
                self._drug_index[brand.lower()] = drug

        self._lab_index: Dict[str, Dict] = {}
        for test in self.lab_db:
            key = test["test_name"].lower()
            self._lab_index[key] = test
            for alias in test.get("aliases", []):
                self._lab_index[alias.lower()] = test

    def lookup_drug(self, name: str, threshold: int = 70) -> Optional[Dict]:
        """
        Look up drug information by name (exact or fuzzy match).

        Args:
            name: Drug name to search for (brand or generic).
            threshold: Minimum fuzzy match score (0-100).

        Returns:
            Drug info dict or None if no match found.
        """
        clean_name = name.lower().strip()

        # Strip common prefixes
        for prefix in ["tab", "tab.", "cap", "cap.", "syr", "syr.", "inj", "inj.", "drops"]:
            if clean_name.startswith(prefix + " "):
                clean_name = clean_name[len(prefix) + 1:].strip()

        # Strip dosage suffixes (e.g., "500mg", "10 mg")
        import re
        clean_name_no_dose = re.sub(r'\s*\d+\s*(mg|mcg|ml|g|iu)\s*$', '', clean_name, flags=re.IGNORECASE).strip()

        # 1. Exact match
        if clean_name in self._drug_index:
            return self._drug_index[clean_name]
        if clean_name_no_dose and clean_name_no_dose in self._drug_index:
            return self._drug_index[clean_name_no_dose]

        # 2. Fuzzy match
        from rapidfuzz import fuzz, process

        candidates = list(self._drug_index.keys())
        # Try matching without dosage first
        search_term = clean_name_no_dose if clean_name_no_dose else clean_name
        result = process.extractOne(search_term, candidates, scorer=fuzz.ratio, score_cutoff=threshold)

        if result:
            matched_key, score, _ = result
            logger.debug(f"Fuzzy matched '{name}' → '{matched_key}' (score: {score})")
            return self._drug_index[matched_key]

        return None

    def lookup_lab_test(self, name: str, threshold: int = 65) -> Optional[Dict]:
        """
        Look up lab test information by name (exact or fuzzy match).

        Args:
            name: Lab test name to search for.
            threshold: Minimum fuzzy match score (0-100).

        Returns:
            Lab test info dict or None if no match found.
        """
        clean_name = name.lower().strip()

        # Exact match
        if clean_name in self._lab_index:
            return self._lab_index[clean_name]

        # Fuzzy match
        from rapidfuzz import fuzz, process

        candidates = list(self._lab_index.keys())
        result = process.extractOne(clean_name, candidates, scorer=fuzz.partial_ratio, score_cutoff=threshold)

        if result:
            matched_key, score, _ = result
            logger.debug(f"Fuzzy matched '{name}' → '{matched_key}' (score: {score})")
            return self._lab_index[matched_key]

        return None

    def get_all_drugs(self) -> List[Dict]:
        """Return the full drug database."""
        return self.drug_db

    def get_all_lab_tests(self) -> List[Dict]:
        """Return the full lab test database."""
        return self.lab_db

    def export_json(self):
        """Export knowledge base to JSON files for fast loading."""
        with open(DRUG_KB_PATH, "w", encoding="utf-8") as f:
            json.dump(self.drug_db, f, indent=2, ensure_ascii=False)
        logger.info(f"Drug KB exported to {DRUG_KB_PATH} ({len(self.drug_db)} entries)")

        with open(LAB_KB_PATH, "w", encoding="utf-8") as f:
            json.dump(self.lab_db, f, indent=2, ensure_ascii=False)
        logger.info(f"Lab KB exported to {LAB_KB_PATH} ({len(self.lab_db)} entries)")


# ─── Standalone: Build & Export ────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    kb = MedicalKB()

    print("=" * 60)
    print("ClariRx Medical Knowledge Base Builder")
    print("=" * 60)
    print(f"\n  Drug entries:     {len(kb.drug_db)}")
    print(f"  Lab test entries: {len(kb.lab_db)}")
    print(f"  Drug index keys:  {len(kb._drug_index)}")
    print(f"  Lab index keys:   {len(kb._lab_index)}")

    # Export JSON
    kb.export_json()
    print(f"\n  ✅ Exported to {DRUG_KB_PATH}")
    print(f"  ✅ Exported to {LAB_KB_PATH}")

    # Test lookups
    print("\n" + "─" * 60)
    print("Testing Drug Lookups")
    print("─" * 60)

    test_drugs = [
        "Amoxicillin 500mg",    # Exact
        "Tab Paracetamol",      # With prefix
        "Am0xicillin",          # OCR typo
        "Cetzine",              # Brand name
        "Alatrol 10mg",         # Brand name
        "Napa 650mg",           # Bangladeshi brand
        "Rivotril",             # Benzo brand
        "UnknownDrug123",       # Should return None
    ]

    for drug_name in test_drugs:
        result = kb.lookup_drug(drug_name)
        if result:
            print(f"  ✅ '{drug_name}' → {result['generic_name']} ({result['drug_class']})")
        else:
            print(f"  ❌ '{drug_name}' → Not found")

    print("\n" + "─" * 60)
    print("Testing Lab Test Lookups")
    print("─" * 60)

    test_labs = [
        "Haemoglobin",              # Exact
        "WBC Count",                # Partial
        "Total Cholesterol",        # Exact
        "HbA1c",                    # Alias
        "Fasting Blood Sugar",      # Exact
        "TSH",                      # Alias
        "UnknownTest",              # Should return None
    ]

    for test_name in test_labs:
        result = kb.lookup_lab_test(test_name)
        if result:
            print(f"  ✅ '{test_name}' → {result['test_name']} ({result['category']})")
        else:
            print(f"  ❌ '{test_name}' → Not found")
