"""
Synthetic Lab Report Generator — Phase 1
=========================================

Generates realistic lab report images (PNG) + a ground-truth CSV for evaluation.

Supported report types (Phase 1):
  - CBC  (Complete Blood Count)
  - Lipid Profile

Each generated report includes:
  - Randomized-but-realistic patient demographics
  - Test values sampled from realistic clinical distributions
  - Some values intentionally pushed outside normal ranges (to test abnormality detection)
  - A corresponding row in the ground-truth CSV

Usage:
    python generate_lab_reports.py --count 50 --output-dir ../raw/lab_reports --gt-csv ../labeled/lab_reports_gt.csv

Dependencies:
    pip install Jinja2 Pillow pandas

Note on rendering:
    This script renders HTML → PNG using Pillow-based pure-Python rendering (no wkhtmltopdf needed).
    For higher-fidelity rendering, install wkhtmltopdf and set USE_IMGKIT=True.
"""

import argparse
import csv
import os
import random
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

# Fix Unicode output on Windows consoles (cp1252 can't handle all chars)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "raw" / "lab_reports"
DEFAULT_GT_CSV = SCRIPT_DIR.parent / "labeled" / "lab_reports_gt.csv"

# Whether to use imgkit (requires wkhtmltopdf installed) for higher-fidelity rendering.
# If False, falls back to saving HTML files that can be opened in a browser.
USE_IMGKIT = False

# ─── Realistic Indian Names & Lab Details ──────────────────────────────────────

FIRST_NAMES_MALE = [
    "Rajesh", "Sunil", "Amit", "Vikram", "Manoj", "Arun", "Sanjay", "Deepak",
    "Ramesh", "Mukesh", "Pramod", "Ajay", "Ravi", "Suresh", "Vinod", "Ashok",
    "Gopal", "Harish", "Mohan", "Naresh", "Pankaj", "Rakesh", "Sachin",
]
FIRST_NAMES_FEMALE = [
    "Sunita", "Priya", "Anita", "Meena", "Kavita", "Rekha", "Neha", "Pooja",
    "Sarita", "Asha", "Geeta", "Renu", "Suman", "Usha", "Lata", "Seema",
    "Archana", "Deepa", "Jyoti", "Kiran", "Manju", "Nandini", "Pallavi",
]
LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Joshi", "Mishra",
    "Agarwal", "Tiwari", "Chauhan", "Yadav", "Reddy", "Nair", "Iyer", "Das",
    "Roy", "Mukherjee", "Banerjee", "Choudhury", "Kapoor", "Malhotra", "Mehta",
]
DOCTOR_NAMES = [
    "Dr. R.K. Sharma", "Dr. A.P. Singh", "Dr. S. Gupta", "Dr. M. Verma",
    "Dr. P. Joshi", "Dr. N. Patel", "Dr. K. Reddy", "Dr. V. Agarwal",
    "Dr. D. Mishra", "Dr. H. Kumar", "Dr. B. Das", "Dr. T. Nair",
]
LAB_NAMES = [
    "SRL Diagnostics", "Dr. Lal PathLabs", "Metropolis Healthcare",
    "Thyrocare Technologies", "Suburban Diagnostics", "Apollo Diagnostics",
    "Neuberg Diagnostics", "Vijaya Diagnostic Centre", "Healthians Lab",
    "City Clinical Laboratory", "Precision Diagnostics", "MedAll Healthcare",
]
LAB_ADDRESSES = [
    "Plot 14, Sector 18, Noida, UP - 201301",
    "42, M.G. Road, Bengaluru, KA - 560001",
    "B-28, Nehru Place, New Delhi - 110019",
    "Ameerpet, Hyderabad, TS - 500016",
    "Andheri West, Mumbai, MH - 400058",
    "Salt Lake, Kolkata, WB - 700091",
    "Anna Nagar, Chennai, TN - 600040",
    "Aundh, Pune, MH - 411007",
    "Gomti Nagar, Lucknow, UP - 226010",
    "C-Scheme, Jaipur, RJ - 302001",
]
LAB_ACCREDITATIONS = [
    "NABL Accredited | ISO 15189:2022 | CAP Accredited",
    "NABL Accredited | ISO 15189:2022",
    "NABL Accredited | ICMR Approved",
    "ISO 15189:2022 Certified Laboratory",
]
PATHOLOGISTS = [
    ("Dr. Shalini Mehta", "MD (Pathology), FIAC"),
    ("Dr. Anand Kulkarni", "MD (Pathology), DCP"),
    ("Dr. Prerna Kapoor", "MD (Pathology), DNB"),
    ("Dr. Vivek Tandon", "MBBS, MD (Pathology)"),
    ("Dr. Kavita Rao", "MD (Pathology), FRCPath"),
]

# ─── Test Definitions (CBC + Lipid Profile) ───────────────────────────────────
# Each test is defined with:
#   - name: display name on the report
#   - unit: measurement unit
#   - normal_low, normal_high: reference range
#   - mean, std: for generating realistic Gaussian-distributed values
#   - abnormal_prob: probability of generating an intentionally out-of-range value
#   - gender_specific: if True, ranges differ by gender (handled separately)

CBC_TESTS = [
    {
        "name": "Haemoglobin (Hb)",
        "unit": "g/dL",
        "normal_low_m": 13.0, "normal_high_m": 17.0,
        "normal_low_f": 12.0, "normal_high_f": 15.5,
        "mean_m": 14.8, "std_m": 1.5,
        "mean_f": 13.2, "std_f": 1.3,
        "abnormal_prob": 0.30,
        "decimals": 1,
        "gender_specific": True,
    },
    {
        "name": "RBC Count",
        "unit": "million/µL",
        "normal_low_m": 4.5, "normal_high_m": 5.5,
        "normal_low_f": 3.8, "normal_high_f": 4.8,
        "mean_m": 5.0, "std_m": 0.5,
        "mean_f": 4.3, "std_f": 0.4,
        "abnormal_prob": 0.20,
        "decimals": 2,
        "gender_specific": True,
    },
    {
        "name": "WBC Count (TLC)",
        "unit": "cells/µL",
        "normal_low": 4000, "normal_high": 11000,
        "mean": 7200, "std": 2200,
        "abnormal_prob": 0.25,
        "decimals": 0,
        "gender_specific": False,
    },
    {
        "name": "Platelet Count",
        "unit": "lakhs/µL",
        "normal_low": 1.5, "normal_high": 4.0,
        "mean": 2.5, "std": 0.7,
        "abnormal_prob": 0.20,
        "decimals": 2,
        "gender_specific": False,
    },
    {
        "name": "PCV / Hematocrit",
        "unit": "%",
        "normal_low_m": 40, "normal_high_m": 50,
        "normal_low_f": 36, "normal_high_f": 44,
        "mean_m": 44, "std_m": 4,
        "mean_f": 39, "std_f": 3.5,
        "abnormal_prob": 0.20,
        "decimals": 1,
        "gender_specific": True,
    },
    {
        "name": "MCV",
        "unit": "fL",
        "normal_low": 80, "normal_high": 100,
        "mean": 88, "std": 7,
        "abnormal_prob": 0.20,
        "decimals": 1,
        "gender_specific": False,
    },
    {
        "name": "MCH",
        "unit": "pg",
        "normal_low": 27, "normal_high": 32,
        "mean": 29.5, "std": 2.5,
        "abnormal_prob": 0.15,
        "decimals": 1,
        "gender_specific": False,
    },
    {
        "name": "MCHC",
        "unit": "g/dL",
        "normal_low": 32, "normal_high": 36,
        "mean": 34, "std": 1.5,
        "abnormal_prob": 0.15,
        "decimals": 1,
        "gender_specific": False,
    },
    {
        "name": "RDW-CV",
        "unit": "%",
        "normal_low": 11.5, "normal_high": 14.5,
        "mean": 13.0, "std": 1.2,
        "abnormal_prob": 0.20,
        "decimals": 1,
        "gender_specific": False,
    },
    {
        "name": "Neutrophils",
        "unit": "%",
        "normal_low": 40, "normal_high": 70,
        "mean": 58, "std": 10,
        "abnormal_prob": 0.20,
        "decimals": 0,
        "gender_specific": False,
    },
    {
        "name": "Lymphocytes",
        "unit": "%",
        "normal_low": 20, "normal_high": 40,
        "mean": 30, "std": 7,
        "abnormal_prob": 0.20,
        "decimals": 0,
        "gender_specific": False,
    },
    {
        "name": "Monocytes",
        "unit": "%",
        "normal_low": 2, "normal_high": 8,
        "mean": 5, "std": 2,
        "abnormal_prob": 0.15,
        "decimals": 0,
        "gender_specific": False,
    },
    {
        "name": "Eosinophils",
        "unit": "%",
        "normal_low": 1, "normal_high": 4,
        "mean": 2.5, "std": 1.5,
        "abnormal_prob": 0.25,
        "decimals": 0,
        "gender_specific": False,
    },
    {
        "name": "Basophils",
        "unit": "%",
        "normal_low": 0, "normal_high": 1,
        "mean": 0.4, "std": 0.3,
        "abnormal_prob": 0.10,
        "decimals": 0,
        "gender_specific": False,
    },
    {
        "name": "ESR",
        "unit": "mm/hr",
        "normal_low_m": 0, "normal_high_m": 15,
        "normal_low_f": 0, "normal_high_f": 20,
        "mean_m": 8, "std_m": 6,
        "mean_f": 10, "std_f": 7,
        "abnormal_prob": 0.25,
        "decimals": 0,
        "gender_specific": True,
    },
]

LIPID_PROFILE_TESTS = [
    {
        "name": "Total Cholesterol",
        "unit": "mg/dL",
        "normal_low": 0, "normal_high": 200,
        "mean": 190, "std": 35,
        "abnormal_prob": 0.35,
        "decimals": 0,
        "gender_specific": False,
    },
    {
        "name": "Triglycerides",
        "unit": "mg/dL",
        "normal_low": 0, "normal_high": 150,
        "mean": 140, "std": 55,
        "abnormal_prob": 0.35,
        "decimals": 0,
        "gender_specific": False,
    },
    {
        "name": "HDL Cholesterol",
        "unit": "mg/dL",
        "normal_low_m": 40, "normal_high_m": 60,
        "normal_low_f": 50, "normal_high_f": 70,
        "mean_m": 46, "std_m": 10,
        "mean_f": 56, "std_f": 12,
        "abnormal_prob": 0.30,
        "decimals": 0,
        "gender_specific": True,
    },
    {
        "name": "LDL Cholesterol",
        "unit": "mg/dL",
        "normal_low": 0, "normal_high": 100,
        "mean": 110, "std": 30,
        "abnormal_prob": 0.35,
        "decimals": 0,
        "gender_specific": False,
    },
    {
        "name": "VLDL Cholesterol",
        "unit": "mg/dL",
        "normal_low": 5, "normal_high": 40,
        "mean": 25, "std": 10,
        "abnormal_prob": 0.20,
        "decimals": 0,
        "gender_specific": False,
    },
    {
        "name": "Total Cholesterol / HDL Ratio",
        "unit": "",
        "normal_low": 0, "normal_high": 5.0,
        "mean": 4.2, "std": 1.0,
        "abnormal_prob": 0.25,
        "decimals": 1,
        "gender_specific": False,
    },
    {
        "name": "LDL / HDL Ratio",
        "unit": "",
        "normal_low": 0, "normal_high": 3.5,
        "mean": 2.5, "std": 0.8,
        "abnormal_prob": 0.25,
        "decimals": 1,
        "gender_specific": False,
    },
]

REPORT_TYPES = {
    "CBC": {
        "title": "COMPLETE BLOOD COUNT (CBC)",
        "tests": CBC_TESTS,
        "method": "Automated Hematology Analyzer (Sysmex XN-1000)",
        "sample_type": "EDTA Whole Blood",
    },
    "LIPID": {
        "title": "LIPID PROFILE",
        "tests": LIPID_PROFILE_TESTS,
        "method": "Enzymatic / Colorimetric (Roche Cobas c501)",
        "sample_type": "Serum (Fasting)",
    },
}


# ─── Value Generation ──────────────────────────────────────────────────────────

def generate_test_value(
    test_def: dict[str, Any],
    gender: str,
) -> tuple[float, float, float, bool]:
    """
    Generate a single test value based on the test definition.

    Returns:
        (value, normal_low, normal_high, is_abnormal)
    """
    # Resolve gender-specific ranges
    if test_def["gender_specific"]:
        suffix = "_m" if gender == "Male" else "_f"
        normal_low = test_def[f"normal_low{suffix}"]
        normal_high = test_def[f"normal_high{suffix}"]
        mean = test_def[f"mean{suffix}"]
        std = test_def[f"std{suffix}"]
    else:
        normal_low = test_def["normal_low"]
        normal_high = test_def["normal_high"]
        mean = test_def["mean"]
        std = test_def["std"]

    # Decide whether to intentionally generate an abnormal value
    force_abnormal = random.random() < test_def["abnormal_prob"]

    if force_abnormal:
        # Push value outside range — either high or low
        if random.random() < 0.5 and normal_low > 0:
            # Generate low abnormal value
            value = normal_low - abs(random.gauss(0, std * 0.5)) - (normal_high - normal_low) * 0.05
        else:
            # Generate high abnormal value
            value = normal_high + abs(random.gauss(0, std * 0.5)) + (normal_high - normal_low) * 0.05
    else:
        # Normal-range value
        value = random.gauss(mean, std * 0.6)
        # Clamp to stay within normal range with some margin
        value = max(normal_low + (normal_high - normal_low) * 0.05, value)
        value = min(normal_high - (normal_high - normal_low) * 0.05, value)

    # Ensure non-negative
    value = max(0, value)

    # Round to specified decimal places
    decimals = test_def["decimals"]
    value = round(value, decimals)
    if decimals == 0:
        value = int(value)

    is_abnormal = value < normal_low or value > normal_high

    return value, normal_low, normal_high, is_abnormal


def generate_patient_demographics() -> dict[str, Any]:
    """Generate randomized patient info."""
    gender = random.choice(["Male", "Female"])
    if gender == "Male":
        first_name = random.choice(FIRST_NAMES_MALE)
    else:
        first_name = random.choice(FIRST_NAMES_FEMALE)
    last_name = random.choice(LAST_NAMES)

    age = random.randint(18, 82)

    # Dates
    report_date = datetime.now() - timedelta(days=random.randint(0, 365))
    sample_date = report_date - timedelta(hours=random.randint(2, 24))

    patient_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

    return {
        "patient_name": f"{first_name} {last_name}",
        "patient_age": age,
        "patient_gender": gender,
        "patient_id": f"PID-{patient_id}",
        "referred_by": random.choice(DOCTOR_NAMES),
        "sample_date": sample_date.strftime("%d-%b-%Y %I:%M %p"),
        "report_date": report_date.strftime("%d-%b-%Y %I:%M %p"),
    }


def generate_lab_details() -> dict[str, str]:
    """Generate randomized lab/clinic details."""
    pathologist = random.choice(PATHOLOGISTS)
    return {
        "lab_name": random.choice(LAB_NAMES),
        "lab_address": random.choice(LAB_ADDRESSES),
        "lab_accreditation": random.choice(LAB_ACCREDITATIONS),
        "pathologist_name": pathologist[0],
        "pathologist_qualification": pathologist[1],
    }


# ─── Report Generation ────────────────────────────────────────────────────────

def generate_single_report(
    report_type: str,
    output_dir: Path,
    report_index: int,
) -> list[dict[str, Any]]:
    """
    Generate a single lab report image and return ground-truth rows.

    Args:
        report_type: "CBC" or "LIPID"
        output_dir: directory to save the rendered HTML/PNG
        report_index: sequential number for filename

    Returns:
        List of ground-truth dicts (one per test in the report)
    """
    config = REPORT_TYPES[report_type]
    patient = generate_patient_demographics()
    lab = generate_lab_details()
    gender = patient["patient_gender"]

    # Generate test values
    test_rows = []
    gt_rows = []
    filename = f"{report_type.lower()}_{report_index:04d}"

    for test_def in config["tests"]:
        value, normal_low, normal_high, is_abnormal = generate_test_value(test_def, gender)

        # Format reference range string
        decimals = test_def["decimals"]
        if normal_low == 0:
            ref_range_str = f"< {normal_high:.{decimals}f}" if decimals > 0 else f"< {int(normal_high)}"
        else:
            low_str = f"{normal_low:.{decimals}f}" if decimals > 0 else str(int(normal_low))
            high_str = f"{normal_high:.{decimals}f}" if decimals > 0 else str(int(normal_high))
            ref_range_str = f"{low_str} - {high_str}"

        # Determine flag class for HTML styling
        flag_class = ""
        if is_abnormal:
            if isinstance(value, (int, float)):
                flag_class = "flag-high" if value > normal_high else "flag-low"

        # Format value for display
        value_str = f"{value:.{decimals}f}" if decimals > 0 else str(int(value) if isinstance(value, float) else value)

        test_rows.append({
            "test_name": test_def["name"],
            "value": value_str,
            "unit": test_def["unit"],
            "reference_range": ref_range_str,
            "flag_class": flag_class,
        })

        gt_rows.append({
            "image_file": f"{filename}.html",
            "report_type": report_type,
            "test_name": test_def["name"],
            "value": value,
            "unit": test_def["unit"],
            "normal_range_low": normal_low,
            "normal_range_high": normal_high,
            "is_abnormal": is_abnormal,
        })

    # Render HTML from template
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("lab_report_template.html")

    rendered_html = template.render(
        **patient,
        **lab,
        report_type_title=config["title"],
        test_rows=test_rows,
        method=config["method"],
        sample_type=config["sample_type"],
    )

    # Save rendered HTML
    html_path = output_dir / f"{filename}.html"
    html_path.write_text(rendered_html, encoding="utf-8")

    # If imgkit is available, also render to PNG
    if USE_IMGKIT:
        try:
            import imgkit
            png_path = output_dir / f"{filename}.png"
            imgkit.from_string(
                rendered_html,
                str(png_path),
                options={
                    "format": "png",
                    "width": "800",
                    "quality": "90",
                    "encoding": "UTF-8",
                },
            )
            # Update gt_rows to reference PNG
            for row in gt_rows:
                row["image_file"] = f"{filename}.png"
            print(f"  ✓ Generated {filename}.png")
        except Exception as e:
            print(f"  ⚠ imgkit failed ({e}), kept HTML: {filename}.html")
    else:
        print(f"  ✓ Generated {filename}.html")

    return gt_rows


def generate_dataset(
    count: int,
    output_dir: Path,
    gt_csv_path: Path,
    report_types: list[str] | None = None,
    seed: int | None = None,
) -> None:
    """
    Generate a full synthetic dataset of lab reports.

    Args:
        count: total number of reports to generate
        output_dir: directory for rendered reports
        gt_csv_path: path for the ground-truth CSV
        report_types: list of report types to generate (default: all)
        seed: random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    if report_types is None:
        report_types = list(REPORT_TYPES.keys())

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    all_gt_rows: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {rt: 0 for rt in report_types}

    print(f"\n{'=' * 60}")
    print(f"  ClariRx — Synthetic Lab Report Generator")
    print(f"{'=' * 60}")
    print(f"  Reports to generate : {count}")
    print(f"  Report types        : {', '.join(report_types)}")
    print(f"  Output directory    : {output_dir}")
    print(f"  Ground truth CSV    : {gt_csv_path}")
    print(f"  Random seed         : {seed}")
    print(f"{'=' * 60}\n")

    for i in range(count):
        # Alternate between report types (roughly equal distribution)
        report_type = report_types[i % len(report_types)]
        type_counts[report_type] += 1

        gt_rows = generate_single_report(
            report_type=report_type,
            output_dir=output_dir,
            report_index=i + 1,
        )
        all_gt_rows.extend(gt_rows)

    # Write ground-truth CSV
    gt_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_file", "report_type", "test_name", "value",
        "unit", "normal_range_low", "normal_range_high", "is_abnormal",
    ]
    with open(gt_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_gt_rows)

    # Summary
    total_tests = len(all_gt_rows)
    abnormal_count = sum(1 for r in all_gt_rows if r["is_abnormal"])
    abnormal_pct = (abnormal_count / total_tests * 100) if total_tests > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  ✅ Generation Complete!")
    print(f"{'=' * 60}")
    print(f"  Total reports     : {count}")
    for rt, cnt in type_counts.items():
        print(f"    {rt:15s} : {cnt}")
    print(f"  Total test rows   : {total_tests}")
    print(f"  Abnormal values   : {abnormal_count} ({abnormal_pct:.1f}%)")
    print(f"  GT CSV written to : {gt_csv_path}")
    print(f"{'=' * 60}\n")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic lab report images + ground-truth CSV for ClariRx evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="Total number of reports to generate (default: 50)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save rendered reports",
    )
    parser.add_argument(
        "--gt-csv", type=str, default=str(DEFAULT_GT_CSV),
        help="Path for the ground-truth CSV",
    )
    parser.add_argument(
        "--types", nargs="+", choices=list(REPORT_TYPES.keys()), default=None,
        help="Report types to generate (default: all). Options: CBC, LIPID",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--imgkit", action="store_true",
        help="Use imgkit/wkhtmltopdf for PNG rendering (requires wkhtmltopdf installed)",
    )
    args = parser.parse_args()

    global USE_IMGKIT
    if args.imgkit:
        USE_IMGKIT = True

    generate_dataset(
        count=args.count,
        output_dir=Path(args.output_dir),
        gt_csv_path=Path(args.gt_csv),
        report_types=args.types,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
