"""
SPARK - Service Payroll Automation & Reporting Kit
Sarah's Appliance Repair
"""

import streamlit as st
import pandas as pd
import yaml
import os
import csv
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import zipfile
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import requests

def _extract_text_and_delimiter(uploaded_file) -> tuple[str, str]:
    """Read uploaded file bytes once, decode w/ BOM handling, and detect delimiter."""
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        # Streamlit UploadedFile is a BytesIO-like; .read() then we won't reuse the pointer
        data = uploaded_file.read()
    text = data.decode("utf-8-sig", errors="ignore")  # handles BOM, weird chars

    # Try to sniff the delimiter (comma, semicolon, tab, pipe)
    sample = text[:65536]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        delim = dialect.delimiter
    except Exception:
        delim = ","  # safe fallback

    return text, delim

def _read_csv_from_text(text: str, sep: str, header=None) -> pd.DataFrame:
    """Consistent CSV reader using the already-detected delimiter."""
    return pd.read_csv(
        io.StringIO(text),
        header=header,
        sep=sep,
        engine="python",           # tolerant of quotes/embedded delimiters
        dtype=str,
        keep_default_na=False
    )


# Quiet the urllib3 LibreSSL warning (optional; safe to leave)
try:
    import warnings, urllib3
    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
except Exception:
    pass

# -------------------------
# Paths & Branding
# -------------------------
STORAGE_DIR = Path("storage")
ASSETS_DIR = Path("assets")
OUTPUT_DIR = STORAGE_DIR / "output"
SETTINGS_FILE = STORAGE_DIR / "technician_settings.yaml"
LOGO_PATH = ASSETS_DIR / "sarahs_logo.png"
LOGO_URL = "https://appliancerepairidaho.com/wp-content/uploads/2025/01/Final-Sarahs-logo-2.png"

COLORS = {
    'primary': '#11A4E4',
    'secondary': '#0C77BD',
    'navy': '#0A244A',
    'white': '#FFFFFF',
    'metal_light': '#D9D9D9',
    'metal_dark': '#6C6C6C'
}

for directory in [STORAGE_DIR, ASSETS_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

st.set_page_config(page_title="SPARK - Payroll Automation", page_icon="⚡", layout="wide")

st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }}
    .stButton>button {{
        background-color: {COLORS['primary']};
        color: white;
        font-weight: bold;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-size: 1.05rem;
    }}
    .stButton>button:hover {{
        background-color: {COLORS['secondary']};
    }}
    .success-box {{
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session State
# -------------------------
if 'tech_settings' not in st.session_state:
    st.session_state.tech_settings = {}
if 'adjustments' not in st.session_state:
    st.session_state.adjustments = []

# -------------------------
# Helpers: normalization & parsing
# -------------------------
def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s or '').lower())

def _find_col(cols, candidates):
    """Return first df column that matches any alias (exact-normalized, then fuzzy-contains)."""
    norm_map = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    # fuzzy contains
    for cand in candidates:
        ck = _norm(cand)
        for c in cols:
            if ck in _norm(c):
                return c
    return None

def _to_money_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce').fillna(0.0)

def _to_money_val(x) -> float:
    return float(pd.to_numeric(str(x).replace('$','').replace(',',''), errors='coerce') or 0.0)

def _parse_possible_date(s):
    s = str(s).strip()
    if re.fullmatch(r'\d{8}', s):  # YYYYMMDD
        return pd.Timestamp(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
    return pd.to_datetime(s, errors='coerce')

def _is_paycode_included(pc):
    s = str(pc or '').strip()
    return s in ('1', '2', 1, 2)

def _monday_of(date=None):
    if date is None:
        date = datetime.now()
    days_since_monday = date.weekday()
    monday = date - timedelta(days=days_since_monday)
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)

# --- Name canonicalization for tips ("Last, First" -> "first last") ---
def _canon_person_name(s: str) -> str:
    """Normalize to 'first last' (lowercase, single spaces). Accepts 'Last, First' or 'First Last'."""
    s = str(s or '').strip()
    if ',' in s:
        parts = [p.strip() for p in s.split(',', 1)]
        if len(parts) == 2:
            s = f"{parts[1]} {parts[0]}"
    s = re.sub(r'\s+', ' ', s).lower()
    return s

def _canon_full_name_from_settings(settings_entry: dict) -> str:
    """Get canonical 'first last' from settings['full_name']."""
    nm = (settings_entry or {}).get('full_name') or ''
    return _canon_person_name(nm)

def _pick_mode(values):
    vals = [str(v).strip() for v in values if str(v).strip()]
    if not vals:
        return None
    counts = Counter(vals)
    return counts.most_common(1)[0][0]

def _looks_like_initials(s: str) -> bool:
    # MG, PE, DR, MB, etc. (2–3 upper letters)
    return bool(re.fullmatch(r'[A-Z]{2,3}', str(s).strip()))

def detect_tech_from_df_and_filename(df: pd.DataFrame, filename: str) -> str:
    """
    Try to find a tech identifier (initials or name) from:
    - common tech columns (many aliases)
    - 'tech-like' column modes
    - the filename (e.g., 'MG_commissions.csv' -> 'MG')
    """
    cols = list(df.columns)

    # 1) Direct header matches (many aliases)
    tech_header_candidates = [
        'Technician','Technician Name','Tech','Tech Name','Tech_Name','TechName',
        'Technician Initials','Tech Initials','TechInitials','Initials',
        'Assigned Tech','Assigned Technician','Route Tech','Route Technician',
        'SourceTechnician','SD_Name','Employee','Employee Name','User','User Name','SalesRep'
    ]
    tech_col = _find_col(cols, tech_header_candidates)

    if tech_col:
        vals = df[tech_col].dropna().astype(str).str.strip()
        initials_candidates = [v for v in vals if _looks_like_initials(v)]
        if initials_candidates:
            return _pick_mode(initials_candidates)
        mode_val = _pick_mode(vals)
        if mode_val:
            return mode_val

    # 2) Any column containing 'tech'/'technician'
    tech_like_cols = [c for c in cols if 'tech' in c.lower() or 'technician' in c.lower()]
    for c in tech_like_cols:
        vals = df[c].dropna().astype(str).str.strip()
        if not vals.empty:
            ini = [v for v in vals if _looks_like_initials(v)]
            if ini:
                return _pick_mode(ini)
            mv = _pick_mode(vals)
            if mv:
                return mv

    # 3) Fallback: initials from filename (robust; ignores extension)
    base = os.path.basename(filename or '')
    name_noext, _ = os.path.splitext(base)
    candidates = re.findall(r'[A-Z]{2,3}', name_noext)
    for cand in candidates:
        if cand not in ('CSV',):  # ignore common junk
            return cand

    return None

# -------------------------
# Settings I/O
# -------------------------
def load_settings():
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        yaml.dump(settings, f, default_flow_style=False)

# -------------------------
# Logo
# -------------------------
def download_logo():
    if not LOGO_PATH.exists():
        try:
            resp = requests.get(LOGO_URL, timeout=10)
            if resp.status_code == 200:
                with open(LOGO_PATH, 'wb') as f:
                    f.write(resp.content)
        except Exception:
            pass

# -------------------------
# CSV Parsing (Jobs / Tips)
# -------------------------
def parse_job_csv(file, week_start_dt):
    """
    Robust SD commission parser:
      - Flexible headers
      - Weekly filter [week_start, week_start+7)
      - PayCode ∈ {1,2} (only if present in the data)
      - $ → numeric
    Returns: (tech_identifier, cleaned_df_with_comm_columns)
    """
    try:
        raw = pd.read_csv(file, dtype=str)
        cols = list(raw.columns)

        # Date
        date_col = _find_col(cols, ['Date', 'JobDate'])
        if date_col:
            raw['__Date'] = raw[date_col].apply(_parse_possible_date)
        else:
            raw['__Date'] = pd.NaT

        # Tech (robust via headers/filename)
        tech_identifier = detect_tech_from_df_and_filename(raw, getattr(file, 'name', 'job.csv'))

        # Invoice/Job#
        inv_col = _find_col(cols, ['InvcNmbr', 'Invoice', 'Job', 'Job#', 'InvcNmbr.'])

        # PayCode
        pay_col = _find_col(cols, ['PayCode', 'Pay Code'])

        # Commission columns (aliases)
        mr_col = _find_col(cols, ['MrchCmssn','MrchCmmsn','MerchCmmsn','MerchCommission','Commission_Merch'])
        pr_col = _find_col(cols, ['PartsCmssn','PartsCmmsn','PartsCommission','Commission_Parts'])
        sc_col = _find_col(cols, ['ScallCmssn','ServiceCallCmssn','ServiceFee','SVCFee','ScallCommission','Commission_Scall'])
        lb_col = _find_col(cols, ['LbrCmmsn','LbrCmssn','LaborCommission','LaborCmmsn','Commission_Labor'])
        tt_col = _find_col(cols, ['TotalCmmsn','TotalCommission','Commission_Total'])

        df = raw.copy()

        # money coercion
        for col, newname in [
            (mr_col, 'MrchCmssn'), (pr_col, 'PartsCmssn'), (sc_col, 'ScallCmssn'),
            (lb_col, 'LbrCmmsn'), (tt_col, 'TotalCmssn')
        ]:
            if col:
                df[newname] = _to_money_series(df[col])
            else:
                df[newname] = 0.0

        # If TotalCmssn missing, compute as sum of parts
        if df['TotalCmssn'].sum() == 0 and any(df[c].sum() for c in ['MrchCmssn','PartsCmssn','ScallCmssn','LbrCmmsn']):
            df['TotalCmssn'] = df[['MrchCmssn','PartsCmssn','ScallCmssn','LbrCmmsn']].sum(axis=1)

        # Attach normalized columns to display later
        df['InvcNmbr'] = df[inv_col] if inv_col else ''
        df['PayCode'] = df[pay_col] if pay_col else ''

        # Date filter: current week
        week_end = week_start_dt + timedelta(days=7)
        df = df[(df['__Date'] >= pd.Timestamp(week_start_dt)) & (df['__Date'] < pd.Timestamp(week_end))]

        # PayCode filter: ONLY if at least one valid 1/2 exists
        if 'PayCode' in df.columns:
            _pc = df['PayCode'].astype(str).str.strip()
            has_included = _pc.isin(['1', '2']).any()
            if has_included:
                df = df[_pc.isin(['1', '2'])].copy()

        # Keep rows with any commission values
        df = df[(df['TotalCmssn'] != 0) | (df['MrchCmssn'] != 0) | (df['PartsCmssn'] != 0) |
                (df['ScallCmssn'] != 0) | (df['LbrCmmsn'] != 0)]

        return tech_identifier, df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error parsing {getattr(file, 'name', 'job.csv')}: {e}")
        return None, None

def parse_tips_csv(file):
    """
    Parse Square tips CSV for the "Display By Employee" section.
    Specifically handles the format with columns like:
    "Sales Summary Displayed by Employee", "Gross Sales", ..., "Tip"
    
    Returns a DataFrame with columns: ['Employee','Tip','NormName','Initials'].
    Only positive tips are kept.
    """
    def clean_role_noise(s: str) -> str:
        # Remove "(Tech)" or "- Something" noise Square may append
        s = re.sub(r'\(.*?\)', '', str(s))
        s = re.sub(r'\s-\s.*$', '', s)
        return s.strip()

    def normalize_name_last_first_to_first_last(s: str) -> str:
        s = clean_role_noise(s)
        if ',' in s:
            parts = [p.strip() for p in s.split(',', 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                return f"{parts[1]} {parts[0]}"
        return s

    def norm_first_last(s: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z ]+', '', str(s))).strip().lower()

    def initials_first_last(s: str) -> str:
        parts = re.sub(r'[^a-zA-Z ]+', '', str(s)).strip().split()
        if not parts:
            return ''
        if len(parts) == 1:
            return parts[0][0].lower()
        return (parts[0][0] + parts[-1][0]).lower()

    # Read the entire file content
    if hasattr(file, "getvalue"):
        content = file.getvalue().decode("utf-8-sig")
    else:
        file.seek(0)
        content = file.read().decode("utf-8-sig")
    
    lines = content.split('\n')
    
    # Find the "Display By Employee" section
    display_by_employee_start = None
    for i, line in enumerate(lines):
        if 'Display By Employee' in line:
            display_by_employee_start = i
            break
    
    if display_by_employee_start is None:
        st.warning("Could not find 'Display By Employee' section in tips CSV")
        return pd.DataFrame(columns=['Employee','Tip','NormName','Initials'])
    
    # Find the actual data header (skip the section title and any empty lines)
    header_line_idx = None
    data_start_idx = None
    
    for i in range(display_by_employee_start + 1, min(display_by_employee_start + 10, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
            
        # Look for the header line that contains both employee and tip columns
        if 'Sales Summary Displayed by Employee' in line and 'Tip' in line:
            header_line_idx = i
            data_start_idx = i + 1
            break
    
    if header_line_idx is None:
        st.warning("Could not find data header in 'Display By Employee' section")
        return pd.DataFrame(columns=['Employee','Tip','NormName','Initials'])
    
    # Parse the header to find column positions
    header_line = lines[header_line_idx]
    header_columns = [col.strip().strip('"') for col in header_line.split(',')]
    
    # Find the employee name column and tip column indices
    employee_col_idx = None
    tip_col_idx = None
    
    for idx, col in enumerate(header_columns):
        if 'Sales Summary Displayed by Employee' in col:
            employee_col_idx = idx
        elif col == 'Tip':
            tip_col_idx = idx
    
    if employee_col_idx is None or tip_col_idx is None:
        st.warning("Could not find required columns (Employee and Tip) in tips CSV")
        return pd.DataFrame(columns=['Employee','Tip','NormName','Initials'])
    
    # Parse the data rows
    tips_data = []
    
    for i in range(data_start_idx, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
            
        # Skip summary/total rows
        if any(keyword in line.lower() for keyword in ['total', 'subtotal', 'all team members']):
            continue
            
        # Parse the CSV line, handling quoted fields
        reader = csv.reader([line])
        try:
            row = next(reader)
            
            # Ensure we have enough columns
            if len(row) <= max(employee_col_idx, tip_col_idx):
                continue
                
            employee_name = row[employee_col_idx].strip().strip('"')
            tip_amount_str = row[tip_col_idx].strip().strip('"')
            
            # Skip empty employee names
            if not employee_name:
                continue
                
            # Convert tip amount to float
            tip_amount = _to_money_val(tip_amount_str)
            
            if tip_amount > 0:
                tips_data.append({
                    'Employee': employee_name,
                    'Tip': tip_amount
                })
                
        except (csv.Error, StopIteration):
            continue
    
    if not tips_data:
        return pd.DataFrame(columns=['Employee','Tip','NormName','Initials'])
    
    # Create DataFrame and add normalized columns
    tips_df = pd.DataFrame(tips_data)
    
    # Convert "Last, First" -> "First Last"
    tips_df['FirstLast'] = tips_df['Employee'].map(normalize_name_last_first_to_first_last)
    tips_df['NormName'] = tips_df['FirstLast'].map(norm_first_last)
    tips_df['Initials'] = tips_df['FirstLast'].map(initials_first_last)
    
    return tips_df[['Employee', 'Tip', 'NormName', 'Initials']].reset_index(drop=True)

def match_tips_to_tech(tips_df: pd.DataFrame, tech_name: str, tech_initials: str) -> float:
    """
    Safer staged match with nickname support and last-name disambiguation:
      1) exact normalized name
      2) word-boundary contains
      3) nickname/alias substitution (e.g., mateo->matthew, matt->matthew)
      4) initials fallback ONLY if unique; otherwise narrow by last name
    """
    if tips_df is None or tips_df.empty:
        return 0.0

    import re

    # minimal nickname map (add your own as needed)
    NICK_MAP = {
        "matt": "matthew",
        "mateo": "matthew",
        "matthew": "matthew",
        "mark": "mark",
        "syd": "sidney",
        "sid": "sidney",
    }

    def norm_first_last(full_name: str) -> str:
        s = re.sub(r'[^a-zA-Z ]+', '', str(full_name)).strip().lower()
        return re.sub(r'\s+', ' ', s)

    def split_first_last(norm_full: str):
        parts = norm_full.split()
        if not parts:
            return "", ""
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[-1]

    def canonical_first(first: str) -> str:
        return NICK_MAP.get(first, first)

    def initials_first_last(s: str) -> str:
        parts = re.sub(r'[^a-zA-Z ]+', '', str(s)).strip().split()
        if not parts:
            return ''
        if len(parts) == 1:
            return parts[0][0].lower()
        return (parts[0][0] + parts[-1][0]).lower()

    # normalize the tech's name and build variants
    norm_full = norm_first_last(tech_name or '')
    first, last = split_first_last(norm_full)
    canon_first = canonical_first(first) if first else ""
    canon_full = (canon_first + (" " + last if last else "")).strip() if canon_first else norm_full

    want_init = (tech_initials or '').lower().replace('.', '').replace(' ', '')
    if not want_init and norm_full:
        want_init = initials_first_last(norm_full)

    # 1) exact name
    exact_sum = tips_df.loc[tips_df['NormName'] == canon_full, 'Tip'].sum()
    if exact_sum:
        return float(exact_sum)

    # 2) word-boundary contains
    if canon_full:
        pattern = rf'(^|\s){re.escape(canon_full)}(\s|$)'
        contains_sum = tips_df.loc[
            tips_df['NormName'].str.contains(pattern, regex=True, na=False),
            'Tip'
        ].sum()
        if contains_sum:
            return float(contains_sum)

    # 3) try alias on first name only if the original first != canonical (e.g., mateo->matthew)
    if canon_full != norm_full and canon_full:
        alias_sum = tips_df.loc[tips_df['NormName'] == canon_full, 'Tip'].sum()
        if alias_sum:
            return float(alias_sum)

    # 4) initials fallback
    if want_init:
        hits = tips_df.loc[tips_df['Initials'].str.lower() == want_init].copy()
        if len(hits) == 1:
            return float(hits['Tip'].iloc[0])

        # if multiple with same initials, narrow by last name word-boundary
        if last:
            last_pat = rf'(^|\s){re.escape(last)}(\s|$)'
            narrowed = hits.loc[hits['NormName'].str.contains(last_pat, regex=True, na=False)]
            if len(narrowed) == 1:
                return float(narrowed['Tip'].iloc[0])

    return 0.0

# --- Wrapper so the UI code can call one function everywhere
def match_tips_sum_for_tech(tips_df: pd.DataFrame, tech_entry: dict) -> float:
    """Compatibility wrapper used by the UI."""
    full_name = (tech_entry or {}).get('full_name') or ''
    initials  = (tech_entry or {}).get('initials') or ''
    return match_tips_to_tech(tips_df, full_name, initials)

def calculate_commission(df):
    """Sum commission values from cleaned df (already PayCode-filtered (if any) and week-filtered)."""
    if df is None or df.empty:
        return 0.0
    total = float(df.get('TotalCmssn', pd.Series(dtype=float)).sum())
    if total == 0.0:
        total = float(df[['MrchCmssn','PartsCmssn','ScallCmssn','LbrCmmsn']].sum().sum())
    return total

# -------------------------
# PDF builder
# -------------------------
def create_paystub_pdf(tech_name, tech_initials, week_start, branch,
                       commission, tips, additions, deductions, net_pay,
                       jobs_df, adjustments_list):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.5*inch, bottomMargin=0.5*inch,
        leftMargin=0.5*inch, rightMargin=0.5*inch
    )

    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=22, textColor=colors.HexColor(COLORS['navy']),
        alignment=TA_CENTER, spaceAfter=8
    )
    sub_style = ParagraphStyle(
        'Sub', parent=styles['Normal'], fontSize=10,
        textColor=colors.HexColor(COLORS['navy']), alignment=TA_LEFT
    )

    # Logo
    if LOGO_PATH.exists():
        try:
            elements.append(Image(str(LOGO_PATH), width=2.3*inch, height=0.6*inch))
        except Exception:
            pass
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("PAY STUB", title_style))
    elements.append(Spacer(1, 0.1*inch))

    week_str = week_start.strftime('%m/%d/%Y')
    header_data = [
        ['Week:', week_str, '', 'Branch:', branch],
        ['Tech:', tech_name, '', 'Initials:', tech_initials],
    ]
    header_table = Table(header_data, colWidths=[0.9*inch, 2.3*inch, 0.4*inch, 0.9*inch, 1.8*inch])
    header_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
        ('FONT', (3, 0), (3, -1), 'Helvetica-Bold', 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor(COLORS['navy'])),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 0.2*inch))

    comp_data = [
        ['Commission:', f'${commission:,.2f}'],
        ['Tips:', f'${tips:,.2f}'],
        ['Additions:', f'${additions:,.2f}'],
        ['Deductions:', f'-${deductions:,.2f}'],
        ['', ''],
        ['NET PAY:', f'${net_pay:,.2f}'],
    ]
    comp_table = Table(comp_data, colWidths=[2.0*inch, 2.0*inch])
    comp_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, 4), 'Helvetica', 11),
        ('FONT', (0, 5), (-1, 5), 'Helvetica-Bold', 14),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 11),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor(COLORS['navy'])),
        ('LINEABOVE', (0, 5), (-1, 5), 2, colors.HexColor(COLORS['primary'])),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor(COLORS['metal_light'])),
    ]))
    elements.append(comp_table)
    elements.append(Spacer(1, 0.25*inch))

    elements.append(Paragraph("<b>JOB DETAILS</b>", sub_style))
    elements.append(Spacer(1, 0.06*inch))

    # Jobs table (up to 30)
    if jobs_df is not None and not jobs_df.empty:
        show = jobs_df.copy()
        if '__Date' in show.columns:
            show['JobDateDisp'] = pd.to_datetime(show['__Date']).dt.strftime('%m/%d/%Y')
        else:
            show['JobDateDisp'] = ''
        for col in ['MrchCmssn','PartsCmssn','ScallCmssn','LbrCmmsn','TotalCmssn']:
            if col not in show.columns:
                show[col] = 0.0

        rows = [['Job #', 'Job Date', 'Merch', 'Parts', 'Scall', 'Labor', 'Total', 'PayCode']]
        for _, r in show.head(30).iterrows():
            rows.append([
                str(r.get('InvcNmbr', '')),
                str(r.get('JobDateDisp', '')),
                f"${float(r.get('MrchCmssn',0.0)):,.2f}",
                f"${float(r.get('PartsCmssn',0.0)):,.2f}",
                f"${float(r.get('ScallCmssn',0.0)):,.2f}",
                f"${float(r.get('LbrCmmsn',0.0)):,.2f}",
                f"${float(r.get('TotalCmssn',0.0)):,.2f}",
                str(r.get('PayCode',''))
            ])

        # Totals
        rows.append([
            'TOTAL', '',
            f"${show['MrchCmssn'].sum():,.2f}",
            f"${show['PartsCmssn'].sum():,.2f}",
            f"${show['ScallCmssn'].sum():,.2f}",
            f"${show['LbrCmmsn'].sum():,.2f}",
            f"${show['TotalCmssn'].sum():,.2f}",
            ''
        ])

        job_table = Table(rows, colWidths=[0.9*inch, 0.9*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.8*inch, 0.6*inch])
        job_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
            ('FONT', (0, 1), (-1, -2), 'Helvetica', 8),
            ('FONT', (0, -1), (-1, -1), 'Helvetica-Bold', 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(COLORS['navy'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (2, 1), (-2, -2), 'RIGHT'),
            ('ALIGN', (0, 0), (1, -1), 'LEFT'),
            ('ALIGN', (-1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor(COLORS['metal_light'])]),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor(COLORS['metal_light'])),
        ]))
        elements.append(job_table)
    else:
        elements.append(Paragraph("No jobs this week", styles['Normal']))

    elements.append(Spacer(1, 0.2*inch))

    if adjustments_list:
        elements.append(Paragraph("<b>ADJUSTMENTS (Additions & Deductions)</b>", sub_style))
        elements.append(Spacer(1, 0.06*inch))
        adj_rows = [['Tech Name', 'Job #', 'Type', 'Amount', 'Reason']]
        for a in adjustments_list:
            amt = float(a['amount'])
            adj_rows.append([
                a['tech'], a['job'], a['type'],
                f"${amt:,.2f}" if a['type'] == 'Addition' else f"-${amt:,.2f}",
                a['reason']
            ])
        adj_table = Table(adj_rows, colWidths=[1.6*inch, 0.9*inch, 1.1*inch, 1.0*inch, 2.3*inch])
        adj_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 8),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(COLORS['navy'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (3, 1), (3, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor(COLORS['metal_light'])]),
        ]))
        elements.append(adj_table)
        elements.append(Spacer(1, 0.1*inch))

    disclaimer = Paragraph(
        "<i>Please note: This pay stub includes Commission, Tips, and Additions/Deductions only. "
        "PTO, hourly pay, and time off details are available on your OnPay pay stub.</i>",
        ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8,
                       textColor=colors.HexColor(COLORS['metal_dark']), alignment=TA_LEFT)
    )
    elements.append(disclaimer)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -------------------------
# App UI
# -------------------------
def main():
    st.markdown("""
    <div class="main-header">
        <h1>⚡ SPARK</h1>
        <h3>Service Payroll Automation & Reporting Kit</h3>
        <p>Sarah's Appliance Repair</p>
    </div>
    """, unsafe_allow_html=True)

    download_logo()
    settings = load_settings()

    # Sidebar
    with st.sidebar:
        st.image(str(LOGO_PATH) if LOGO_PATH.exists() else LOGO_URL, width=200)
        st.markdown("---")
        week_start = st.date_input("Week Start (Monday)", value=_monday_of(), help="Select the Monday of the pay period")
        week_start = datetime.combine(week_start, datetime.min.time())
        st.markdown("---")
        st.metric("Settings File", "✓ Active" if settings else "⚠ Empty")
        st.info("SPARK build: 2025-10-29 13:40")

    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Files", "👥 Technicians", "📝 Adjustments", "▶️ Run Payroll"])

    # TAB 1
    with tab1:
        st.header("Step 1: Upload Weekly Files")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Job CSVs (one per technician)")
            job_files = st.file_uploader("Upload job CSVs", type=['csv'], accept_multiple_files=True, key="job_files")
            if job_files:
                st.success(f"✓ {len(job_files)} job files uploaded")
                for f in job_files:
                    st.text(f"  • {f.name}")

        with col2:
            st.subheader("Tips CSV (Square)")
            tips_file = st.file_uploader("Upload Square tips report", type=['csv'], key="tips_file")
            if tips_file:
                st.success(f"✓ Tips file uploaded: {tips_file.name}")

        if job_files and tips_file and st.button("🔍 Process & Detect Technicians", type="primary"):
            with st.spinner("Processing files..."):
                tips_df = parse_tips_csv(tips_file)
                st.session_state['tips_df'] = tips_df

                detected = {}
                for jf in job_files:
                    tech_id, jobs_df = parse_job_csv(jf, week_start)
                    # Debug line (visible if you want): uncomment to see raw detections on upload
                    # st.write("Debug upload:", jf.name, "→ tech_id:", tech_id, "| rows:", 0 if jobs_df is None else len(jobs_df))
                    if tech_id and jobs_df is not None:
                        detected[tech_id] = {
                            'jobs_df': jobs_df,
                            'commission': calculate_commission(jobs_df),
                            'file': jf.name
                        }

                st.session_state['detected_techs'] = detected
                st.session_state['week_start'] = week_start

                # seed settings from detected initials/names
                for tech_id in detected.keys():
                    if tech_id not in settings:
                        settings[tech_id] = {
                            'full_name': '',            # set in Tab 2
                            'initials': tech_id,
                            'branch': 'New Mexico',
                            'commission_rate': 0.0,     # optional for future calc
                            'active': True
                        }
                save_settings(settings)
                st.session_state.tech_settings = settings
                st.success(f"✓ Detected {len(detected)} technicians!")
                st.rerun()

    # TAB 2
    with tab2:
        st.header("Step 2: Configure Technicians")
        if 'detected_techs' in st.session_state:
            detected = st.session_state['detected_techs']
            settings = st.session_state.get('tech_settings', load_settings())
            st.info(f"📊 Found {len(detected)} technicians from uploaded files")
            for tech_id, data in detected.items():
                with st.expander(f"⚙️ {tech_id} - {data['file']}", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        full_name = st.text_input("Full Name", value=settings.get(tech_id, {}).get('full_name', ''), key=f"name_{tech_id}", placeholder="e.g., Mateo Grant")
                    with col2:
                        branch = st.selectbox("Branch", options=['Idaho', 'New Mexico'],
                                              index=0 if settings.get(tech_id, {}).get('branch') == 'Idaho' else 1,
                                              key=f"branch_{tech_id}")
                    with col3:
                        commission_rate = st.number_input("Commission %", min_value=0.0, max_value=100.0,
                                                          value=settings.get(tech_id, {}).get('commission_rate', 0.0),
                                                          step=0.1, key=f"rate_{tech_id}")
                    with col4:
                        active = st.checkbox("Active", value=settings.get(tech_id, {}).get('active', True),
                                             key=f"active_{tech_id}")

                    settings[tech_id] = {
                        'full_name': full_name or tech_id,
                        'initials': tech_id,
                        'branch': branch,
                        'commission_rate': commission_rate,
                        'active': active
                    }
                    st.markdown(f"**Preview:** Commission detected from CSV: ${data['commission']:,.2f} | Jobs: {len(data['jobs_df'])}")

            if st.button("💾 Save All Technician Settings", type="primary"):
                save_settings(settings)
                st.session_state.tech_settings = settings
                st.success("✓ Settings saved!")
        else:
            st.warning("⚠️ Please upload and process files first (Tab 1)")

    # TAB 3
    with tab3:
        st.header("Step 3: Enter Adjustments (Optional)")
        if 'detected_techs' in st.session_state:
            detected = st.session_state['detected_techs']
            tech_list = list(detected.keys())
            st.markdown("Add any additions or deductions for this pay period:")
            with st.form("adjustment_form"):
                c1, c2, c3, c4, c5 = st.columns([2, 1, 2, 2, 1])
                with c1: adj_tech = st.selectbox("Tech (initials/name)", options=tech_list)
                with c2: adj_job = st.text_input("Job #", value="0")
                with c3: adj_type = st.selectbox("Type", options=["Addition", "Deduction"])
                with c4: adj_reason = st.text_input("Reason")
                with c5: adj_amount = st.number_input("Amount", min_value=0.0, step=0.01)
                if st.form_submit_button("➕ Add Adjustment") and adj_amount > 0:
                    st.session_state.adjustments.append({
                        'tech': adj_tech, 'job': adj_job, 'type': adj_type,
                        'reason': adj_reason, 'amount': adj_amount
                    })
                    st.success("✓ Adjustment added!")
                    st.rerun()

            if st.session_state.adjustments:
                st.markdown("---")
                st.subheader("Current Adjustments")
                for idx, adj in enumerate(st.session_state.adjustments):
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        sign = "+" if adj['type'] == 'Addition' else "-"
                        st.text(f"{adj['tech']} | Job {adj['job']} | {adj['type']} | {sign}${adj['amount']:,.2f} | {adj['reason']}")
                    with c2:
                        if st.button("🗑️", key=f"del_{idx}"):
                            st.session_state.adjustments.pop(idx)
                            st.rerun()
                if st.button("🗑️ Clear All Adjustments"):
                    st.session_state.adjustments = []
                    st.rerun()
        else:
            st.warning("⚠️ Please upload and process files first (Tab 1)")

    # TAB 4
    with tab4:
        st.header("Step 4: Generate Paystubs")
        if 'detected_techs' not in st.session_state:
            st.warning("⚠️ Please upload and process files first (Tab 1)")
        else:
            detected = st.session_state['detected_techs']
            settings = st.session_state.get('tech_settings', load_settings())
            tips_df = st.session_state.get('tips_df', pd.DataFrame())
            week_start = st.session_state.get('week_start')

            st.subheader("📊 Payroll Summary")

            # --- Debug panel (to diagnose empty commissions/tips quickly) ---
            with st.expander("🔎 Debug: job rows, dates & tips", expanded=False):
                if tips_df is None or tips_df.empty:
                    st.warning("Tips DF is empty. Re-upload the Square tips CSV in Tab 1.")
                else:
                    st.write("Tips sample:", tips_df.head(8))
                for tech_id, data in detected.items():
                    df = data.get('jobs_df')
                    st.write(f"**{tech_id}** — rows in jobs_df:", 0 if df is None else len(df))
                    if df is not None and not df.empty:
                        # show date range and paycodes present
                        if '__Date' in df.columns:
                            try:
                                mind = pd.to_datetime(df['__Date']).min()
                                maxd = pd.to_datetime(df['__Date']).max()
                                st.write("Dates:", str(mind.date()) if pd.notna(mind) else "n/a",
                                         "→", str(maxd.date()) if pd.notna(maxd) else "n/a")
                            except Exception:
                                pass
                        if 'PayCode' in df.columns:
                            st.write("Unique PayCodes:", sorted(df['PayCode'].astype(str).str.strip().unique()))

            summary_rows = []
            for tech_id, data in detected.items():
                if settings.get(tech_id, {}).get('active', True):
                    tech_entry = settings.get(tech_id, {})
                    tech_name = tech_entry.get('full_name', tech_id)
                    branch = tech_entry.get('branch', 'New Mexico')
                    commission = calculate_commission(data['jobs_df'])
                    tips_sum = match_tips_sum_for_tech(tips_df, tech_entry)

                    tech_adjustments = [a for a in st.session_state.adjustments if a['tech'] == tech_id]
                    additions = sum(a['amount'] for a in tech_adjustments if a['type'] == 'Addition')
                    deductions = sum(a['amount'] for a in tech_adjustments if a['type'] == 'Deduction')
                    net_pay = commission + tips_sum + additions - deductions

                    summary_rows.append({
                        'Tech': tech_name, 'Initials/ID': tech_id, 'Branch': branch,
                        'Jobs': len(data['jobs_df']),
                        'Commission': commission, 'Tips': tips_sum,
                        'Additions': additions, 'Deductions': deductions, 'Net Pay': net_pay
                    })

            # Warn if all commissions are $0 (common sign of wrong week)
            if summary_rows and all(abs(r['Commission']) < 0.005 for r in summary_rows):
                st.warning("All commissions are $0. Double-check the **Week Start (Monday)** in the sidebar matches the job dates in your CSVs.")

            if summary_rows:
                s_df = pd.DataFrame(summary_rows)
                st.dataframe(
                    s_df.style.format({
                        'Commission': '${:,.2f}',
                        'Tips': '${:,.2f}',
                        'Additions': '${:,.2f}',
                        'Deductions': '${:,.2f}',
                        'Net Pay': '${:,.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown(f"**TOTALS:** Commission: ${s_df['Commission'].sum():,.2f} | Tips: ${s_df['Tips'].sum():,.2f} | Net Pay: ${s_df['Net Pay'].sum():,.2f}")

            st.markdown("---")

            if st.button("🚀 RUN PAYROLL - Generate All PDFs", type="primary"):
                with st.spinner("Generating paystubs..."):
                    week_str = week_start.strftime('%Y-%m-%d')
                    output_folder = OUTPUT_DIR / f"payroll_{week_str}"
                    output_folder.mkdir(exist_ok=True)

                    pdf_files = []
                    progress = st.progress(0)
                    active_techs = [t for t in detected.keys() if settings.get(t, {}).get('active', True)]
                    for idx, tech_id in enumerate(active_techs, start=1):
                        progress.progress(idx / max(1, len(active_techs)))

                        tech_entry = settings.get(tech_id, {})
                        tech_name = tech_entry.get('full_name', tech_id)
                        branch = tech_entry.get('branch', 'New Mexico')

                        commission = calculate_commission(detected[tech_id]['jobs_df'])
                        tips_sum = match_tips_sum_for_tech(tips_df, tech_entry)

                        tech_adjustments = [a for a in st.session_state.adjustments if a['tech'] == tech_id]
                        additions = sum(a['amount'] for a in tech_adjustments if a['type'] == 'Addition')
                        deductions = sum(a['amount'] for a in tech_adjustments if a['type'] == 'Deduction')
                        net_pay = commission + tips_sum + additions - deductions

                        pdf_buffer = create_paystub_pdf(
                            tech_name=tech_name, tech_initials=tech_id, week_start=week_start, branch=branch,
                            commission=commission, tips=tips_sum, additions=additions, deductions=deductions, net_pay=net_pay,
                            jobs_df=detected[tech_id]['jobs_df'], adjustments_list=tech_adjustments
                        )

                        safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', tech_name)
                        pdf_filename = f"paystub_{safe_name}_{week_str}.pdf"
                        with open(output_folder / pdf_filename, 'wb') as f:
                            f.write(pdf_buffer.read())
                        pdf_files.append(output_folder / pdf_filename)

                    # summary.csv
                    pd.DataFrame(summary_rows).to_csv(output_folder / "summary.csv", index=False)

                    # zip
                    zip_path = OUTPUT_DIR / f"payroll_{week_str}.zip"
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for p in pdf_files:
                            zipf.write(p, p.name)
                        zipf.write(output_folder / "summary.csv", "summary.csv")

                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Payroll Complete!</h3>
                        <p><b>{len(pdf_files)} paystubs generated</b></p>
                        <p>Week: {week_start.strftime('%m/%d/%Y')}</p>
                        <p>Total Net Pay: ${pd.DataFrame(summary_rows)['Net Pay'].sum():,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="📥 DOWNLOAD ZIP (All Paystubs + Summary)",
                            data=f.read(),
                            file_name=f"payroll_{week_str}.zip",
                            mime="application/zip",
                            type="primary",
                            use_container_width=True
                        )

if __name__ == "__main__":
    main()