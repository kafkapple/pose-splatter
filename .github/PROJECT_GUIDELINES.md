# Project Organization Guidelines

## 📁 Directory Structure

```
pose-splatter/
├── src/                    # Source code
├── configs/                # Configuration files
├── data/                   # Data directory
├── output/                 # Output results
├── reports/                # All reports and documentation
│   ├── README.md          # Reports index
│   ├── analysis/          # Analysis reports
│   └── notes/             # Research notes
├── scripts/               # Utility scripts
├── tests/                 # Test files
└── docs/                  # Additional documentation
```

## 📝 Report File Guidelines

### Where to Save Reports

**✅ CORRECT**: Save to `reports/` directory
```bash
# Visualization reports
reports/VISUALIZATION_REPORT.md
reports/SAFE_EXECUTION_GUIDE.md

# Dated technical reports
reports/251110_pose_splatter_visualization.md

# Analysis reports
reports/analysis/251110_rendering_quality.md
```

**❌ INCORRECT**: Do NOT save to project root
```bash
# Wrong - clutters root directory
./VISUALIZATION_REPORT.md
./WORK_SUMMARY.md
```

### File Naming Conventions

1. **General Reports**: `UPPERCASE_TITLE.md`
   - `VISUALIZATION_REPORT.md`
   - `SAFE_EXECUTION_GUIDE.md`

2. **Dated Reports**: `YYMMDD_topic.md`
   - `251110_pose_splatter_visualization.md`
   - `251109_experiment_baseline.md`

3. **Analysis**: `YYMMDD_analysis_topic.md`
   - `251110_analysis_rendering_quality.md`

4. **Notes**: `YYMMDD_notes_topic.md`
   - `251110_notes_meeting.md`

## 🔧 Creating New Documentation

### Step 1: Choose Location
```bash
# Visualization/implementation reports
reports/

# Analysis reports
reports/analysis/

# Research notes
reports/notes/

# Experiment reports
docs/reports/
```

### Step 2: Use Template
See `reports/README.md` for templates

### Step 3: Update Indexes
- Update `reports/README.md` index
- Update main `README.md` documentation section
- Add entry to `reports/CHANGELOG.md` if applicable

## 📊 Root Directory Rules

### Files That SHOULD Be in Root
- `README.md` - Main project README
- `LICENSE` - License file
- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment
- `.gitignore` - Git ignore rules
- Core Python scripts (e.g., `train_script.py`, `render_image.py`)

### Files That Should NOT Be in Root
- ❌ Detailed reports (use `reports/`)
- ❌ Analysis documents (use `reports/analysis/`)
- ❌ Research notes (use `reports/notes/`)
- ❌ Work summaries (use `reports/`)
- ❌ Change logs (use `reports/CHANGELOG.md`)

## 🎯 Quick Reference

### Creating a New Report
```bash
# 1. Decide type and location
TYPE="visualization"  # or "analysis", "notes"
DATE=$(date +%y%m%d)

# 2. Create file
touch reports/${DATE}_${TYPE}_topic.md

# 3. Update index
echo "- [Topic](${DATE}_${TYPE}_topic.md)" >> reports/README.md
```

### Moving Existing Reports
```bash
# Move to reports directory
mv *.md reports/

# Except main README
mv reports/README.md .

# Update links in main README
# Edit README.md to point to reports/ directory
```

## 📚 Documentation Types

| Type | Location | Example |
|------|----------|---------|
| Main README | Root | `README.md` |
| Reports | `reports/` | `VISUALIZATION_REPORT.md` |
| Analysis | `reports/analysis/` | `251110_analysis_*.md` |
| Notes | `reports/notes/` | `251110_notes_*.md` |
| Experiments | `docs/reports/` | `251109_experiment_*.md` |
| API Docs | `docs/api/` | `api_reference.md` |

---

**Enforcement**: All contributors should follow these guidelines to maintain project organization.

**Last Updated**: 2025-11-10
