# NIREON V4 Explorer Test Runner

A comprehensive, modular test runner for validating NIREON's end-to-end generative flow.

## 📁 Directory Structure

```
./00_explorer_runner/
├── run_explorer.py        # Main entry point
├── config.yaml           # Configuration file
├── orchestrator.py       # Test orchestration logic
├── result_capturer.py    # Signal capture and state tracking
├── report_generator.py   # Report generation (HTML, JSON, CSV)
├── debug_helpers.py      # Debug utilities
├── utils.py             # Common utilities
├── reports/             # Generated reports directory
└── README.md            # This file
```

## 🚀 Quick Start

1. **Basic run with default config:**
   ```bash
   python ./00_explorer_runner/run_explorer.py
   ```

2. **Run specific seeds:**
   ```bash
   python ./00_explorer_runner/run_explorer.py --seeds retail_survival education_future
   ```

3. **Run with custom timeout:**
   ```bash
   python ./00_explorer_runner/run_explorer.py --timeout 180
   ```

4. **Run in parallel mode:**
   ```bash
   python ./00_explorer_runner/run_explorer.py --parallel
   ```

5. **Dry run (see what would be executed):**
   ```bash
   python ./00_explorer_runner/run_explorer.py --dry-run
   ```

## 📋 Configuration

The `config.yaml` file contains all configuration options:

### Execution Settings
- `iterations`: Number of times to run each seed
- `timeout`: Maximum time to wait for completion (seconds)
- `parallel_execution`: Run multiple seeds simultaneously
- `retry_on_failure`: Automatically retry failed runs

### Seed Ideas
Configure your test seeds with:
- `id`: Unique identifier
- `text`: The seed idea text
- `objective`: What you want the system to achieve
- `tags`: Categories for organization

### Reporting
- `generate_html`: Create interactive HTML reports
- `generate_json`: Create detailed JSON data
- `generate_csv_summary`: Create CSV for data analysis
- `include_visualizations`: Add charts and graphs

## 📊 Generated Reports

Reports are saved in `./00_explorer_runner/reports/` with timestamps:

1. **HTML Report** (`explorer_report_YYYYMMDD_HHMMSS.html`)
   - Interactive dashboard with charts
   - Executive summary
   - Detailed results for each seed
   - Visual timeline of execution

2. **JSON Report** (`explorer_report_YYYYMMDD_HHMMSS.json`)
   - Complete data structure
   - All captured events
   - Full idea hierarchy

3. **CSV Summary** (`explorer_summary_YYYYMMDD_HHMMSS.csv`)
   - Tabular summary for spreadsheet analysis
   - Key metrics per seed

## 🔍 Command Line Options

```
--config, -c PATH      Path to config file (default: ./config.yaml)
--seeds, -s IDS        Specific seed IDs to run
--timeout, -t SECONDS  Override timeout from config
--iterations, -n NUM   Override iterations from config
--parallel, -p         Run seeds in parallel
--no-report           Skip report generation
--debug               Enable debug logging
--dry-run             Show what would run without executing
```

## 🎯 Key Features

### 1. **Robust Completion Detection**
- Waits for ALL ideas to be assessed, not just the first completion signal
- Prevents premature test termination

### 2. **Comprehensive State Tracking**
- Tracks every generated idea
- Monitors assessment coverage
- Records high-trust ideas
- Detects Proto task triggers

### 3. **Rich Reporting**
- Interactive HTML dashboards
- Beautiful dark-mode UI
- Real-time charts using Chart.js
- Execution timeline visualization

### 4. **Debug Support**
- Signal flow tracking
- Reactor rules inspection
- Component verification
- Detailed logging

### 5. **Parallel Execution**
- Run multiple seeds simultaneously
- Automatic error handling
- Consolidated reporting

## 📈 Metrics Tracked

- **Ideas Generated**: Total count and per-run average
- **Assessment Coverage**: Percentage of ideas that received trust scores
- **High Trust Ideas**: Ideas scoring above threshold (default: 6.0)
- **Proto Tasks**: Quantifier agent activations
- **Max Depth**: Deepest level of idea generation
- **Execution Time**: Duration of each run

## 🛠️ Adding New Seeds

1. Edit `config.yaml`
2. Add a new seed entry:
   ```yaml
   - id: "your_seed_id"
     text: "Your seed idea text"
     objective: "What you want to achieve"
     tags: ["category1", "category2"]
   ```

## 🐛 Troubleshooting

### Test Timeouts
- Increase timeout: `--timeout 300`
- Check if ideas are being generated but not assessed
- Enable debug mode: `--debug`

### Missing Components
- Verify bootstrap succeeded
- Check reactor rules are loaded
- Ensure quantifier agent is registered

### No Proto Tasks Triggered
- Verify high-trust ideas exist (>6.0)
- Check quantifier rule is enabled
- Confirm advanced.yaml is loaded

## 🎨 Customization

### Custom Report Themes
Edit the CSS in `report_generator.py` to customize the HTML report appearance.

### Additional Metrics
Add new metrics in `ResultCapturer.get_summary_stats()`.

### New Visualizations
Extend `_get_chart_scripts()` in `report_generator.py` to add more charts.

## 📝 Example Output

```
================================================================================
NIREON V4 EXPLORER TEST RUNNER
================================================================================
Start Time: 2024-01-15 14:30:00
Config File: ./00_explorer_runner/config.yaml
Seeds to Process: 4
  - retail_survival: How can a brick-and-mortar electronics retailer...
  - education_future: What would a truly personalized education system...
  - climate_innovation: How can we create carbon-negative cities that...
  - health_longevity: What breakthrough approaches could extend human...

🚀 Bootstrapping NIREON system...
✅ Bootstrap complete: 45 components loaded

🧪 Starting test execution...

============================================================
Running seed 1/4: retail_survival
============================================================
✅ All ideas assessed successfully
🎉🎉🎉 QuantifierAgent Triggered! Proto task created.
✅ Proto execution successful!

📊 Generating reports...
Reports generated:
  - html: ./00_explorer_runner/reports/explorer_report_20240115_143245.html
  - json: ./00_explorer_runner/reports/explorer_report_20240115_143245.json
  - csv: ./00_explorer_runner/reports/explorer_summary_20240115_143245.csv

================================================================================
EXECUTION SUMMARY
================================================================================
Total Runs: 4
✅ Successful: 3
❌ Failed: 1

Failed Runs:
  - health_longevity: Timeout after 120s
```