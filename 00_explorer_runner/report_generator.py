"""
Report Generator Module
Generates comprehensive reports from test results.
Located at: ./00_explorer_runner/report_generator.py
"""

import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import html


class ReportGenerator:
    """Generates various report formats from test results."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.report_dir = Path(self.config['reporting']['report_dir'])
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all_reports(self, test_results: List[Dict[str, Any]]) -> Dict[str, Path]:
        """Generate all configured report types."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_paths = {}
        
        # Generate JSON report
        if self.config['reporting']['generate_json']:
            json_path = self._generate_json_report(test_results, timestamp)
            report_paths['json'] = json_path
            
        # Generate HTML report
        if self.config['reporting']['generate_html']:
            html_path = self._generate_html_report(test_results, timestamp)
            report_paths['html'] = html_path
            
        # Generate CSV summary
        if self.config['reporting']['generate_csv_summary']:
            csv_path = self._generate_csv_summary(test_results, timestamp)
            report_paths['csv'] = csv_path
            
        return report_paths
    
    def _generate_json_report(self, results: List[Dict[str, Any]], timestamp: str) -> Path:
        """Generate detailed JSON report."""
        json_path = self.report_dir / f'explorer_report_{timestamp}.json'
        
        report_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'config': self.config
            },
            'summary': self._generate_summary(results),
            'results': results
        }
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"JSON report written to {json_path}")
                self.logger.debug(f"JSON report content: {json.dumps(report_data, indent=2)}")
        except Exception as e:
            self.logger.error(f"Failed to write JSON report: {e}")
            raise
        return json_path
    
    def _generate_html_report(self, results: List[Dict[str, Any]], timestamp: str) -> Path:
        """Generate interactive HTML report with detailed results."""
        html_path = self.report_dir / f'explorer_report_{timestamp}.html'
        
        summary = self._generate_summary(results)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIREON Explorer Report - {timestamp}</title>
    <style>
        {self._get_html_styles()}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 NIREON Explorer Test Report</h1>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </header>
        
        <section class="summary">
            <h2>📊 Executive Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{summary['total_runs']}</div>
                    <div class="stat-label">Total Runs</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-value">{summary['successful_runs']}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-card failure">
                    <div class="stat-value">{summary['failed_runs']}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary['success_rate']:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
            </div>
        </section>
        
        <section class="metrics">
            <h2>📈 Key Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Ideas Generated</h3>
                    <div class="metric-value">{summary['total_ideas']}</div>
                    <div class="metric-detail">Avg: {summary['avg_ideas_per_run']:.1f} per run</div>
                </div>
                <div class="metric-card">
                    <h3>Assessment Coverage</h3>
                    <div class="metric-value">{summary['avg_assessment_coverage']:.1f}%</div>
                    <div class="metric-detail">Of all generated ideas</div>
                </div>
                <div class="metric-card">
                    <h3>High Trust Ideas</h3>
                    <div class="metric-value">{summary['total_high_trust']}</div>
                    <div class="metric-detail">Trust score > 6.0</div>
                </div>
                <div class="metric-card">
                    <h3>Proto Tasks</h3>
                    <div class="metric-value">{summary['proto_triggered_count']}</div>
                    <div class="metric-detail">Quantifier activations</div>
                </div>
            </div>
        </section>
        
        <section class="charts">
            <h2>📊 Visualizations</h2>
            <div class="chart-grid">
                <div class="chart-container">
                    <canvas id="successChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="ideasChart"></canvas>
                </div>
            </div>
        </section>
        
        <section class="results">
            <h2>🔍 Detailed Results</h2>
            {self._generate_detailed_results_html(results)}
        </section>
        
        <section class="timeline">
            <h2>⏱️ Execution Timeline</h2>
            {self._generate_timeline_html(results)}
        </section>
    </div>
    
    <script>
        {self._get_chart_scripts(summary, results)}
        {self._get_interaction_scripts()}
    </script>
</body>
</html>"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return html_path
    
    def _generate_csv_summary(self, results: List[Dict[str, Any]], timestamp: str) -> Path:
        """Generate CSV summary for easy analysis."""
        csv_path = self.report_dir / f'explorer_summary_{timestamp}.csv'
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Seed ID', 'Test Passed', 'Duration (s)', 'Ideas Generated',
                'Ideas Assessed', 'Assessment Coverage %', 'High Trust Count',
                'Max Depth', 'Proto Triggered', 'Failure Reason'
            ])
            
            # Data rows
            for result in results:
                stats = result.get('summary_stats', {})
                writer.writerow([
                    result['seed_id'],
                    'Yes' if result['test_passed'] else 'No',
                    f"{result.get('duration_seconds', 0):.2f}",
                    stats.get('total_ideas', 0),
                    stats.get('total_assessed', 0),
                    f"{(stats.get('total_assessed', 0) / stats.get('total_ideas', 1) * 100):.1f}",
                    stats.get('high_trust_count', 0),
                    stats.get('max_depth', 0),
                    'Yes' if stats.get('proto_triggered', False) else 'No',
                    result.get('failure_reason', '')
                ])
                
        return csv_path
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_runs = len(results)
        successful_runs = sum(1 for r in results if r.get('test_passed', False))
        failed_runs = total_runs - successful_runs
        
        # Aggregate metrics
        total_ideas = sum(r.get('summary_stats', {}).get('total_ideas', 0) for r in results)
        total_assessed = sum(r.get('summary_stats', {}).get('total_assessed', 0) for r in results)
        total_high_trust = sum(r.get('summary_stats', {}).get('high_trust_count', 0) for r in results)
        proto_triggered_count = sum(1 for r in results if r.get('summary_stats', {}).get('proto_triggered', False))
        
        # Calculate averages
        avg_ideas = total_ideas / total_runs if total_runs > 0 else 0
        avg_coverage = (total_assessed / total_ideas * 100) if total_ideas > 0 else 0
        
        return {
            'total_runs': total_runs,
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'success_rate': (successful_runs / total_runs * 100) if total_runs > 0 else 0,
            'total_ideas': total_ideas,
            'total_assessed': total_assessed,
            'total_high_trust': total_high_trust,
            'proto_triggered_count': proto_triggered_count,
            'avg_ideas_per_run': avg_ideas,
            'avg_assessment_coverage': avg_coverage
        }
    
    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, #1a1a2e, #0f0f1e);
            border-radius: 1rem;
            border: 1px solid #2a2a3e;
        }
        
        h1 {
            font-size: 3rem;
            background: linear-gradient(45deg, #00ff88, #0088ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .timestamp {
            color: #888;
            font-size: 0.9rem;
        }
        
        section {
            margin-bottom: 3rem;
        }
        
        h2 {
            font-size: 1.8rem;
            margin-bottom: 1.5rem;
            color: #00ff88;
        }
        
        h3 {
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: #0088ff;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
        }
        
        .stat-card {
            background: #1a1a2e;
            padding: 2rem;
            border-radius: 0.75rem;
            border: 1px solid #2a2a3e;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 255, 136, 0.2);
        }
        
        .stat-card.success {
            border-color: #00ff88;
        }
        
        .stat-card.failure {
            border-color: #ff4444;
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            color: #888;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 1px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }
        
        .metric-card {
            background: #1a1a2e;
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid #2a2a3e;
        }
        
        .metric-card h3 {
            color: #0088ff;
            font-size: 1rem;
            margin-bottom: 0.75rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.25rem;
        }
        
        .metric-detail {
            color: #888;
            font-size: 0.9rem;
        }
        
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
        }
        
        .chart-container {
            background: #1a1a2e;
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid #2a2a3e;
        }
        
        .result-card {
            background: #1a1a2e;
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid #2a2a3e;
            margin-bottom: 1rem;
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            cursor: pointer;
        }
        
        .result-header:hover {
            opacity: 0.8;
        }
        
        .result-title {
            font-size: 1.2rem;
            font-weight: bold;
        }
        
        .result-status {
            padding: 0.25rem 0.75rem;
            border-radius: 0.25rem;
            font-size: 0.85rem;
            font-weight: bold;
        }
        
        .status-success {
            background: #00ff88;
            color: #000;
        }
        
        .status-failure {
            background: #ff4444;
            color: #fff;
        }
        
        .result-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .detail-item {
            display: flex;
            flex-direction: column;
        }
        
        .detail-label {
            color: #888;
            font-size: 0.85rem;
        }
        
        .detail-value {
            font-weight: bold;
        }
        
        .timeline {
            position: relative;
            padding-left: 2rem;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 2px;
            background: linear-gradient(to bottom, #00ff88, #0088ff);
        }
        
        .timeline-item {
            position: relative;
            margin-bottom: 1.5rem;
            padding-left: 1.5rem;
        }
        
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -2.5rem;
            top: 0.5rem;
            width: 12px;
            height: 12px;
            background: #00ff88;
            border-radius: 50%;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        }
        
        /* New styles for collapsible content */
        .collapsible {
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            max-height: 0;
        }
        
        .collapsible.expanded {
            max-height: none;
        }
        
        .expand-indicator {
            margin-left: 0.5rem;
            font-size: 1.2rem;
            transition: transform 0.3s;
            display: inline-block;
        }
        
        .expanded .expand-indicator {
            transform: rotate(180deg);
        }
        
        .idea-container {
            background: #0f0f1e;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            border: 1px solid #2a2a3e;
        }
        
        .idea-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.5rem;
        }
        
        .idea-title {
            font-weight: bold;
            color: #00ff88;
            flex: 1;
            margin-right: 1rem;
        }
        
        .idea-text {
            font-size: 0.9rem;
            color: #ccc;
            line-height: 1.5;
            max-height: 100px;
            overflow-y: auto;
            padding-right: 0.5rem;
        }
        
        .trust-score {
            font-weight: bold;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.85rem;
            white-space: nowrap;
        }
        
        .high-trust {
            background: #00ff88;
            color: #000;
        }
        
        .low-trust {
            background: #ff4444;
            color: #fff;
        }
        
        .seed-objective {
            background: #1a1a3e;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            font-style: italic;
            color: #aaa;
        }
        
        .proto-tasks {
            background: #1a2a3e;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            border: 1px solid #0088ff;
        }
        
        .proto-task-item {
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background: #0a1a2e;
            border-radius: 0.25rem;
        }
        
        .proto-task-id {
            font-family: monospace;
            color: #0088ff;
            font-weight: bold;
        }
        
        .event-timeline {
            max-height: 200px;
            overflow-y: auto;
            background: #0f0f1e;
            padding: 0.5rem;
            border-radius: 0.5rem;
            font-size: 0.85rem;
            font-family: monospace;
            margin-top: 1rem;
        }
        
        .event-item {
            margin-bottom: 0.25rem;
            color: #888;
        }
        
        .event-type {
            color: #00ff88;
            font-weight: bold;
        }
        
        .failure-reason {
            background: #ff444433;
            border: 1px solid #ff4444;
            color: #ff8888;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            margin-top: 0.5rem;
        }
        
        /* Scrollbar styling for dark theme */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1a1a2e;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #2a2a3e;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #3a3a4e;
        }
        """
    
    def _get_chart_scripts(self, summary: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
        """Generate Chart.js scripts."""
        return f"""
        // Success/Failure Chart
        const successCtx = document.getElementById('successChart').getContext('2d');
        new Chart(successCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Successful', 'Failed'],
                datasets: [{{
                    data: [{summary['successful_runs']}, {summary['failed_runs']}],
                    backgroundColor: ['#00ff88', '#ff4444'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Test Results',
                        color: '#e0e0e0'
                    }},
                    legend: {{
                        labels: {{
                            color: '#e0e0e0'
                        }}
                    }}
                }}
            }}
        }});
        
        // Ideas per Run Chart
        const ideasData = {json.dumps([r.get('summary_stats', {}).get('total_ideas', 0) for r in results])};
        const seedLabels = {json.dumps([r['seed_id'] for r in results])};
        
        const ideasCtx = document.getElementById('ideasChart').getContext('2d');
        new Chart(ideasCtx, {{
            type: 'bar',
            data: {{
                labels: seedLabels,
                datasets: [{{
                    label: 'Ideas Generated',
                    data: ideasData,
                    backgroundColor: '#0088ff',
                    borderColor: '#0088ff',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Ideas Generated per Seed',
                        color: '#e0e0e0'
                    }},
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{
                            color: '#e0e0e0'
                        }},
                        grid: {{
                            color: '#2a2a3e'
                        }}
                    }},
                    y: {{
                        ticks: {{
                            color: '#e0e0e0'
                        }},
                        grid: {{
                            color: '#2a2a3e'
                        }}
                    }}
                }}
            }}
        }});
        """
    
    def _get_interaction_scripts(self) -> str:
        """Generate JavaScript for interactive elements."""
        return """
        function toggleCollapse(id) {
            const element = document.getElementById(id);
            const header = element.previousElementSibling;
            const indicator = header.querySelector('.expand-indicator');
            
            element.classList.toggle('expanded');
            header.classList.toggle('expanded');
            
            // Animate max-height for smooth transition
            if (element.classList.contains('expanded')) {
                element.style.maxHeight = element.scrollHeight + 'px';
                setTimeout(() => {
                    element.style.maxHeight = 'none';
                }, 300);
            } else {
                element.style.maxHeight = element.scrollHeight + 'px';
                setTimeout(() => {
                    element.style.maxHeight = '0';
                }, 10);
            }
        }
        """
    
    def _generate_detailed_results_html(self, results: List[Dict[str, Any]]) -> str:
        """Generate HTML for detailed results with collapsible sections."""
        html_parts = []
        
        for result in results:
            status_class = 'status-success' if result['test_passed'] else 'status-failure'
            status_text = '✅ PASSED' if result['test_passed'] else '❌ FAILED'
            
            stats = result.get('summary_stats', {})
            seed_id_safe = html.escape(result['seed_id']).replace(' ', '_')
            
            # Main result card
            html_parts.append(f"""
            <div class="result-card">
                <div class="result-header" onclick="toggleCollapse('{seed_id_safe}_details')">
                    <div class="result-title">{html.escape(result['seed_id'])}</div>
                    <div>
                        <span class="result-status {status_class}">{status_text}</span>
                        <span class="expand-indicator">▼</span>
                    </div>
                </div>
                <div class="result-text">{html.escape(result.get('seed_text', '')[:100])}...</div>
                <div class="result-details">
                    <div class="detail-item">
                        <span class="detail-label">Duration</span>
                        <span class="detail-value">{result.get('duration_seconds', 0):.2f}s</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Ideas</span>
                        <span class="detail-value">{stats.get('total_ideas', 0)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Assessed</span>
                        <span class="detail-value">{stats.get('total_assessed', 0)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">High Trust</span>
                        <span class="detail-value">{stats.get('high_trust_count', 0)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Max Depth</span>
                        <span class="detail-value">{stats.get('max_depth', 0)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Proto</span>
                        <span class="detail-value">{'Yes' if stats.get('proto_triggered', False) else 'No'}</span>
                    </div>
                </div>
                {f'<div class="failure-reason">⚠️ {html.escape(result.get("failure_reason", ""))}</div>' if not result['test_passed'] else ''}
                
                <div id="{seed_id_safe}_details" class="collapsible">
                    {self._generate_seed_details_html(result)}
                </div>
            </div>
            """)
        
        return '\n'.join(html_parts)
    
    def _generate_seed_details_html(self, result: Dict[str, Any]) -> str:
        """Generate detailed HTML for a single seed result."""
        parts = []
        
        # Add objective if present
        if 'objective' in result:
            parts.append(f"""
            <div class="seed-objective">
                <strong>Objective:</strong> {html.escape(result['objective'])}
            </div>
            """)
        
        # Add ideas section
        run_data = result.get('run_data', {})
        if run_data:
            ideas_html = self._extract_ideas_html(run_data)
            if ideas_html:
                parts.append(f"""
                <h3>Generated Ideas ({len(ideas_html)} Total)</h3>
                {''.join(ideas_html)}
                """)
        
        # Add proto tasks if present
        proto_tasks = self._extract_proto_tasks(result)
        if proto_tasks:
            parts.append(f"""
            <div class="proto-tasks">
                <h3>Proto Tasks Generated ({len(proto_tasks)})</h3>
                {self._generate_proto_tasks_html(proto_tasks)}
            </div>
            """)
        
        # Add event timeline
        events = run_data.get('events', [])
        if events:
            parts.append(f"""
            <h3>Event Timeline</h3>
            <div class="event-timeline">
                {self._generate_events_timeline_html(events[:20])}  <!-- Limit to first 20 events -->
            </div>
            """)
        
        return '\n'.join(parts)
    
    def _extract_ideas_html(self, run_data: Dict[str, Any]) -> List[str]:
        """Extract and format ideas from run data."""
        ideas_html = []
        
        # Process seed idea variations
        seed_idea = run_data.get('seed_idea', {})
        variations = seed_idea.get('variations', {})
        
        for idea_id, idea_data in variations.items():
            # Extract title from text (first line or sentence)
            text = idea_data.get('text', '')
            lines = text.split('\n')
            title = lines[0].strip('*#').strip() if lines else 'Untitled Idea'
            if len(title) > 100:
                title = title[:97] + '...'
            
            trust_score = idea_data.get('trust_score')
            trust_class = 'high-trust' if trust_score and trust_score > 6.0 else 'low-trust' if trust_score else ''
            
            ideas_html.append(f"""
            <div class="idea-container">
                <div class="idea-header">
                    <div class="idea-title">{html.escape(title)}</div>
                    {f'<span class="trust-score {trust_class}">Trust: {trust_score:.2f}</span>' if trust_score else ''}
                </div>
                <div class="idea-text">{html.escape(text)}</div>
            </div>
            """)
        
        return ideas_html
    
    def _extract_proto_tasks(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract proto tasks from events."""
        proto_tasks = []
        events = result.get('run_data', {}).get('events', [])
        
        for event in events:
            if event.get('type') == 'proto_task_created':
                details = event.get('details', {})
                payload = details.get('payload', {})
                proto_block = payload.get('proto_block', {})
                if proto_block:
                    proto_tasks.append({
                        'id': proto_block.get('id', 'unknown'),
                        'description': proto_block.get('description', ''),
                        'objective': proto_block.get('objective', ''),
                        'eidos': proto_block.get('eidos', 'unknown')
                    })
        
        return proto_tasks
    
    def _generate_proto_tasks_html(self, proto_tasks: List[Dict[str, Any]]) -> str:
        """Generate HTML for proto tasks."""
        parts = []
        for task in proto_tasks:
            parts.append(f"""
            <div class="proto-task-item">
                <div><span class="proto-task-id">{html.escape(task['id'])}</span> ({task['eidos']})</div>
                <div>{html.escape(task['description'])}</div>
            </div>
            """)
        return '\n'.join(parts)
    
    def _generate_events_timeline_html(self, events: List[Dict[str, Any]]) -> str:
        """Generate HTML for events timeline."""
        parts = []
        for event in events:
            timestamp = event.get('timestamp', '')
            if timestamp:
                # Extract just the time portion
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                except:
                    time_str = timestamp
            else:
                time_str = 'unknown'
                
            event_type = event.get('type', 'unknown')
            details = event.get('details', {})
            
            # Format details based on event type
            detail_str = ''
            if event_type == 'idea_generated':
                detail_str = f"idea_id: {details.get('idea_id', 'unknown')[:8]}..."
            elif event_type == 'trust_assessment':
                detail_str = f"score: {details.get('trust_score', 'N/A')}"
            elif event_type == 'proto_task_created':
                payload = details.get('payload', {})
                proto_block = payload.get('proto_block', {})
                detail_str = f"proto: {proto_block.get('id', 'unknown')}"
            
            parts.append(f"""
            <div class="event-item">
                [{time_str}] <span class="event-type">{event_type}</span> {detail_str}
            </div>
            """)
        
        return '\n'.join(parts)
    
    def _generate_timeline_html(self, results: List[Dict[str, Any]]) -> str:
        """Generate execution timeline HTML."""
        timeline_items = []
        
        for result in results:
            start_time = datetime.fromisoformat(result['start_time'])
            timeline_items.append(f"""
            <div class="timeline-item">
                <div class="timeline-time">{start_time.strftime('%H:%M:%S')}</div>
                <div class="timeline-content">
                    <strong>{html.escape(result['seed_id'])}</strong> - 
                    {'✅ Completed' if result['test_passed'] else '❌ Failed'} 
                    in {result.get('duration_seconds', 0):.2f}s
                </div>
            </div>
            """)
        
        return '<div class="timeline">' + '\n'.join(timeline_items) + '</div>'