"""
Interactive visualization dashboard for network intrusion detection.
Built with Dash for real-time monitoring and analysis.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__)

# Sample data for demonstration
def generate_sample_data():
    """Generate sample network traffic data."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=168, freq='h')
    
    normal_traffic = np.random.normal(100, 20, 168)
    intrusions = np.random.poisson(5, 168)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'normal_packets': normal_traffic,
        'intrusions': intrusions,
        'total_packets': normal_traffic + intrusions,
        'attack_rate': (intrusions / (normal_traffic + intrusions) * 100)
    })
    
    return df

# Helper functions to create charts (MUST be defined before layout)
def create_traffic_timeline(df):
    """Create traffic timeline chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['normal_packets'],
        name='Normal Traffic',
        mode='lines',
        fill='tozeroy',
        line=dict(color='#2ecc71', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['intrusions'],
        name='Intrusions',
        mode='lines',
        line=dict(color='#e74c3c', width=2)
    ))
    
    fig.update_layout(
        title='Network Traffic Timeline (Last 7 Days)',
        xaxis_title='Time',
        yaxis_title='Number of Packets',
        hovermode='x unified',
        template='plotly_dark',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_attack_distribution(df):
    """Create attack distribution pie chart."""
    intrusions = df['intrusions'].sum()
    normal = df['normal_packets'].sum()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Normal Traffic', 'Intrusions'],
        values=[normal, intrusions],
        marker=dict(colors=['#2ecc71', '#e74c3c']),
        hole=.3
    )])
    
    fig.update_layout(
        title='Traffic Distribution',
        template='plotly_dark',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_detection_rate_chart(df):
    """Create detection rate chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['attack_rate'],
        mode='lines+markers',
        name='Attack Rate (%)',
        line=dict(color='#f39c12', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Detection Rate Over Time',
        xaxis_title='Time',
        yaxis_title='Attack Rate (%)',
        hovermode='x unified',
        template='plotly_dark',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_protocol_distribution():
    """Create protocol distribution chart."""
    protocols = ['TCP', 'UDP', 'ICMP', 'Other']
    values = [45, 30, 15, 10]
    
    fig = go.Figure(data=[go.Bar(
        x=protocols,
        y=values,
        marker=dict(color=['#3498db', '#2ecc71', '#e74c3c', '#95a5a6'])
    )])
    
    fig.update_layout(
        title='Protocol Distribution',
        xaxis_title='Protocol',
        yaxis_title='Count',
        template='plotly_dark',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    return fig

# Load sample data
df_traffic = generate_sample_data()

# Define app layout
app.layout = html.Div([
    html.Div([
        html.H1('Network Intrusion Detection System Dashboard', 
                className='header-title'),
        html.P('Real-time monitoring and analysis of network traffic',
              className='header-subtitle')
    ], className='header'),
    
    # KPI Cards
    html.Div([
        html.Div([
            html.H3('Total Packets', className='kpi-label'),
            html.H2(f'{int(df_traffic["total_packets"].sum()):,}', className='kpi-value'),
            html.P('Last 7 days', className='kpi-period')
        ], className='kpi-card kpi-primary'),
        
        html.Div([
            html.H3('Intrusions Detected', className='kpi-label'),
            html.H2(f'{int(df_traffic["intrusions"].sum())}', className='kpi-value kpi-alert'),
            html.P('Last 7 days', className='kpi-period')
        ], className='kpi-card kpi-alert-card'),
        
        html.Div([
            html.H3('Detection Rate', className='kpi-label'),
            html.H2(f'{df_traffic["attack_rate"].mean():.2f}%', className='kpi-value'),
            html.P('Average', className='kpi-period')
        ], className='kpi-card kpi-info'),
        
        html.Div([
            html.H3('System Status', className='kpi-label'),
            html.H2('ACTIVE', className='kpi-value kpi-success'),
            html.P('Monitoring', className='kpi-period')
        ], className='kpi-card kpi-success-card'),
    ], className='kpi-container'),
    
    # Charts
    html.Div([
        html.Div([
            dcc.Graph(
                id='traffic-timeline',
                figure=create_traffic_timeline(df_traffic)
            )
        ], className='chart-container'),
        
        html.Div([
            dcc.Graph(
                id='attack-distribution',
                figure=create_attack_distribution(df_traffic)
            )
        ], className='chart-container'),
    ], className='charts-row'),
    
    html.Div([
        html.Div([
            dcc.Graph(
                id='detection-rate-chart',
                figure=create_detection_rate_chart(df_traffic)
            )
        ], className='chart-container'),
        
        html.Div([
            dcc.Graph(
                id='protocol-distribution',
                figure=create_protocol_distribution()
            )
        ], className='chart-container'),
    ], className='charts-row'),
    
    # Recent Alerts Table
    html.Div([
        html.H3('Recent Alerts', className='section-title'),
        html.Table([
            html.Thead(
                html.Tr([
                    html.Th('Timestamp'),
                    html.Th('Source IP'),
                    html.Th('Destination IP'),
                    html.Th('Attack Type'),
                    html.Th('Severity'),
                    html.Th('Status')
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td('2025-12-21 14:32:15'),
                    html.Td('192.168.1.105'),
                    html.Td('10.0.0.50'),
                    html.Td('Port Scan'),
                    html.Td('High', className='severity-high'),
                    html.Td('Blocked', className='status-blocked')
                ]),
                html.Tr([
                    html.Td('2025-12-21 14:28:42'),
                    html.Td('203.0.113.45'),
                    html.Td('10.0.0.1'),
                    html.Td('DDoS Attack'),
                    html.Td('Critical', className='severity-critical'),
                    html.Td('Blocked', className='status-blocked')
                ]),
                html.Tr([
                    html.Td('2025-12-21 14:15:03'),
                    html.Td('198.51.100.89'),
                    html.Td('10.0.0.100'),
                    html.Td('Malware Communication'),
                    html.Td('High', className='severity-high'),
                    html.Td('Quarantined', className='status-quarantined')
                ]),
            ])
        ], className='alerts-table'),
    ], className='section-container'),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=30 * 1000,  # 30 seconds
        n_intervals=0
    )
], className='dashboard-container')

# Callback for auto-refresh
@app.callback(
    Output('traffic-timeline', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_traffic_timeline(n):
    """Update traffic timeline chart."""
    df = generate_sample_data()
    return create_traffic_timeline(df)

# CSS Styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0f0f0f;
                color: #ecf0f1;
                margin: 0;
                padding: 0;
            }}
            
            .dashboard-container {{
                padding: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 2px solid #e74c3c;
                padding-bottom: 20px;
            }}
            
            .header-title {{
                font-size: 2.5em;
                margin: 0;
                color: #ecf0f1;
            }}
            
            .header-subtitle {{
                font-size: 1em;
                color: #95a5a6;
                margin: 10px 0 0 0;
            }}
            
            .kpi-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            
            .kpi-card {{
                background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            
            .kpi-alert-card {{
                border-left-color: #e74c3c;
            }}
            
            .kpi-success-card {{
                border-left-color: #2ecc71;
            }}
            
            .kpi-label {{
                font-size: 0.9em;
                color: #95a5a6;
                margin: 0;
                text-transform: uppercase;
            }}
            
            .kpi-value {{
                font-size: 2em;
                margin: 10px 0;
                color: #ecf0f1;
            }}
            
            .kpi-value.kpi-alert {{
                color: #e74c3c;
            }}
            
            .kpi-value.kpi-success {{
                color: #2ecc71;
            }}
            
            .kpi-period {{
                margin: 0;
                color: #7f8c8d;
                font-size: 0.85em;
            }}
            
            .charts-row {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            
            .chart-container {{
                background: #1a1a1a;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            
            .section-container {{
                background: #1a1a1a;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 40px;
            }}
            
            .section-title {{
                font-size: 1.5em;
                margin-top: 0;
                margin-bottom: 20px;
                color: #ecf0f1;
                border-bottom: 1px solid #34495e;
                padding-bottom: 10px;
            }}
            
            .alerts-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.95em;
            }}
            
            .alerts-table thead {{
                background: #2c3e50;
            }}
            
            .alerts-table th {{
                padding: 12px;
                text-align: left;
                color: #ecf0f1;
                font-weight: 600;
            }}
            
            .alerts-table td {{
                padding: 12px;
                border-bottom: 1px solid #34495e;
            }}
            
            .alerts-table tr:hover {{
                background: #34495e;
            }}
            
            .severity-high {{
                color: #e74c3c;
                font-weight: 600;
            }}
            
            .severity-critical {{
                color: #c0392b;
                font-weight: 600;
            }}
            
            .status-blocked {{
                color: #e74c3c;
                background: rgba(231, 76, 60, 0.1);
                padding: 4px 8px;
                border-radius: 4px;
            }}
            
            .status-quarantined {{
                color: #f39c12;
                background: rgba(243, 156, 18, 0.1);
                padding: 4px 8px;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    logger.info("Starting Dash visualization dashboard...")
    app.run(debug=True, host='0.0.0.0', port=8050)
