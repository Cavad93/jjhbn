#!/usr/bin/env python3
"""
Paper Trading Dashboard

Interactive real-time dashboard for monitoring paper trading bot.
Uses Plotly + Dash for visualization.

Features:
- Real-time equity curve
- Open positions table
- Trade history
- Performance metrics
- Auto-refresh every 10 seconds

Usage:
    python3 paper_dashboard.py
    Open browser: http://localhost:8050
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Dash & Plotly
try:
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output
    import plotly.graph_objs as go
    import plotly.express as px
except ImportError:
    print("ERROR: Dash not installed. Install with:")
    print("pip install dash plotly pandas")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_trading.paper_exchange import PaperExchange

# ============================================================================
# CONFIGURATION
# ============================================================================

class DashboardConfig:
    """Dashboard configuration"""
    DATA_DIR = Path(__file__).parent / 'data'

    # Files
    TRADES_CSV = DATA_DIR / 'paper_trades.csv'
    EXCHANGE_STATE = DATA_DIR / 'paper_exchange_state.json'
    BOT_STATE = DATA_DIR / 'paper_bot_state.json'

    # Refresh interval
    REFRESH_INTERVAL = 10000  # 10 seconds in milliseconds

    # Port
    PORT = 8050
    HOST = '0.0.0.0'  # Доступен извне


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Loads data from paper trading bot files"""

    def __init__(self, config: DashboardConfig):
        self.config = config

    def load_exchange_state(self) -> Optional[Dict]:
        """Loads PaperExchange state"""
        if not self.config.EXCHANGE_STATE.exists():
            return None

        try:
            with open(self.config.EXCHANGE_STATE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading exchange state: {e}")
            return None

    def load_bot_state(self) -> Optional[Dict]:
        """Loads bot state"""
        if not self.config.BOT_STATE.exists():
            return None

        try:
            with open(self.config.BOT_STATE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading bot state: {e}")
            return None

    def load_trades(self) -> pd.DataFrame:
        """Loads trade history from CSV"""
        if not self.config.TRADES_CSV.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.config.TRADES_CSV)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error loading trades: {e}")
            return pd.DataFrame()

    def get_equity_history(self) -> List[Dict]:
        """Gets equity history from exchange state"""
        state = self.load_exchange_state()
        if not state:
            return []

        # Check if equity_history exists in state (preferred)
        if 'equity_history' in state and state['equity_history']:
            return state['equity_history']

        # Fallback: create simple equity history
        initial = state.get('initial_capital', 1000.0)
        current_equity = state.get('balance', initial)

        now = datetime.now().timestamp()
        day_ago = now - 86400

        return [
            {'timestamp': day_ago, 'equity': initial},
            {'timestamp': now, 'equity': current_equity}
        ]


# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """Calculates trading metrics"""

    @staticmethod
    def calculate_metrics(trades_df: pd.DataFrame, state: Dict) -> Dict:
        """Calculates all performance metrics"""
        if trades_df.empty or 'pnl' not in trades_df.columns:
            return MetricsCalculator._empty_metrics()

        # Filter closed trades
        closed_trades = trades_df[trades_df['action'] == 'CLOSE'].copy()

        if closed_trades.empty:
            return MetricsCalculator._empty_metrics()

        # Basic stats
        total_trades = len(closed_trades)
        wins = closed_trades[closed_trades['pnl'] > 0]
        losses = closed_trades[closed_trades['pnl'] <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        # PnL stats
        total_pnl = closed_trades['pnl'].sum()
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0

        # Profit factor
        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

        # Returns for Sharpe/Sortino
        returns = closed_trades['pnl_percent'].values / 100 if 'pnl_percent' in closed_trades else []

        # Sharpe Ratio (assuming 252 trading periods/year, 0% risk-free rate)
        sharpe = 0
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                sharpe = (mean_return / std_return) * np.sqrt(252)

        # Sortino Ratio (using downside deviation)
        sortino = 0
        if len(returns) > 1:
            mean_return = np.mean(returns)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    sortino = (mean_return / downside_std) * np.sqrt(252)

        # Max Drawdown
        initial_capital = state.get('initial_capital', 1000) if state else 1000
        cumulative_pnl = closed_trades['pnl'].cumsum()
        equity_curve = initial_capital + cumulative_pnl
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0

        # Average trade duration
        avg_duration_seconds = 0
        if 'duration' in closed_trades.columns:
            avg_duration_seconds = closed_trades['duration'].mean()

        # ROI
        roi = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': win_count,  # Added for test compatibility
            'losing_trades': loss_count,  # Added for test compatibility
            'wins': win_count,
            'losses': loss_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,  # Added for test compatibility
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'avg_duration': avg_duration_seconds
        }

    @staticmethod
    def _empty_metrics() -> Dict:
        """Returns empty metrics"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'roi': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'avg_duration': 0
        }


# ============================================================================
# DASHBOARD APP
# ============================================================================

class PaperDashboard:
    """Main dashboard class"""

    def __init__(self):
        self.config = DashboardConfig()
        self.loader = DataLoader(self.config)
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("📊 Paper Trading Dashboard",
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
                html.P("Real-time monitoring of paper trading bot",
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': 0}),
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),

            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.config.REFRESH_INTERVAL,
                n_intervals=0
            ),

            # Overview Section
            html.Div([
                html.H2("📈 Overview", style={'color': '#34495e'}),
                html.Div(id='overview-cards', children=[]),
            ], style={'marginBottom': '30px'}),

            # Equity Curve
            html.Div([
                html.H2("💰 Equity Curve", style={'color': '#34495e'}),
                dcc.Graph(id='equity-curve'),
            ], style={'marginBottom': '30px'}),

            # Two columns: Open Positions & Top-10
            html.Div([
                html.Div([
                    html.H2("🔓 Open Positions", style={'color': '#34495e'}),
                    html.Div(id='open-positions-table'),
                ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),

                html.Div([
                    html.H2("🏆 Top-10 Coins", style={'color': '#34495e'}),
                    html.Div(id='top-coins-table'),
                ], style={'width': '38%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            ], style={'marginBottom': '30px'}),

            # Trade History
            html.Div([
                html.H2("📋 Trade History (Last 50)", style={'color': '#34495e'}),
                html.Div(id='trade-history-table'),
            ], style={'marginBottom': '30px'}),

            # Performance Metrics
            html.Div([
                html.H2("📊 Performance Metrics", style={'color': '#34495e'}),
                html.Div(id='performance-metrics'),
            ], style={'marginBottom': '30px'}),

            # Footer
            html.Div([
                html.P(f"Auto-refresh: {self.config.REFRESH_INTERVAL/1000:.0f}s | "
                      f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                      id='footer-text',
                      style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px'})
            ])
        ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

    def setup_callbacks(self):
        """Setup Dash callbacks for real-time updates"""

        @self.app.callback(
            [Output('overview-cards', 'children'),
             Output('equity-curve', 'figure'),
             Output('open-positions-table', 'children'),
             Output('top-coins-table', 'children'),
             Output('trade-history-table', 'children'),
             Output('performance-metrics', 'children'),
             Output('footer-text', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Update all dashboard components"""

            # Load data
            state = self.loader.load_exchange_state()
            bot_state = self.loader.load_bot_state()
            trades_df = self.loader.load_trades()

            # 1. Overview Cards
            overview = self.create_overview_cards(state, trades_df)

            # 2. Equity Curve
            equity_fig = self.create_equity_curve(state, trades_df)

            # 3. Open Positions
            open_positions = self.create_open_positions_table(state)

            # 4. Top-10 Coins
            top_coins = self.create_top_coins_table(bot_state)

            # 5. Trade History
            trade_history = self.create_trade_history_table(trades_df)

            # 6. Performance Metrics
            metrics = self.create_performance_metrics(trades_df, state)

            # 7. Footer
            footer = f"Auto-refresh: {self.config.REFRESH_INTERVAL/1000:.0f}s | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            return overview, equity_fig, open_positions, top_coins, trade_history, metrics, footer

    def create_overview_cards(self, state: Optional[Dict], trades_df: pd.DataFrame) -> html.Div:
        """Creates overview cards"""
        if not state:
            return html.Div("No data available", style={'color': 'red'})

        # Calculate metrics
        initial_capital = state.get('initial_capital', 1000)
        current_equity = state.get('balance', initial_capital)

        # PnL
        total_pnl = current_equity - initial_capital
        pnl_percent = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0
        pnl_color = '#27ae60' if total_pnl >= 0 else '#e74c3c'

        # Closed trades
        closed_positions = state.get('closed_positions', [])
        total_trades = len(closed_positions)

        wins = sum(1 for p in closed_positions if p.get('pnl_after_commission', 0) > 0)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # Open positions
        open_positions = len(state.get('positions', {}))

        # Cards
        cards = [
            self._create_card("Current Capital", f"${current_equity:,.2f}",
                            f"{total_pnl:+,.2f} ({pnl_percent:+.2f}%)", pnl_color),
            self._create_card("Win Rate", f"{win_rate:.1f}%",
                            f"{wins}W / {total_trades-wins}L", '#3498db'),
            self._create_card("Open Positions", str(open_positions),
                            f"Total: {total_trades}", '#9b59b6'),
            self._create_card("Total Trades", str(total_trades),
                            f"Active: {open_positions}", '#e67e22'),
        ]

        return html.Div(cards, style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})

    def _create_card(self, title: str, value: str, subtitle: str, color: str) -> html.Div:
        """Creates a metric card"""
        return html.Div([
            html.H4(title, style={'margin': '0', 'color': '#7f8c8d', 'fontSize': '14px'}),
            html.H2(value, style={'margin': '10px 0', 'color': color, 'fontSize': '32px'}),
            html.P(subtitle, style={'margin': '0', 'color': '#95a5a6', 'fontSize': '12px'}),
        ], style={
            'backgroundColor': '#fff',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'minWidth': '200px',
            'margin': '10px'
        })

    def create_equity_curve(self, state: Optional[Dict], trades_df: pd.DataFrame) -> go.Figure:
        """Creates equity curve chart"""
        equity_history = self.loader.get_equity_history()

        if not equity_history:
            fig = go.Figure()
            fig.add_annotation(text="No data available",
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig

        df = pd.DataFrame(equity_history)

        # Equity line
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#3498db', width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.1)'
        ))

        # ATH marker
        ath_idx = df['equity'].idxmax()
        ath_value = df.loc[ath_idx, 'equity']
        fig.add_trace(go.Scatter(
            x=[df.loc[ath_idx, 'timestamp']],
            y=[ath_value],
            mode='markers+text',
            name='ATH',
            marker=dict(color='#f39c12', size=12, symbol='star'),
            text=[f'ATH: ${ath_value:,.2f}'],
            textposition='top center'
        ))

        fig.update_layout(
            title='Equity Over Time',
            xaxis_title='Time',
            yaxis_title='Equity (USDT)',
            hovermode='x unified',
            height=400
        )

        return fig

    def create_open_positions_table(self, state: Optional[Dict]) -> dash_table.DataTable:
        """Creates open positions table"""
        if not state or not state.get('positions'):
            return html.Div("No open positions", style={'color': '#95a5a6', 'fontStyle': 'italic'})

        positions = []
        for pos_id, pos in state['positions'].items():
            # Calculate unrealized PnL (simplified, in real app would fetch current price)
            entry_price = pos['entry_price']
            current_price = entry_price * 1.01  # Mock +1% for demo

            if pos['side'] == 'LONG':
                unrealized_pnl = (current_price - entry_price) * pos['amount']
            else:
                unrealized_pnl = (entry_price - current_price) * pos['amount']

            duration = datetime.now() - datetime.fromtimestamp(pos['opened_at'])

            positions.append({
                'Symbol': pos['symbol'],
                'Direction': pos['side'],
                'Entry': f"${entry_price:.2f}",
                'Current': f"${current_price:.2f}",
                'TP': f"${pos.get('tp_price', 0):.2f}",
                'SL': f"${pos.get('sl_price', 0):.2f}",
                'Unrealized PnL': f"${unrealized_pnl:+.2f}",
                'Duration': str(duration).split('.')[0],
                'EV': f"{pos.get('metadata', {}).get('entry_snapshot', {}).get('ev', 0):+.4f}"
            })

        df = pd.DataFrame(positions)

        return dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_cell={'textAlign': 'left', 'padding': '8px'},
            style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Direction} = "LONG"'},
                    'backgroundColor': '#d4edda',
                },
                {
                    'if': {'filter_query': '{Direction} = "SHORT"'},
                    'backgroundColor': '#f8d7da',
                }
            ]
        )

    def create_top_coins_table(self, bot_state: Optional[Dict]) -> html.Div:
        """Creates top-10 coins table"""
        if not bot_state or 'active_symbols' not in bot_state:
            return html.Div("No data", style={'color': '#95a5a6', 'fontStyle': 'italic'})

        coins = []
        for symbol in bot_state.get('active_symbols', [])[:10]:
            coins.append(html.Li(symbol, style={'padding': '5px 0'}))

        if not coins:
            return html.Div("No active symbols", style={'color': '#95a5a6', 'fontStyle': 'italic'})

        return html.Ul(coins, style={'listStyle': 'none', 'padding': '0'})

    def create_trade_history_table(self, trades_df: pd.DataFrame) -> dash_table.DataTable:
        """Creates trade history table"""
        if trades_df.empty:
            return html.Div("No trade history", style={'color': '#95a5a6', 'fontStyle': 'italic'})

        # Filter closed trades
        closed = trades_df[trades_df['action'] == 'CLOSE'].copy()
        if closed.empty:
            return html.Div("No closed trades", style={'color': '#95a5a6', 'fontStyle': 'italic'})

        # Last 50
        closed = closed.tail(50).sort_values('timestamp', ascending=False)

        # Format
        display_df = pd.DataFrame({
            'Symbol': closed['symbol'],
            'Direction': closed['direction'],
            'Entry': closed['entry_price'].apply(lambda x: f"${x:.2f}"),
            'Exit': closed['exit_price'].apply(lambda x: f"${x:.2f}"),
            'PnL ($)': closed['pnl'].apply(lambda x: f"${x:+.2f}"),
            'PnL (%)': closed['pnl_percent'].apply(lambda x: f"{x:+.2f}%"),
            'Reason': closed['reason'],
            'Outcome': closed['pnl'].apply(lambda x: '✅' if x > 0 else '❌')
        })

        return dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in display_df.columns],
            style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
            style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Outcome} = "✅"'},
                    'backgroundColor': '#d4edda',
                },
                {
                    'if': {'filter_query': '{Outcome} = "❌"'},
                    'backgroundColor': '#f8d7da',
                }
            ],
            page_size=10
        )

    def create_performance_metrics(self, trades_df: pd.DataFrame, state: Optional[Dict]) -> html.Div:
        """Creates performance metrics display"""
        metrics = MetricsCalculator.calculate_metrics(trades_df, state)

        metrics_display = [
            html.Div([
                html.Div([
                    html.H4("Win Rate", style={'margin': '0', 'fontSize': '14px', 'color': '#7f8c8d'}),
                    html.H3(f"{metrics['win_rate']:.2f}%", style={'margin': '5px 0', 'color': '#27ae60'}),
                ], style={'flex': '1', 'padding': '15px'}),

                html.Div([
                    html.H4("Profit Factor", style={'margin': '0', 'fontSize': '14px', 'color': '#7f8c8d'}),
                    html.H3(f"{metrics['profit_factor']:.2f}", style={'margin': '5px 0', 'color': '#3498db'}),
                ], style={'flex': '1', 'padding': '15px'}),

                html.Div([
                    html.H4("Sharpe Ratio", style={'margin': '0', 'fontSize': '14px', 'color': '#7f8c8d'}),
                    html.H3(f"{metrics['sharpe_ratio']:.2f}", style={'margin': '5px 0', 'color': '#9b59b6'}),
                ], style={'flex': '1', 'padding': '15px'}),

                html.Div([
                    html.H4("Max Drawdown", style={'margin': '0', 'fontSize': '14px', 'color': '#7f8c8d'}),
                    html.H3(f"{metrics['max_drawdown']:.2f}%", style={'margin': '5px 0', 'color': '#e74c3c'}),
                ], style={'flex': '1', 'padding': '15px'}),
            ], style={'display': 'flex', 'backgroundColor': '#fff', 'borderRadius': '8px',
                     'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '10px'}),

            html.Div([
                html.Div([
                    html.H4("Avg Win", style={'margin': '0', 'fontSize': '14px', 'color': '#7f8c8d'}),
                    html.H3(f"${metrics['avg_win']:.2f}", style={'margin': '5px 0', 'color': '#27ae60'}),
                ], style={'flex': '1', 'padding': '15px'}),

                html.Div([
                    html.H4("Avg Loss", style={'margin': '0', 'fontSize': '14px', 'color': '#7f8c8d'}),
                    html.H3(f"${metrics['avg_loss']:.2f}", style={'margin': '5px 0', 'color': '#e74c3c'}),
                ], style={'flex': '1', 'padding': '15px'}),

                html.Div([
                    html.H4("Total Trades", style={'margin': '0', 'fontSize': '14px', 'color': '#7f8c8d'}),
                    html.H3(f"{metrics['total_trades']}", style={'margin': '5px 0', 'color': '#34495e'}),
                ], style={'flex': '1', 'padding': '15px'}),

                html.Div([
                    html.H4("Sortino Ratio", style={'margin': '0', 'fontSize': '14px', 'color': '#7f8c8d'}),
                    html.H3(f"{metrics['sortino_ratio']:.2f}", style={'margin': '5px 0', 'color': '#e67e22'}),
                ], style={'flex': '1', 'padding': '15px'}),
            ], style={'display': 'flex', 'backgroundColor': '#fff', 'borderRadius': '8px',
                     'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ]

        return html.Div(metrics_display)

    def run(self):
        """Run the dashboard"""
        print("=" * 80)
        print("📊 PAPER TRADING DASHBOARD")
        print("=" * 80)
        print(f"Starting dashboard on http://{self.config.HOST}:{self.config.PORT}")
        print(f"Auto-refresh: {self.config.REFRESH_INTERVAL/1000:.0f} seconds")
        print("Press Ctrl+C to stop")
        print("=" * 80)

        self.app.run_server(
            host=self.config.HOST,
            port=self.config.PORT,
            debug=False
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Entry point"""
    dashboard = PaperDashboard()
    dashboard.run()


if __name__ == '__main__':
    main()
