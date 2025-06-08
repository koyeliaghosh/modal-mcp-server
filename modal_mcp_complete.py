#!/usr/bin/env python3
"""
Complete Modal MCP Server with Stock-api-config Secret - FIXED VERSION
Ready for deployment with your created secret
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Modal imports
import modal

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import mcp.types as types

# Data analysis imports
import yfinance as yf
import pandas as pd
import numpy as np

# Web server imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal App Definition with Your Secret
app = modal.App("mcp-stock-analysis-server")

# Your secret (note the exact name you created)
stock_secrets = modal.Secret.from_name("Stock-api-config")

# Define Modal image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "mcp>=1.0.0",
    "yfinance>=0.2.20", 
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "websockets>=12.0",
    "python-dateutil>=2.8.0",
    "aiohttp>=3.8.0",
    "python-dotenv>=1.0.0"
])

class CloudStockAnalyzer:
    """Stock analysis engine with your Modal secrets"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.ticker_cache = {}
        
        # Initialize with your secret values
        self.mcp_server_secret = os.getenv('MCP_SERVER_SECRET', 'default-secret')
        self.yahoo_api_key = os.getenv('YAHOO_FINANCE_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        
        logger.info(f"🔐 MCP Server initialized with secret: {bool(self.mcp_server_secret)}")
        logger.info(f"📊 API Keys configured: Yahoo={bool(self.yahoo_api_key)}, "
                   f"AV={bool(self.alpha_vantage_key)}, Finnhub={bool(self.finnhub_key)}")
    
    def find_ticker_symbol(self, user_input: str) -> str:
        """Smart ticker search with company name mapping"""
        user_input = user_input.strip()
        
        if user_input in self.ticker_cache:
            return self.ticker_cache[user_input]
        
        # Test if it's already a ticker
        if len(user_input) <= 5 and user_input.replace('.', '').isalpha():
            test_ticker = user_input.upper()
            try:
                stock = yf.Ticker(test_ticker)
                info = stock.info
                if info and info.get('symbol'):
                    self.ticker_cache[user_input] = test_ticker
                    return test_ticker
            except:
                pass
        
        # Company name mappings
        company_mappings = {
            # Tech Giants
            'apple': 'AAPL', 'apple inc': 'AAPL',
            'microsoft': 'MSFT', 'microsoft corp': 'MSFT', 'microsoft corporation': 'MSFT',
            'google': 'GOOGL', 'alphabet': 'GOOGL', 'alphabet inc': 'GOOGL',
            'amazon': 'AMZN', 'amazon.com': 'AMZN', 'amazon inc': 'AMZN',
            'meta': 'META', 'facebook': 'META', 'meta platforms': 'META',
            'tesla': 'TSLA', 'tesla inc': 'TSLA', 'tesla motors': 'TSLA',
            'netflix': 'NFLX', 'netflix inc': 'NFLX',
            'nvidia': 'NVDA', 'nvidia corp': 'NVDA', 'nvidia corporation': 'NVDA',
            
            # Financial
            'jpmorgan': 'JPM', 'jp morgan': 'JPM', 'jpmorgan chase': 'JPM',
            'bank of america': 'BAC', 'bofa': 'BAC',
            'wells fargo': 'WFC', 'wells fargo bank': 'WFC',
            'goldman sachs': 'GS', 'goldman': 'GS',
            'morgan stanley': 'MS',
            'berkshire hathaway': 'BRK.B', 'berkshire': 'BRK.B',
            
            # Healthcare
            'johnson & johnson': 'JNJ', 'j&j': 'JNJ', 'jnj': 'JNJ',
            'pfizer': 'PFE', 'pfizer inc': 'PFE',
            'moderna': 'MRNA', 'moderna inc': 'MRNA',
            'merck': 'MRK', 'merck & co': 'MRK',
            
            # Consumer
            'walmart': 'WMT', 'walmart inc': 'WMT',
            'target': 'TGT', 'target corp': 'TGT',
            'coca cola': 'KO', 'coca-cola': 'KO', 'coke': 'KO',
            'pepsi': 'PEP', 'pepsico': 'PEP',
            'procter & gamble': 'PG', 'p&g': 'PG',
            'nike': 'NKE', 'nike inc': 'NKE',
            'mcdonalds': 'MCD', "mcdonald's": 'MCD',
            'starbucks': 'SBUX', 'starbucks corp': 'SBUX',
            
            # Energy
            'exxon mobil': 'XOM', 'exxon': 'XOM',
            'chevron': 'CVX', 'chevron corp': 'CVX',
            'bp': 'BP', 'british petroleum': 'BP',
            
            # Industrial
            'boeing': 'BA', 'boeing company': 'BA',
            'general electric': 'GE', 'ge': 'GE',
            'caterpillar': 'CAT', 'cat': 'CAT',
            
            # Semiconductors
            'intel': 'INTC', 'intel corp': 'INTC',
            'amd': 'AMD', 'advanced micro devices': 'AMD',
            'qualcomm': 'QCOM', 'qualcomm inc': 'QCOM',
            
            # ETFs
            'spy': 'SPY', 's&p 500': 'SPY', 'sp500': 'SPY',
            'qqq': 'QQQ', 'nasdaq': 'QQQ', 'nasdaq 100': 'QQQ',
            'vti': 'VTI', 'total stock market': 'VTI',
        }
        
        normalized_input = user_input.lower().strip()
        
        # Direct lookup
        if normalized_input in company_mappings:
            ticker = company_mappings[normalized_input]
            self.ticker_cache[user_input] = ticker
            return ticker
        
        # Partial matching
        for company_name, ticker in company_mappings.items():
            if company_name in normalized_input or normalized_input in company_name:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    if info and info.get('symbol'):
                        self.ticker_cache[user_input] = ticker
                        return ticker
                except:
                    continue
        
        return user_input.upper()
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get stock data with caching"""
        cache_key = f"{symbol}_{period}"
        current_time = datetime.now()
        
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (current_time - timestamp).seconds < self.cache_timeout:
                logger.info(f"📋 Cache hit for {symbol}")
                return data
        
        try:
            logger.info(f"📡 Fetching fresh data for {symbol}")
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if not data.empty:
                self.cache[cache_key] = (data, current_time)
                logger.info(f"✅ Successfully cached data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        if data.empty:
            return {}
        
        try:
            # RSI calculation
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD calculation
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9).mean()
            
            # Moving averages
            ma20 = data['Close'].rolling(window=20).mean()
            ma50 = data['Close'].rolling(window=50).mean()
            
            return {
                'rsi': float(rsi.iloc[-1]) if not rsi.empty else 50,
                'macd': float(macd.iloc[-1]) if not macd.empty else 0,
                'macd_signal': float(macd_signal.iloc[-1]) if not macd_signal.empty else 0,
                'ma20': float(ma20.iloc[-1]) if not ma20.empty else 0,
                'ma50': float(ma50.iloc[-1]) if not ma50.empty else 0,
                'current_price': float(data['Close'].iloc[-1]) if not data['Close'].empty else 0
            }
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return {}
    
    def calculate_investment_score(self, symbol: str) -> Dict:
        """Calculate comprehensive investment score"""
        try:
            ticker = self.find_ticker_symbol(symbol)
            logger.info(f"🔍 Analyzing: '{symbol}' → {ticker}")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get YTD data (2025)
            ytd_start = datetime(2025, 1, 1)
            ytd_data = stock.history(start=ytd_start.strftime("%Y-%m-%d"))
            
            if ytd_data.empty:
                return {'error': f'No YTD 2025 data available for {ticker}'}
            
            # Calculate YTD return
            ytd_return = ((ytd_data['Close'].iloc[-1] - ytd_data['Close'].iloc[0]) / 
                         ytd_data['Close'].iloc[0]) * 100
            
            # Get 1-year data for volatility and technical analysis
            year_data = self.get_stock_data(ticker, "1y")
            volatility = 0
            max_drawdown = 0
            
            if year_data is not None and not year_data.empty:
                returns = year_data['Close'].pct_change().dropna()
                if not returns.empty:
                    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
                
                # Calculate max drawdown
                rolling_max = year_data['Close'].expanding().max()
                drawdown = (year_data['Close'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100
            
            # Technical indicators
            technical = self.calculate_technical_indicators(year_data) if year_data is not None else {}
            
            # Fundamental metrics with safe defaults
            pe_ratio = info.get('trailingPE') or 0
            forward_pe = info.get('forwardPE') or 0
            roe = info.get('returnOnEquity') or 0
            profit_margin = info.get('profitMargins') or 0
            revenue_growth = info.get('revenueGrowth') or 0
            
            # Calculate investment score (0-100)
            score = 50  # Base score
            
            # YTD Performance (30% weight)
            if ytd_return > 25:
                score += 25
            elif ytd_return > 15:
                score += 20
            elif ytd_return > 5:
                score += 15
            elif ytd_return > 0:
                score += 10
            elif ytd_return > -10:
                score += 5
            else:
                score -= 15
            
            # Technical indicators (25% weight)
            rsi = technical.get('rsi', 50)
            if 30 <= rsi <= 70:  # Sweet spot
                score += 12
            elif rsi < 30:  # Oversold
                score += 8
            elif rsi > 70:  # Overbought
                score -= 5
            
            # MACD signal
            macd = technical.get('macd', 0)
            macd_signal = technical.get('macd_signal', 0)
            if macd > macd_signal:  # Bullish
                score += 8
            else:
                score -= 3
            
            # Valuation (25% weight)
            if pe_ratio and 8 < pe_ratio < 20:
                score += 15
            elif pe_ratio and pe_ratio < 8:
                score += 20  # Very undervalued
            elif pe_ratio and 20 < pe_ratio < 30:
                score += 5
            elif pe_ratio and pe_ratio > 35:
                score -= 10
            
            # Growth and profitability (20% weight)
            if revenue_growth and revenue_growth > 0.20:
                score += 15
            elif revenue_growth and revenue_growth > 0.10:
                score += 10
            elif revenue_growth and revenue_growth > 0.05:
                score += 5
            
            if profit_margin and profit_margin > 0.15:
                score += 5
            elif profit_margin and profit_margin > 0.10:
                score += 3
            
            # Risk adjustment
            if volatility < 15:
                score += 5
            elif volatility > 35:
                score -= 10
            
            if max_drawdown > -15:
                score += 5
            elif max_drawdown < -30:
                score -= 8
            
            # Ensure score bounds
            score = max(0, min(100, score))
            
            # Determine risk level and recommendation
            if volatility < 15:
                risk_level = "Low"
            elif volatility < 25:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            if score >= 80:
                recommendation = "Strong Buy"
            elif score >= 70:
                recommendation = "Buy"
            elif score >= 60:
                recommendation = "Hold"
            elif score >= 50:
                recommendation = "Weak Hold"
            else:
                recommendation = "Sell"
            
            result = {
                'symbol': ticker.upper(),
                'search_input': symbol,
                'company_name': info.get('longName', 'N/A'),
                'current_price': float(ytd_data['Close'].iloc[-1]),
                'ytd_return': float(ytd_return),
                'volatility': float(volatility),
                'max_drawdown': float(max_drawdown),
                'pe_ratio': float(pe_ratio) if pe_ratio else 0,
                'forward_pe': float(forward_pe) if forward_pe else 0,
                'roe': float(roe * 100) if roe else 0,
                'profit_margin': float(profit_margin * 100) if profit_margin else 0,
                'revenue_growth': float(revenue_growth * 100) if revenue_growth else 0,
                'investment_score': int(score),
                'recommendation': recommendation,
                'risk_level': risk_level,
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0) or 0,
                'technical_indicators': technical,
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"✅ Analysis complete for {ticker}: Score {score}/100, Recommendation: {recommendation}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return {'error': f'Error analyzing {symbol}: {str(e)}'}

# Initialize analyzer (will be done in each function with secrets)
analyzer = None

# MCP Server Setup
mcp_server = Server("modal-stock-analysis-mcp")

@mcp_server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="get_stock_price",
            description="Get current stock price and basic information for any stock",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol or company name (e.g., AAPL, Apple, Tesla, Microsoft)"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="analyze_stock_comprehensive",
            description="Comprehensive AI-powered stock analysis with 100-point scoring system",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol or company name (supports smart search)"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="compare_stocks_ytd",
            description="Compare multiple stocks for YTD 2025 performance with AI rankings",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock symbols or company names to compare"
                    }
                },
                "required": ["symbols"]
            }
        ),
        Tool(
            name="smart_ticker_search",
            description="Smart search to find ticker symbols from company names or descriptions",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Company name, description, or search term"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_market_trends",
            description="Analyze market trends and sector performance",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock symbols to analyze for market trends"
                    }
                },
                "required": ["symbols"]
            }
        )
    ]

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle MCP tool calls with Modal secrets"""
    try:
        global analyzer
        if analyzer is None:
            analyzer = CloudStockAnalyzer()
        
        if name == "get_stock_price":
            symbol = arguments.get("symbol", "")
            if not symbol:
                return [TextContent(type="text", text='{"error": "Symbol is required"}')]
            
            ticker = analyzer.find_ticker_symbol(symbol)
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="2d")
            
            if hist.empty:
                return [TextContent(type="text", text=f'{{"error": "No data found for {symbol} → {ticker}"}}')]
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            result = {
                "tool": "get_stock_price",
                "search_input": symbol,
                "found_ticker": ticker,
                "company_name": info.get('longName', 'N/A'),
                "current_price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "previous_close": round(prev_close, 2),
                "market_cap": info.get('marketCap', 0),
                "volume": int(hist['Volume'].iloc[-1]),
                "sector": info.get('sector', 'N/A'),
                "timestamp": datetime.now().isoformat()
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "analyze_stock_comprehensive":
            symbol = arguments.get("symbol", "")
            if not symbol:
                return [TextContent(type="text", text='{"error": "Symbol is required"}')]
            
            analysis = analyzer.calculate_investment_score(symbol)
            analysis["tool"] = "analyze_stock_comprehensive"
            
            return [TextContent(type="text", text=json.dumps(analysis, indent=2))]
        
        elif name == "compare_stocks_ytd":
            symbols = arguments.get("symbols", [])
            if not symbols:
                return [TextContent(type="text", text='{"error": "Symbols list is required"}')]
            
            comparisons = []
            search_results = []
            
            for symbol in symbols:
                ticker = analyzer.find_ticker_symbol(symbol)
                search_results.append(f"{symbol} → {ticker}")
                
                analysis = analyzer.calculate_investment_score(symbol)
                if 'error' not in analysis:
                    comparisons.append(analysis)
            
            # Sort by investment score
            comparisons.sort(key=lambda x: x.get('investment_score', 0), reverse=True)
            
            result = {
                "tool": "compare_stocks_ytd",
                "search_inputs": symbols,
                "search_results": search_results,
                "comparison_results": comparisons,
                "winner": comparisons[0] if comparisons else None,
                "total_analyzed": len(comparisons),
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "smart_ticker_search":
            query = arguments.get("query", "")
            if not query:
                return [TextContent(type="text", text='{"error": "Query is required"}')]
            
            ticker = analyzer.find_ticker_symbol(query)
            
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                if info and info.get('symbol'):
                    result = {
                        "tool": "smart_ticker_search",
                        "search_query": query,
                        "found_ticker": ticker,
                        "company_name": info.get('longName', 'N/A'),
                        "sector": info.get('sector', 'N/A'),
                        "market_cap": info.get('marketCap', 0),
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    result = {
                        "tool": "smart_ticker_search",
                        "search_query": query,
                        "found_ticker": ticker,
                        "success": False,
                        "error": f"Could not validate ticker {ticker}",
                        "suggestions": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                result = {
                    "tool": "smart_ticker_search",
                    "search_query": query,
                    "found_ticker": ticker,
                    "success": False,
                    "error": f"Error validating ticker: {str(e)}",
                    "suggestions": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                    "timestamp": datetime.now().isoformat()
                }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_market_trends":
            symbols = arguments.get("symbols", [])
            if not symbols:
                return [TextContent(type="text", text='{"error": "Symbols list is required"}')]
            
            sector_data = {}
            trend_analysis = []
            
            for symbol in symbols:
                analysis = analyzer.calculate_investment_score(symbol)
                if 'error' not in analysis:
                    sector = analysis.get('sector', 'Unknown')
                    if sector not in sector_data:
                        sector_data[sector] = []
                    sector_data[sector].append(analysis)
                    trend_analysis.append(analysis)
            
            # Calculate sector trends
            sector_trends = {}
            for sector, stocks in sector_data.items():
                avg_score = sum(s['investment_score'] for s in stocks) / len(stocks)
                avg_ytd = sum(s['ytd_return'] for s in stocks) / len(stocks)
                sector_trends[sector] = {
                    "average_score": round(avg_score, 1),
                    "average_ytd_return": round(avg_ytd, 2),
                    "stock_count": len(stocks),
                    "trend": "Bullish" if avg_score > 70 else "Bearish" if avg_score < 50 else "Neutral"
                }
            
            result = {
                "tool": "get_market_trends",
                "analyzed_symbols": symbols,
                "sector_trends": sector_trends,
                "overall_market_score": round(sum(s['investment_score'] for s in trend_analysis) / len(trend_analysis), 1) if trend_analysis else 0,
                "total_stocks_analyzed": len(trend_analysis),
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        else:
            return [TextContent(type="text", text=f'{{"error": "Unknown tool: {name}"}}')]
    
    except Exception as e:
        error_msg = f"Error executing tool '{name}': {str(e)}"
        logger.error(error_msg)
        error_result = {
            "tool": name,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

# Modal Functions with Your Secret

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=3600,
    keep_warm=1,
    allow_concurrent_inputs=100,
    secrets=[stock_secrets]  # Your secret here!
)
@modal.web_endpoint(method="GET", label="health")
def modal_health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "platform": "Modal",
        "server": "mcp-stock-analysis",
        "mcp_secret_configured": bool(os.getenv('MCP_SERVER_SECRET')),
        "timestamp": datetime.now().isoformat()
    }

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=3600,
    keep_warm=1,
    allow_concurrent_inputs=100,
    secrets=[stock_secrets]
)
@modal.web_endpoint(method="GET", label="tools")
def modal_get_tools():
    """Get available MCP tools"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tools = loop.run_until_complete(handle_list_tools())
        loop.close()
        return {
            "tools": [tool.model_dump() for tool in tools],
            "total_tools": len(tools),
            "server": "modal-mcp-stock-analysis"
        }
    except Exception as e:
        logger.error(f"Error in modal_get_tools: {e}")
        return {"error": str(e)}, 500

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=3600,
    keep_warm=1,
    allow_concurrent_inputs=100,
    secrets=[stock_secrets]
)
@modal.web_endpoint(method="POST", label="call")
def modal_call_tool(request: dict):
    """Execute MCP tool calls"""
    try:
        tool_name = request.get("name")
        arguments = request.get("arguments", {})
        
        if not tool_name:
            return {"error": "Tool name is required"}, 400
        
        # Initialize analyzer with secrets
        global analyzer
        analyzer = CloudStockAnalyzer()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(handle_call_tool(tool_name, arguments))
        loop.close()
        
        return {
            "success": True,
            "tool": tool_name,
            "result": [content.text for content in result],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Error in modal_call_tool: {e}")
        return {"error": str(e)}, 500

@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=3600,
    secrets=[stock_secrets]
)
@modal.web_endpoint(method="POST", label="analyze")
def modal_analyze_stock(request: dict):
    """Quick stock analysis endpoint"""
    try:
        symbol = request.get("symbol", "")
        if not symbol:
            return {"error": "Symbol is required"}, 400
        
        # Initialize analyzer
        analyzer = CloudStockAnalyzer()
        
        # Perform analysis
        result = analyzer.calculate_investment_score(symbol)
        
        return {
            "success": True,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {"error": str(e)}, 500

@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=3600,
    secrets=[stock_secrets]
)
@modal.web_endpoint(method="POST", label="search")
def modal_search_ticker(request: dict):
    """Smart ticker search endpoint"""
    try:
        query = request.get("query", "")
        if not query:
            return {"error": "Query is required"}, 400
        
        analyzer = CloudStockAnalyzer()
        ticker = analyzer.find_ticker_symbol(query)
        
        # Validate the ticker
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info and info.get('symbol'):
                return {
                    "success": True,
                    "search_query": query,
                    "found_ticker": ticker,
                    "company_name": info.get('longName', 'N/A'),
                    "sector": info.get('sector', 'N/A'),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "search_query": query,
                    "found_ticker": ticker,
                    "error": "Could not validate ticker",
                    "suggestions": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
                }
        except Exception as e:
            return {
                "success": False,
                "search_query": query,
                "error": f"Validation error: {str(e)}",
                "suggestions": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            }
    
    except Exception as e:
        return {"error": str(e)}, 500

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=3600,
    keep_warm=1,
    secrets=[stock_secrets]
)
def run_fastapi_server():
    """Run the full FastAPI server with MCP support"""
    global analyzer
    analyzer = CloudStockAnalyzer()
    
    logger.info("🚀 Starting Modal FastAPI MCP Server")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")

@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=3600,
    secrets=[stock_secrets]
)
def test_secret():
    """Test function to verify your secret is working"""
    secret_value = os.getenv('MCP_SERVER_SECRET')
    return {
        "secret_found": bool(secret_value),
        "secret_value": secret_value,
        "all_env_vars": list(os.environ.keys()),
        "timestamp": datetime.now().isoformat()
    }

# Local development and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Test the secret locally
            print("🧪 Testing Modal secret...")
            with modal.enable_output():
                result = test_secret.remote()
                print(f"✅ Secret test result: {result}")
        
        elif sys.argv[1] == "stdio":
            # Run stdio MCP server locally
            async def run_stdio():
                global analyzer
                analyzer = CloudStockAnalyzer()
                
                async with stdio_server() as (read_stream, write_stream):
                    await mcp_server.run(
                        read_stream,
                        write_stream,
                        InitializationOptions(
                            server_name="modal-stock-analysis-mcp",
                            server_version="1.0.0",
                            capabilities=mcp_server.get_capabilities()
                        )
                    )
            
            asyncio.run(run_stdio())
        
        elif sys.argv[1] == "deploy":
            # Deploy to Modal
            print("🚀 Deploying to Modal...")
            import subprocess
            subprocess.run(["modal", "deploy", __file__])
    
    else:
        print("Usage:")
        print("  python modal_mcp.py test     # Test secret")
        print("  python modal_mcp.py stdio   # Run stdio MCP server")
        print("  python modal_mcp.py deploy  # Deploy to Modal")
        print("\nOr deploy with: modal deploy modal_mcp.py")
