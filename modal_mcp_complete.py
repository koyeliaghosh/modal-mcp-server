#!/usr/bin/env python3
"""
Complete Modal MCP Server with FastAPI - FIXED VERSION
Includes all components: MCP Server + FastAPI + Modal endpoints
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal App Definition
app = modal.App("mcp-stock-analysis-server")

# Your secret
stock_secrets = modal.Secret.from_name("Stock-api-config")

# Define Modal image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "mcp>=1.0.0",
    "yfinance>=0.2.20", 
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "python-dateutil>=2.8.0"
])

class CloudStockAnalyzer:
    """Stock analysis engine"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300
        self.ticker_cache = {}
        
        self.mcp_server_secret = os.getenv('MCP_SERVER_SECRET', 'default-secret')
        logger.info(f"🔐 MCP Server initialized")
    
    def find_ticker_symbol(self, user_input: str) -> str:
        """Smart ticker search"""
        user_input = user_input.strip()
        
        if user_input in self.ticker_cache:
            return self.ticker_cache[user_input]
        
        # Test if already a ticker
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
        
        # Company mappings
        company_mappings = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
            'amazon': 'AMZN', 'meta': 'META', 'facebook': 'META', 'tesla': 'TSLA',
            'netflix': 'NFLX', 'nvidia': 'NVDA', 'jpmorgan': 'JPM', 'jp morgan': 'JPM',
            'bank of america': 'BAC', 'wells fargo': 'WFC', 'goldman sachs': 'GS',
            'johnson & johnson': 'JNJ', 'j&j': 'JNJ', 'pfizer': 'PFE', 'walmart': 'WMT',
            'coca cola': 'KO', 'pepsi': 'PEP', 'nike': 'NKE', 'mcdonalds': 'MCD',
            'boeing': 'BA', 'intel': 'INTC', 'amd': 'AMD', 'spy': 'SPY', 'qqq': 'QQQ'
        }
        
        normalized_input = user_input.lower().strip()
        
        if normalized_input in company_mappings:
            ticker = company_mappings[normalized_input]
            self.ticker_cache[user_input] = ticker
            return ticker
        
        return user_input.upper()
    
    def calculate_investment_score(self, symbol: str) -> Dict:
        """Calculate investment score"""
        try:
            ticker = self.find_ticker_symbol(symbol)
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get YTD data
            ytd_start = datetime(2025, 1, 1)
            ytd_data = stock.history(start=ytd_start.strftime("%Y-%m-%d"))
            
            if ytd_data.empty:
                return {'error': f'No data available for {ticker}'}
            
            # Calculate YTD return
            ytd_return = ((ytd_data['Close'].iloc[-1] - ytd_data['Close'].iloc[0]) / 
                         ytd_data['Close'].iloc[0]) * 100
            
            # Basic scoring
            score = 50
            if ytd_return > 15:
                score += 20
            elif ytd_return > 5:
                score += 15
            elif ytd_return > 0:
                score += 10
            
            # PE ratio scoring
            pe_ratio = info.get('trailingPE', 0)
            if pe_ratio and 10 < pe_ratio < 25:
                score += 15
            
            score = max(0, min(100, score))
            
            if score >= 70:
                recommendation = "Buy"
            elif score >= 50:
                recommendation = "Hold"
            else:
                recommendation = "Sell"
            
            return {
                'symbol': ticker.upper(),
                'search_input': symbol,
                'company_name': info.get('longName', 'N/A'),
                'current_price': float(ytd_data['Close'].iloc[-1]),
                'ytd_return': float(ytd_return),
                'pe_ratio': float(pe_ratio) if pe_ratio else 0,
                'investment_score': int(score),
                'recommendation': recommendation,
                'sector': info.get('sector', 'N/A'),
                'market_cap': info.get('marketCap', 0) or 0,
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {'error': f'Error analyzing {symbol}: {str(e)}'}

# Initialize analyzer
analyzer = None

# MCP Server Setup
mcp_server = Server("modal-stock-analysis-mcp")

@mcp_server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List MCP tools"""
    return [
        Tool(
            name="get_stock_price",
            description="Get current stock price and basic information",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol or company name"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="analyze_stock_comprehensive",
            description="Comprehensive stock analysis with scoring",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol or company name"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="smart_ticker_search",
            description="Smart search for ticker symbols",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Company name or search term"}
                },
                "required": ["query"]
            }
        )
    ]

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle MCP tool calls"""
    try:
        global analyzer
        if analyzer is None:
            analyzer = CloudStockAnalyzer()
        
        if name == "get_stock_price":
            symbol = arguments.get("symbol", "")
            if not symbol:
                return [TextContent(type="text", text='{"error": "Symbol required"}')]
            
            ticker = analyzer.find_ticker_symbol(symbol)
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            info = stock.info
            
            if hist.empty:
                return [TextContent(type="text", text=f'{{"error": "No data for {ticker}"}}')]
            
            result = {
                "tool": "get_stock_price",
                "symbol": ticker,
                "company_name": info.get('longName', 'N/A'),
                "current_price": round(float(hist['Close'].iloc[-1]), 2),
                "market_cap": info.get('marketCap', 0),
                "sector": info.get('sector', 'N/A')
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "analyze_stock_comprehensive":
            symbol = arguments.get("symbol", "")
            if not symbol:
                return [TextContent(type="text", text='{"error": "Symbol required"}')]
            
            analysis = analyzer.calculate_investment_score(symbol)
            analysis["tool"] = "analyze_stock_comprehensive"
            
            return [TextContent(type="text", text=json.dumps(analysis, indent=2))]
        
        elif name == "smart_ticker_search":
            query = arguments.get("query", "")
            if not query:
                return [TextContent(type="text", text='{"error": "Query required"}')]
            
            ticker = analyzer.find_ticker_symbol(query)
            
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                result = {
                    "tool": "smart_ticker_search",
                    "search_query": query,
                    "found_ticker": ticker,
                    "company_name": info.get('longName', 'N/A'),
                    "sector": info.get('sector', 'N/A'),
                    "success": bool(info.get('symbol'))
                }
            except:
                result = {
                    "tool": "smart_ticker_search",
                    "search_query": query,
                    "found_ticker": ticker,
                    "success": False,
                    "error": "Could not validate ticker"
                }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        else:
            return [TextContent(type="text", text=f'{{"error": "Unknown tool: {name}"}}')]
    
    except Exception as e:
        error_result = {
            "tool": name,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

# FastAPI Application
fastapi_app = FastAPI(
    title="Modal MCP Stock Analysis Server",
    description="Real MCP server with stock analysis capabilities",
    version="1.0.0"
)

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.get("/health")
async def fastapi_health():
    """FastAPI health check"""
    return {
        "status": "healthy",
        "server": "fastapi-mcp-stock-analysis",
        "timestamp": datetime.now().isoformat()
    }

@fastapi_app.get("/mcp/tools")
async def fastapi_get_tools():
    """Get MCP tools via FastAPI"""
    tools = await handle_list_tools()
    return {"tools": [tool.model_dump() for tool in tools]}

@fastapi_app.post("/mcp/call")
async def fastapi_call_tool(request: dict):
    """Execute MCP tool via FastAPI"""
    try:
        tool_name = request.get("name")
        arguments = request.get("arguments", {})
        
        if not tool_name:
            raise HTTPException(status_code=400, detail="Tool name required")
        
        result = await handle_call_tool(tool_name, arguments)
        
        return {
            "success": True,
            "result": [content.text for content in result],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility function to run async code in Modal functions
def run_async_in_thread(coro):
    """Helper to run async code in Modal functions"""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(coro)

# Modal Endpoints

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=3600,
    keep_warm=1,
    allow_concurrent_inputs=100,
    secrets=[stock_secrets]
)
@modal.web_endpoint(method="GET", label="health")
def modal_health():
    """Modal health endpoint"""
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
    """Get MCP tools via Modal"""
    try:
        tools = run_async_in_thread(handle_list_tools())
        
        return {
            "tools": [tool.model_dump() for tool in tools],
            "total_tools": len(tools),
            "server": "modal-mcp-stock-analysis",
            "timestamp": datetime.now().isoformat()
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
    """Execute MCP tool via Modal"""
    try:
        tool_name = request.get("name")
        arguments = request.get("arguments", {})
        
        if not tool_name:
            return {"error": "Tool name required"}, 400
        
        global analyzer
        analyzer = CloudStockAnalyzer()
        
        result = run_async_in_thread(handle_call_tool(tool_name, arguments))
        
        return {
            "success": True,
            "tool": tool_name,
            "result": [content.text for content in result],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in modal_call_tool: {e}")
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
    """Quick stock analysis via Modal"""
    try:
        symbol = request.get("symbol", "")
        if not symbol:
            return {"error": "Symbol required"}, 400
        
        analyzer = CloudStockAnalyzer()
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
    """Smart ticker search via Modal"""
    try:
        query = request.get("query", "")
        if not query:
            return {"error": "Query required"}, 400
        
        analyzer = CloudStockAnalyzer()
        ticker = analyzer.find_ticker_symbol(query)
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "success": True,
                "search_query": query,
                "found_ticker": ticker,
                "company_name": info.get('longName', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "timestamp": datetime.now().isoformat()
            }
        except:
            return {
                "success": False,
                "search_query": query,
                "error": "Could not validate ticker",
                "suggestions": ["AAPL", "MSFT", "GOOGL"]
            }
    
    except Exception as e:
        return {"error": str(e)}, 500

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=3600,
    secrets=[stock_secrets]
)
def run_fastapi_server():
    """Run FastAPI server"""
    global analyzer
    analyzer = CloudStockAnalyzer()
    
    logger.info("🚀 Starting FastAPI MCP Server")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")

# Local development
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "fastapi":
        # Run FastAPI locally
        analyzer = CloudStockAnalyzer()
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
    else:
        print("Deploy with: modal deploy modal_mcp_complete.py")