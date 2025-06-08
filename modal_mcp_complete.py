#!/usr/bin/env python3
"""
Complete Modal MCP Server - FINAL CLEAN VERSION
Removes all async complications that were causing errors
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any

import modal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal App Definition
app = modal.App("mcp-stock-analysis-server")

# Your secret
stock_secrets = modal.Secret.from_name("Stock-api-config")

# Define Modal image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
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
        logger.info(f"🔐 Stock Analyzer initialized")
    
    def find_ticker_symbol(self, user_input: str) -> str:
        """Smart ticker search"""
        user_input = user_input.strip()
        
        if user_input in self.ticker_cache:
            return self.ticker_cache[user_input]
        
        # Test if already a ticker
        if len(user_input) <= 5 and user_input.replace('.', '').isalpha():
            test_ticker = user_input.upper()
            try:
                import yfinance as yf
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
            import yfinance as yf
            
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

# Tool definitions (no MCP server complications)
TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Get current stock price and basic information",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol or company name"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "analyze_stock_comprehensive",
        "description": "Comprehensive stock analysis with scoring",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol or company name"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "smart_ticker_search",
        "description": "Smart search for ticker symbols",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Company name or search term"}
            },
            "required": ["query"]
        }
    }
]

# Modal Endpoints

@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=60
)
@modal.web_endpoint(method="GET", label="test")
def modal_test():
    """Simple test endpoint"""
    return {
        "message": "Modal MCP Server is working!",
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "version": "clean-no-async"
    }

@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=300,
    secrets=[stock_secrets]
)
@modal.web_endpoint(method="GET", label="health")
def modal_health():
    """Health check endpoint"""
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        
        return {
            "status": "healthy",
            "platform": "Modal",
            "server": "mcp-stock-analysis-clean",
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "yfinance": "available",
                "pandas": "available",
                "numpy": "available"
            },
            "secret_configured": bool(os.getenv('MCP_SERVER_SECRET'))
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"error": str(e), "status": "unhealthy"}, 500

@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=300,
    secrets=[stock_secrets]
)
@modal.web_endpoint(method="GET", label="tools")
def modal_get_tools():
    """Get available tools"""
    try:
        return {
            "success": True,
            "tools": TOOLS,
            "total_tools": len(TOOLS),
            "server": "modal-mcp-stock-analysis-clean",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in tools endpoint: {e}")
        return {"error": str(e), "success": False}, 500

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=300,
    secrets=[stock_secrets]
)
@modal.web_endpoint(method="POST", label="call")
def modal_call_tool():
    """Execute a tool"""
    def handler(request: dict):
        try:
            import yfinance as yf
            
            tool_name = request.get("name")
            arguments = request.get("arguments", {})
            
            if not tool_name:
                return {"error": "Tool name required", "success": False}, 400
            
            analyzer = CloudStockAnalyzer()
            
            if tool_name == "get_stock_price":
                symbol = arguments.get("symbol", "")
                if not symbol:
                    return {"error": "Symbol required", "success": False}, 400
                
                ticker = analyzer.find_ticker_symbol(symbol)
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                info = stock.info
                
                if hist.empty:
                    return {"error": f"No data for {ticker}", "success": False}, 400
                
                result = {
                    "tool": "get_stock_price",
                    "symbol": ticker,
                    "company_name": info.get('longName', 'N/A'),
                    "current_price": round(float(hist['Close'].iloc[-1]), 2),
                    "market_cap": info.get('marketCap', 0),
                    "sector": info.get('sector', 'N/A'),
                    "success": True
                }
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": [json.dumps(result, indent=2)],
                    "timestamp": datetime.now().isoformat()
                }
                
            elif tool_name == "analyze_stock_comprehensive":
                symbol = arguments.get("symbol", "")
                if not symbol:
                    return {"error": "Symbol required", "success": False}, 400
                
                analysis = analyzer.calculate_investment_score(symbol)
                analysis["tool"] = "analyze_stock_comprehensive"
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": [json.dumps(analysis, indent=2)],
                    "timestamp": datetime.now().isoformat()
                }
                
            elif tool_name == "smart_ticker_search":
                query = arguments.get("query", "")
                if not query:
                    return {"error": "Query required", "success": False}, 400
                
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
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": [json.dumps(result, indent=2)],
                    "timestamp": datetime.now().isoformat()
                }
                
            else:
                return {"error": f"Unknown tool: {tool_name}", "success": False}, 400
                
        except Exception as e:
            logger.error(f"Error in call endpoint: {e}")
            return {"error": str(e), "success": False}, 500
    
    return handler

@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=300,
    secrets=[stock_secrets]
)
@modal.web_endpoint(method="POST", label="analyze")
def modal_analyze_stock():
    """Quick stock analysis"""
    def handler(request: dict):
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
    
    return handler

@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=300,
    secrets=[stock_secrets]
)
@modal.web_endpoint(method="POST", label="search")
def modal_search_ticker():
    """Smart ticker search"""
    def handler(request: dict):
        try:
            query = request.get("query", "")
            if not query:
                return {"error": "Query required"}, 400
            
            analyzer = CloudStockAnalyzer()
            ticker = analyzer.find_ticker_symbol(query)
            
            try:
                import yfinance as yf
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
    
    return handler

# FastAPI Application (separate from Modal endpoints)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

fastapi_app = FastAPI(
    title="Modal MCP Stock Analysis Server",
    description="Clean MCP server with stock analysis capabilities",
    version="2.0.0"
)

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
        "server": "fastapi-mcp-stock-analysis-clean",
        "timestamp": datetime.now().isoformat()
    }

@fastapi_app.get("/tools")
async def fastapi_get_tools():
    """Get tools via FastAPI"""
    return {"tools": TOOLS}

@fastapi_app.post("/call")
async def fastapi_call_tool(request: dict):
    """Execute tool via FastAPI"""
    try:
        # This would use the same logic as the Modal call endpoint
        # but runs in FastAPI instead
        return {"message": "FastAPI endpoint - implement tool logic here"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=3600,
    secrets=[stock_secrets]
)
def run_fastapi_server():
    """Run FastAPI server"""
    logger.info("🚀 Starting FastAPI MCP Server")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")

# Local development
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "fastapi":
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
    else:
        print("Deploy with: python -m modal deploy modal_mcp_complete.py")