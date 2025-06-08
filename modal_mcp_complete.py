#!/usr/bin/env python3
"""
Modal MCP Server - Fresh Start - NO ASYNC COMPLICATIONs
"""

import json
import logging
import os
from datetime import datetime

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
    "numpy>=1.24.0"
])

# Tool definitions
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
    }
]

class StockAnalyzer:
    """Simple stock analyzer"""
    
    def __init__(self):
        self.company_mappings = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 
            'amazon': 'AMZN', 'meta': 'META', 'tesla': 'TSLA',
            'netflix': 'NFLX', 'nvidia': 'NVDA'
        }
    
    def find_ticker(self, symbol):
        """Find ticker symbol"""
        symbol = symbol.strip().lower()
        return self.company_mappings.get(symbol, symbol.upper())
    
    def get_stock_data(self, symbol):
        """Get stock data"""
        import yfinance as yf
        
        ticker = self.find_ticker(symbol)
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        info = stock.info
        
        if hist.empty:
            return {"error": f"No data for {ticker}"}
        
        return {
            "symbol": ticker,
            "company_name": info.get('longName', 'N/A'),
            "current_price": round(float(hist['Close'].iloc[-1]), 2),
            "market_cap": info.get('marketCap', 0),
            "sector": info.get('sector', 'N/A')
        }
    
    def analyze_stock(self, symbol):
        """Analyze stock"""
        import yfinance as yf
        
        ticker = self.find_ticker(symbol)
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        if hist.empty:
            return {"error": f"No data for {ticker}"}
        
        current_price = float(hist['Close'].iloc[-1])
        year_ago_price = float(hist['Close'].iloc[0])
        ytd_return = ((current_price - year_ago_price) / year_ago_price) * 100
        
        score = 50
        if ytd_return > 15:
            score += 25
        elif ytd_return > 5:
            score += 15
        
        recommendation = "Buy" if score >= 70 else "Hold" if score >= 50 else "Sell"
        
        return {
            "symbol": ticker,
            "company_name": info.get('longName', 'N/A'),
            "current_price": current_price,
            "ytd_return": round(ytd_return, 2),
            "investment_score": score,
            "recommendation": recommendation,
            "sector": info.get('sector', 'N/A')
        }

# Modal Endpoints - NO ASYNC ANYWHERE

@app.function(image=image, cpu=1, memory=1024, timeout=60)
@modal.web_endpoint(method="GET", label="test")
def modal_test():
    """Test endpoint"""
    return {
        "message": "Working!",
        "timestamp": datetime.now().isoformat(),
        "status": "healthy"
    }

@app.function(image=image, cpu=1, memory=2048, timeout=300, secrets=[stock_secrets])
@modal.web_endpoint(method="GET", label="health")
def modal_health():
    """Health endpoint"""
    return {
        "status": "healthy",
        "platform": "Modal",
        "server": "mcp-stock-analysis",
        "timestamp": datetime.now().isoformat()
    }

@app.function(image=image, cpu=1, memory=2048, timeout=300, secrets=[stock_secrets])
@modal.web_endpoint(method="GET", label="tools")
def modal_get_tools():
    """Tools endpoint - NO ASYNC"""
    return {
        "success": True,
        "tools": TOOLS,
        "total_tools": len(TOOLS),
        "server": "modal-mcp-stock-analysis",
        "timestamp": datetime.now().isoformat()
    }

@app.function(image=image, cpu=2, memory=4096, timeout=300, secrets=[stock_secrets])
@modal.web_endpoint(method="POST", label="call")
def modal_call_tool():
    """Call tool endpoint - NO ASYNC"""
    def handler(request: dict):
        try:
            tool_name = request.get("name")
            arguments = request.get("arguments", {})
            
            if not tool_name:
                return {"error": "Tool name required"}, 400
            
            analyzer = StockAnalyzer()
            
            if tool_name == "get_stock_price":
                symbol = arguments.get("symbol", "")
                if not symbol:
                    return {"error": "Symbol required"}, 400
                
                result = analyzer.get_stock_data(symbol)
                result["tool"] = "get_stock_price"
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": [json.dumps(result, indent=2)],
                    "timestamp": datetime.now().isoformat()
                }
                
            elif tool_name == "analyze_stock_comprehensive":
                symbol = arguments.get("symbol", "")
                if not symbol:
                    return {"error": "Symbol required"}, 400
                
                result = analyzer.analyze_stock(symbol)
                result["tool"] = "analyze_stock_comprehensive"
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": [json.dumps(result, indent=2)],
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                return {"error": f"Unknown tool: {tool_name}"}, 400
                
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"error": str(e)}, 500
    
    return handler

if __name__ == "__main__":
    print("Deploy with: python -m modal deploy modal_mcp_complete.py")
