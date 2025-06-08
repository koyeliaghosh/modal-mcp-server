import modal
from fastapi import FastAPI
from datetime import datetime
import json

app = modal.App("mcp-stock-analysis-server")

# Image with FastAPI dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "yfinance>=0.2.20", 
    "pandas>=2.0.0", 
    "numpy>=1.24.0"
])

# Create FastAPI app
fastapi_app = FastAPI(title="MCP Stock Analysis Server", version="1.0.0")

# Tool definitions - ADD THE MISSING TOOL
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
    def __init__(self):
        self.company_mappings = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 
            'amazon': 'AMZN', 'meta': 'META', 'tesla': 'TSLA',
            'netflix': 'NFLX', 'nvidia': 'NVDA', 'amd': 'AMD'
        }
    
    def find_ticker(self, symbol):
        symbol = symbol.strip().lower()
        return self.company_mappings.get(symbol, symbol.upper())
    
    def get_stock_data(self, symbol):
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
            "sector": info.get('sector', 'N/A'),
            "success": True
        }
    
    # ADD THIS METHOD
    def analyze_stock_comprehensive(self, symbol):
        """Comprehensive stock analysis with investment scoring"""
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
        
        # Investment scoring algorithm
        score = 50
        if ytd_return > 15:
            score += 25
        elif ytd_return > 5:
            score += 15
        elif ytd_return > 0:
            score += 5
        
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
            "symbol": ticker,
            "company_name": info.get('longName', 'N/A'),
            "current_price": current_price,
            "ytd_return": round(ytd_return, 2),
            "pe_ratio": pe_ratio or 0,
            "investment_score": score,
            "recommendation": recommendation,
            "sector": info.get('sector', 'N/A'),
            "market_cap": info.get('marketCap', 0),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": True
        }

# FastAPI Routes
@fastapi_app.get("/")
def root():
    return {
        "message": "MCP Stock Analysis Server",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@fastapi_app.get("/health")
def health():
    return {
        "status": "healthy",
        "platform": "Modal + FastAPI",
        "server": "mcp-stock-analysis",
        "timestamp": datetime.now().isoformat()
    }

@fastapi_app.get("/tools")
def get_tools():
    return {
        "success": True,
        "tools": TOOLS,
        "total_tools": len(TOOLS),
        "timestamp": datetime.now().isoformat()
    }

@fastapi_app.post("/call")
def call_tool(request: dict):
    try:
        tool_name = request.get("name")
        arguments = request.get("arguments", {})
        
        if not tool_name:
            return {"error": "Tool name required", "success": False}
        
        analyzer = StockAnalyzer()
        
        if tool_name == "get_stock_price":
            symbol = arguments.get("symbol", "")
            if not symbol:
                return {"error": "Symbol required", "success": False}
            
            result = analyzer.get_stock_data(symbol)
            result["tool"] = "get_stock_price"
            
            return {
                "success": True,
                "tool": tool_name,
                "result": [json.dumps(result, indent=2)],
                "timestamp": datetime.now().isoformat()
            }
            
        # ADD THIS ELIF BLOCK
        elif tool_name == "analyze_stock_comprehensive":
            symbol = arguments.get("symbol", "")
            if not symbol:
                return {"error": "Symbol required", "success": False}
            
            result = analyzer.analyze_stock_comprehensive(symbol)
            result["tool"] = "analyze_stock_comprehensive"
            
            return {
                "success": True,
                "tool": tool_name,
                "result": [json.dumps(result, indent=2)],
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            return {"error": f"Unknown tool: {tool_name}", "success": False}
            
    except Exception as e:
        return {"error": str(e), "success": False}

# Deploy FastAPI app using Modal's ASGI support
@app.function(image=image, cpu=2, memory=4096)
@modal.asgi_app()
def web_app():
    return fastapi_app