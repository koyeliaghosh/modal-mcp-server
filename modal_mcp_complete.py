import modal
from fastapi import FastAPI
from datetime import datetime
import json

app = modal.App("mcp-stock-analysis-server")

# Simple image with required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "yfinance>=0.2.20", 
    "pandas>=2.0.0", 
    "numpy>=1.24.0"
])


# Create FastAPI app
fastapi_app = FastAPI(title="MCP Stock Analysis Server", version="1.0.0")

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
    }
]

class StockAnalyzer:
    def __init__(self):
        self.company_mappings = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 
            'amazon': 'AMZN', 'meta': 'META', 'tesla': 'TSLA'
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
        
        if tool_name == "get_stock_price":
            symbol = arguments.get("symbol", "")
            if not symbol:
                return {"error": "Symbol required", "success": False}
            
            analyzer = StockAnalyzer()
            result = analyzer.get_stock_data(symbol)
            
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
