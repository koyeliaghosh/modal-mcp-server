# 🚀 Modal MCP Stock Analysis Server

A cloud-hosted Model Context Protocol (MCP) server for real-time stock analysis, built with Modal and FastAPI.

## 📋 Overview

This MCP server provides stock analysis tools through a RESTful API, demonstrating how to implement the Model Context Protocol in a distributed, cloud-native architecture.

### 🏗️ Architecture
- **Backend**: Modal serverless platform
- **Framework**: FastAPI for HTTP endpoints
- **Protocol**: Model Context Protocol (MCP) implementation
- **Data Source**: yfinance for real-time stock market data

## ✨ Features

### 🔧 MCP Tools Available:
1. **`get_stock_price`** - Retrieve current stock price and basic company information
2. **`analyze_stock_comprehensive`** - Complete investment analysis with scoring algorithm
3. **`smart_ticker_search`** - Convert company names to ticker symbols

### 📊 Analysis Capabilities:
- Real-time stock price lookup
- YTD return calculations
- P/E ratio analysis
- Investment scoring (0-100 scale)
- Buy/Hold/Sell recommendations
- Smart company name → ticker conversion

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Modal account ([modal.com](https://modal.com))
- Modal CLI installed

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd modal-mcp-server
   ```

2. **Install Modal CLI**
   ```bash
   pip install modal
   ```

3. **Authenticate with Modal**
   ```bash
   modal token new
   ```

4. **Deploy the MCP server**
   ```bash
   modal deploy modal_mcp_complete.py
   ```

## 📡 API Endpoints

### Base URL
```
https://your-username--mcp-stock-analysis-server-web-app.modal.run
```

### Available Endpoints:

#### **GET /** - Server Information
```bash
curl https://your-modal-url/
```

#### **GET /health** - Health Check
```bash
curl https://your-modal-url/health
```

#### **GET /tools** - MCP Tools Discovery
```bash
curl https://your-modal-url/tools
```

#### **POST /call** - Execute MCP Tool
```bash
curl -X POST https://your-modal-url/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_stock_price",
    "arguments": {"symbol": "AAPL"}
  }'
```

## 🛠️ Usage Examples

### Stock Price Lookup
```bash
curl -X POST https://your-modal-url/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_stock_price",
    "arguments": {"symbol": "TSLA"}
  }'
```

**Response:**
```json
{
  "success": true,
  "tool": "get_stock_price",
  "result": [{
    "symbol": "TSLA",
    "company_name": "Tesla, Inc.",
    "current_price": 248.50,
    "market_cap": 790000000000,
    "sector": "Consumer Cyclical"
  }],
  "timestamp": "2025-06-10T..."
}
```

### Comprehensive Analysis
```bash
curl -X POST https://your-modal-url/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analyze_stock_comprehensive", 
    "arguments": {"symbol": "AAPL"}
  }'
```

**Response:**
```json
{
  "success": true,
  "tool": "analyze_stock_comprehensive",
  "result": [{
    "symbol": "AAPL",
    "company_name": "Apple Inc.",
    "current_price": 185.50,
    "ytd_return": 12.5,
    "pe_ratio": 28.5,
    "investment_score": 75,
    "recommendation": "Buy",
    "sector": "Technology"
  }],
  "timestamp": "2025-06-10T..."
}
```

### Smart Ticker Search
```bash
curl -X POST https://your-modal-url/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "smart_ticker_search",
    "arguments": {"query": "microsoft"}
  }'
```

## 🔧 Configuration

### Environment Variables
- `MCP_SERVER_SECRET` - Optional server secret for authentication

### Modal Secrets
The server can optionally use Modal secrets for configuration:
```python
stock_secrets = modal.Secret.from_name("Stock-api-config")
```

## 📁 Project Structure

```
modal-mcp-server/
├── modal_mcp_complete.py     # Main MCP server implementation
├── README.md                 # This file
└── requirements.txt          # Dependencies (handled by Modal)
```

## 🧠 Investment Scoring Algorithm

The comprehensive analysis uses a proprietary scoring algorithm:

```python
score = 50  # Base score
if ytd_return > 15: score += 25
elif ytd_return > 5: score += 15
elif ytd_return > 0: score += 5

if 10 < pe_ratio < 25: score += 15

# Final score: 0-100
recommendation = "Buy" if score >= 70 else "Hold" if score >= 50 else "Sell"
```

## 🎯 Supported Stock Symbols

### Direct Ticker Symbols:
- **Tech**: AAPL, GOOGL, MSFT, TSLA, META, NVDA, AMD, INTC
- **Finance**: JPM, BAC, WFC, GS
- **Consumer**: WMT, KO, PEP, NKE, MCD
- **ETFs**: SPY, QQQ

### Company Name Mapping:
- apple → AAPL
- microsoft → MSFT  
- google/alphabet → GOOGL
- amazon → AMZN
- tesla → TSLA
- meta/facebook → META
- netflix → NFLX
- nvidia → NVDA

## 🚀 Development

### Local Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test tools discovery
curl http://localhost:8000/tools

# Test tool execution
curl -X POST http://localhost:8000/call \
  -H "Content-Type: application/json" \
  -d '{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}'
```

### Deployment
```bash
# Deploy to Modal
modal deploy modal_mcp_complete.py

# Check deployment status
modal app list

# View logs
modal logs <app-id>
```

## 🔍 Monitoring & Debugging

### Health Checks
Monitor server health via the `/health` endpoint:
```bash
curl https://your-modal-url/health
```

### Error Handling
The server provides detailed error responses:
```json
{
  "error": "No data available for INVALID",
  "success": false,
  "timestamp": "2025-06-10T..."
}
```

## 📚 MCP Protocol Implementation

This server implements the Model Context Protocol specification:

- **Tool Discovery**: `GET /tools` returns available MCP tools
- **Tool Execution**: `POST /call` executes tools with proper request/response format
- **Error Handling**: Standardized error responses
- **Type Safety**: JSON schema validation for tool inputs

## 🤝 Integration Examples

### Python Client
```python
import requests

def call_mcp_tool(tool_name, arguments):
    response = requests.post(
        "https://your-modal-url/call",
        json={"name": tool_name, "arguments": arguments}
    )
    return response.json()

# Get stock price
result = call_mcp_tool("get_stock_price", {"symbol": "AAPL"})
```

### JavaScript Client
```javascript
async function callMCPTool(toolName, arguments) {
  const response = await fetch('https://your-modal-url/call', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: toolName, arguments })
  });
  return response.json();
}

// Analyze stock
const analysis = await callMCPTool('analyze_stock_comprehensive', { symbol: 'TSLA' });
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Modal** - For serverless cloud infrastructure
- **Hugging Face** - For hackathon organization
- **yfinance** - For stock market data
- **FastAPI** - For web framework
- **MCP Community** - For protocol specification

## 🔗 Related Links

- **Frontend Interface**: [Gradio MCP Client](https://huggingface.co/spaces/koyelia/mcp-stock-analysis-hackathon)
- **Modal Platform**: [modal.com](https://modal.com)
- **MCP Specification**: [Model Context Protocol](https://spec.modelcontextprotocol.io/)

---

**Built for the Hugging Face Gradio MCP Hackathon** 🚀
