#!/bin/bash
# Modal MCP Server Deployment Script

echo "🚀 Modal MCP Stock Analysis Server Deployment"
echo "=============================================="

# Check if Modal CLI is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
fi

# Check if user is authenticated
echo "🔐 Checking Modal authentication..."
if ! modal token show &> /dev/null; then
    echo "❌ Not authenticated with Modal. Please run:"
    echo "   modal setup"
    exit 1
fi

echo "✅ Modal CLI authenticated"

# Check if secret exists
echo "🔍 Checking for secret 'Stock-api-config'..."
if modal secret list | grep -q "Stock-api-config"; then
    echo "✅ Secret 'Stock-api-config' found"
else
    echo "❌ Secret 'Stock-api-config' not found!"
    echo "Please create the secret in Modal dashboard first:"
    echo "1. Go to https://modal.com/secrets"
    echo "2. Create secret named 'Stock-api-config'"
    echo "3. Add environment variables as needed"
    exit 1
fi

# Test the secret first
echo "🧪 Testing secret configuration..."
python3 -c "
import modal
app = modal.App()

@app.function(secrets=[modal.Secret.from_name('Stock-api-config')])
def test_secret():
    import os
    return {
        'secret_found': bool(os.getenv('MCP_SERVER_SECRET')),
        'keys_found': list(os.environ.keys())
    }

if __name__ == '__main__':
    with modal.enable_output():
        result = test_secret.remote()
        print(f'Secret test result: {result}')
"

echo "✅ Secret test completed"

# Deploy the MCP server
echo "🚀 Deploying MCP server to Modal..."
modal deploy modal_mcp_complete.py

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo ""
    echo "🌐 Your MCP server endpoints:"
    echo "Health: https://[your-username]--mcp-stock-analysis-server-health.modal.run"
    echo "Tools:  https://[your-username]--mcp-stock-analysis-server-tools.modal.run"
    echo "Call:   https://[your-username]--mcp-stock-analysis-server-call.modal.run"
    echo "Analyze: https://[your-username]--mcp-stock-analysis-server-analyze.modal.run"
    echo "Search: https://[your-username]--mcp-stock-analysis-server-search.modal.run"
    echo ""
    echo "🧪 Test your deployment:"
    echo "curl https://[your-username]--mcp-stock-analysis-server-health.modal.run"
    echo ""
    echo "📚 Example MCP tool call:"
    echo 'curl -X POST https://[your-username]--mcp-stock-analysis-server-call.modal.run \'
    echo '  -H "Content-Type: application/json" \'
    echo '  -d '"'"'{"name": "analyze_stock_comprehensive", "arguments": {"symbol": "Apple"}}'"'"
else
    echo "❌ Deployment failed!"
    exit 1
fi