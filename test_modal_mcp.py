#!/usr/bin/env python3
"""
Test Modal MCP Server - Python Script
Tests all endpoints and MCP tools for the deployed Modal server
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

class ModalMCPTester:
    def __init__(self, base_url: str):
        """Initialize with your Modal deployment URL"""
        self.base_url = base_url.rstrip('/')
        self.endpoints = {
            'health': f'{self.base_url}-health.modal.run',
            'tools': f'{self.base_url}-tools.modal.run',
            'call': f'{self.base_url}-call.modal.run',
            'analyze': f'{self.base_url}-analyze.modal.run',
            'search': f'{self.base_url}-search.modal.run'
        }
        
        # Set request timeout
        self.timeout = 60
        
        print(f"🔗 Initialized tester for: {self.base_url}")
        print("📡 Endpoints:")
        for name, url in self.endpoints.items():
            print(f"  • {name}: {url}")
        print()
    
    def make_request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """Make HTTP request with error handling"""
        try:
            if method.upper() == 'GET':
                response = requests.get(url, timeout=self.timeout, **kwargs)
            elif method.upper() == 'POST':
                response = requests.post(url, timeout=self.timeout, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"    ❌ HTTP {response.status_code}: {response.text[:100]}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"    ⏰ Request timed out after {self.timeout} seconds")
            return None
        except requests.exceptions.ConnectionError:
            print(f"    🔌 Connection error - check if server is running")
            return None
        except Exception as e:
            print(f"    ❌ Request error: {str(e)}")
            return None
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        print("🏥 Testing health endpoint...")
        
        data = self.make_request('GET', self.endpoints['health'])
        if data:
            print(f"  ✅ Health check passed")
            print(f"    • Status: {data.get('status')}")
            print(f"    • Platform: {data.get('platform')}")
            print(f"    • MCP Secret: {data.get('mcp_secret_configured', 'Unknown')}")
            return True
        else:
            print(f"  ❌ Health check failed")
            return False
    
    def test_tools(self) -> bool:
        """Test tools listing"""
        print("\n🛠️ Testing tools endpoint...")
        
        data = self.make_request('GET', self.endpoints['tools'])
        if data:
            tools = data.get('tools', [])
            print(f"  ✅ Found {len(tools)} MCP tools:")
            for i, tool in enumerate(tools, 1):
                name = tool.get('name', 'Unknown')
                desc = tool.get('description', 'No description')[:60]
                print(f"    {i}. {name}: {desc}...")
            return True
        else:
            print(f"  ❌ Tools endpoint failed")
            return False
    
    def test_smart_search(self) -> bool:
        """Test smart ticker search"""
        print("\n🔍 Testing smart search endpoint...")
        
        test_queries = [
            "Apple",
            "Tesla", 
            "Microsoft",
            "Google",
            "AAPL",
            "Invalid Company XYZ"
        ]
        
        successful_searches = 0
        
        for query in test_queries:
            print(f"  Searching: '{query}'")
            
            data = self.make_request('POST', self.endpoints['search'], 
                                   json={"query": query})
            
            if data:
                if data.get('success'):
                    ticker = data.get('found_ticker', 'Unknown')
                    company = data.get('company_name', 'Unknown')
                    print(f"    ✅ '{query}' → {ticker} ({company})")
                    successful_searches += 1
                else:
                    error = data.get('error', 'Unknown error')
                    suggestions = data.get('suggestions', [])
                    print(f"    ⚠️ '{query}' → {error}")
                    if suggestions:
                        print(f"       Suggestions: {', '.join(suggestions[:3])}")
            else:
                print(f"    ❌ Search failed for '{query}'")
        
        print(f"  📊 Search Results: {successful_searches}/{len(test_queries)} successful")
        return successful_searches > 0
    
    def test_stock_analysis(self) -> bool:
        """Test comprehensive stock analysis"""
        print("\n📊 Testing stock analysis endpoint...")
        
        test_stocks = ["Apple", "TSLA", "Microsoft", "GOOGL"]
        successful_analyses = 0
        
        for stock in test_stocks:
            print(f"  Analyzing: '{stock}'")
            
            data = self.make_request('POST', self.endpoints['analyze'],
                                   json={"symbol": stock})
            
            if data and data.get('success'):
                analysis = data.get('analysis', {})
                symbol = analysis.get('symbol', 'Unknown')
                score = analysis.get('investment_score', 0)
                ytd_return = analysis.get('ytd_return', 0)
                recommendation = analysis.get('recommendation', 'Unknown')
                
                print(f"    ✅ {symbol} - Score: {score}/100, YTD: {ytd_return:+.2f}%, Rec: {recommendation}")
                successful_analyses += 1
            else:
                error = data.get('error') if data else 'Request failed'
                print(f"    ❌ Analysis failed: {error}")
        
        print(f"  📊 Analysis Results: {successful_analyses}/{len(test_stocks)} successful")
        return successful_analyses > 0
    
    def test_mcp_tools(self) -> bool:
        """Test MCP tool calls through the call endpoint"""
        print("\n⚡ Testing MCP tool calls...")
        
        successful_calls = 0
        total_calls = 0
        
        # Test 1: Smart ticker search via MCP
        print("  1. Testing MCP smart_ticker_search...")
        total_calls += 1
        
        data = self.make_request('POST', self.endpoints['call'],
                               json={
                                   "name": "smart_ticker_search",
                                   "arguments": {"query": "Apple"}
                               })
        
        if data and data.get('success'):
            try:
                result = json.loads(data['result'][0])
                if result.get('success'):
                    ticker = result.get('found_ticker')
                    company = result.get('company_name')
                    print(f"    ✅ Found: {ticker} - {company}")
                    successful_calls += 1
                else:
                    print(f"    ⚠️ Search unsuccessful: {result.get('error')}")
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"    ❌ Response parsing error: {e}")
        else:
            print(f"    ❌ MCP call failed")
        
        # Test 2: Comprehensive analysis via MCP
        print("  2. Testing MCP analyze_stock_comprehensive...")
        total_calls += 1
        
        data = self.make_request('POST', self.endpoints['call'],
                               json={
                                   "name": "analyze_stock_comprehensive", 
                                   "arguments": {"symbol": "AAPL"}
                               })
        
        if data and data.get('success'):
            try:
                result = json.loads(data['result'][0])
                if 'error' not in result:
                    symbol = result.get('symbol')
                    score = result.get('investment_score')
                    print(f"    ✅ {symbol} Analysis: Score {score}/100")
                    successful_calls += 1
                else:
                    print(f"    ❌ Analysis error: {result['error']}")
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"    ❌ Response parsing error: {e}")
        else:
            print(f"    ❌ MCP call failed")
        
        # Test 3: Stock comparison via MCP
        print("  3. Testing MCP compare_stocks_ytd...")
        total_calls += 1
        
        data = self.make_request('POST', self.endpoints['call'],
                               json={
                                   "name": "compare_stocks_ytd",
                                   "arguments": {"symbols": ["Apple", "Tesla", "Microsoft"]}
                               })
        
        if data and data.get('success'):
            try:
                result = json.loads(data['result'][0])
                total_analyzed = result.get('total_analyzed', 0)
                if total_analyzed > 0:
                    print(f"    ✅ Successfully compared {total_analyzed} stocks")
                    winner = result.get('winner')
                    if winner:
                        print(f"       Winner: {winner['symbol']} (Score: {winner['investment_score']})")
                    successful_calls += 1
                else:
                    print(f"    ⚠️ No stocks analyzed successfully")
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"    ❌ Response parsing error: {e}")
        else:
            print(f"    ❌ MCP call failed")
        
        # Test 4: Market trends via MCP
        print("  4. Testing MCP get_market_trends...")
        total_calls += 1
        
        data = self.make_request('POST', self.endpoints['call'],
                               json={
                                   "name": "get_market_trends",
                                   "arguments": {"symbols": ["AAPL", "TSLA", "MSFT", "GOOGL"]}
                               })
        
        if data and data.get('success'):
            try:
                result = json.loads(data['result'][0])
                market_score = result.get('overall_market_score', 0)
                stocks_analyzed = result.get('total_stocks_analyzed', 0)
                print(f"    ✅ Market analysis: {stocks_analyzed} stocks, overall score: {market_score}")
                successful_calls += 1
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"    ❌ Response parsing error: {e}")
        else:
            print(f"    ❌ MCP call failed")
        
        print(f"  📊 MCP Tool Results: {successful_calls}/{total_calls} successful")
        return successful_calls > 0
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success"""
        print("🧪 Starting Comprehensive Modal MCP Server Tests")
        print("=" * 60)
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Base URL: {self.base_url}")
        print()
        
        tests = [
            ("Health Check", self.test_health),
            ("Tools Listing", self.test_tools), 
            ("Smart Search", self.test_smart_search),
            ("Stock Analysis", self.test_stock_analysis),
            ("MCP Tool Calls", self.test_mcp_tools)
        ]
        
        passed_tests = 0
        
        for test_name, test_func in tests:
            try:
                print(f"🔄 Running: {test_name}")
                if test_func():
                    passed_tests += 1
                    print(f"✅ {test_name} - PASSED")
                else:
                    print(f"❌ {test_name} - FAILED")
                
                # Brief pause between tests
                time.sleep(2)
                
            except Exception as e:
                print(f"💥 {test_name} - CRASHED: {str(e)}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {passed_tests}/{len(tests)} tests")
        print(f"❌ Failed: {len(tests) - passed_tests}/{len(tests)} tests")
        print(f"📈 Success Rate: {(passed_tests/len(tests)*100):.1f}%")
        
        if passed_tests == len(tests):
            print("\n🎉 ALL TESTS PASSED! Your Modal MCP server is working perfectly!")
            print("\n🚀 Your server is ready for:")
            print("  • Integration with Claude Desktop")
            print("  • HTTP API calls from applications") 
            print("  • WebSocket connections")
            print("  • Production use")
        elif passed_tests >= len(tests) * 0.8:
            print("\n✅ MOSTLY WORKING! Some minor issues detected.")
            print("  • Core functionality is operational")
            print("  • Safe to use for most purposes")
            print("  • Consider investigating failed tests")
        else:
            print("\n⚠️ SIGNIFICANT ISSUES DETECTED!")
            print("  • Multiple tests failed")
            print("  • Check server logs and configuration")
            print("  • Verify secret setup and deployment")
        
        print(f"\n🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return passed_tests == len(tests)

def get_modal_url() -> str:
    """Get Modal URL from user input with validation"""
    print("🔗 Modal URL Input")
    print("-" * 30)
    print("Enter your Modal deployment base URL.")
    print("Examples:")
    print("  • https://username--mcp-stock-analysis-server")
    print("  • username--mcp-stock-analysis-server")
    print()
    
    while True:
        url = input("🌐 Enter your Modal base URL: ").strip()
        
        if not url:
            print("❌ URL cannot be empty. Please try again.")
            continue
        
        # Clean and validate URL
        if not url.startswith('http'):
            url = 'https://' + url
        
        # Remove any endpoint suffix
        if url.endswith('.modal.run'):
            parts = url.split('-')
            if len(parts) >= 3:
                # Remove endpoint part (like 'health', 'tools', etc.)
                url = '-'.join(parts[:-1])
        
        print(f"🎯 Using base URL: {url}")
        confirm = input("✅ Is this correct? (y/n): ").strip().lower()
        
        if confirm in ['y', 'yes', '']:
            return url
        else:
            print("Let's try again...\n")

def main():
    """Main test function"""
    print("🚀 Modal MCP Server Tester v1.0")
    print("================================")
    print("This script tests your deployed Modal MCP server")
    print("to ensure all endpoints and tools are working correctly.\n")
    
    try:
        # Get URL from user
        base_url = get_modal_url()
        
        # Create tester instance
        tester = ModalMCPTester(base_url)
        
        # Ask if user wants to run all tests or specific ones
        print("\n🎮 Test Options:")
        print("1. Run all tests (recommended)")
        print("2. Quick health check only")
        print("3. Exit")
        
        choice = input("\n👉 Enter your choice (1-3): ").strip()
        
        if choice == '1':
            success = tester.run_all_tests()
            if success:
                print("\n🎊 Congratulations! Your Modal MCP server is fully operational!")
            else:
                print("\n🔧 Some issues found. Check the output above for details.")
        
        elif choice == '2':
            print("\n🏥 Running quick health check...")
            if tester.test_health():
                print("✅ Health check passed! Server is responding.")
            else:
                print("❌ Health check failed. Server may be down.")
        
        elif choice == '3':
            print("👋 Goodbye!")
            return
        
        else:
            print("❌ Invalid choice. Please run the script again.")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Testing interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        print("Please check your Modal URL and try again.")

if __name__ == "__main__":
    main()