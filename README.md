# 🌌 Apeiron Tools

> **"Access Every Knowledge The World Has"**

**The world's first infinite AI tool orchestration network on Bittensor Subnet #**

Transform AI from "50 tools max" to unlimited capability through decentralized MCP server orchestration.

---

## 🚀 Vision

Traditional AI systems are crippled by context window limitations. While they can access maybe 50 tools at once, the real world requires thousands of specialized capabilities working together.

**Apeiron Tools breaks these chains.**

We create a decentralized network where:
- 🎯 **1,024+ miners** each host specialized MCP servers
- 🔧 **Unlimited tools** orchestrated across distributed infrastructure  
- 🧠 **Complex queries** spanning multiple domains simultaneously
- ⚡ **Sub-5 second** response times for orchestrated workflows
- 🌐 **True AI autonomy** beyond context limitations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BITTENSOR SUBNET #                    │
├─────────────────────────────────────────────────────────────┤
│  🔍 VALIDATORS                    🤖 MINERS (1024+)         │
│  ├─ Complex Query Generation      ├─ MCP Server Clusters   │
│  ├─ Multi-Criteria Scoring       ├─ Tool Orchestration    │
│  ├─ Performance Monitoring       ├─ Parallel Execution    │
│  └─ Network Health Management    └─ Result Aggregation    │
├─────────────────────────────────────────────────────────────┤
│                    MCP TOOL ECOSYSTEM                       │
│  📁 Filesystem  🌐 Web Content  💾 Databases  🔧 APIs      │
│  🗃️ Memory     📊 Analytics    🔍 Search     ⚙️ Compute    │
│  🔗 Git/GitHub 📧 Communication 📈 Monitoring 🎯 Custom    │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Use Cases

### 🧠 **Research & Analysis**
"Analyze this market: gather competitor data from 20+ sources, cross-reference with patent databases, analyze financial trends, generate strategic insights, and create an interactive dashboard"

### 🔧 **Development Workflows**  
"Build a complete CI/CD pipeline: analyze code quality across repositories, run comprehensive tests, generate documentation, deploy to multiple environments, and notify stakeholders"

### 📊 **Data Operations**
"Process this dataset: clean and validate data, perform statistical analysis, generate visualizations, store in multiple formats, create automated reports, and set up monitoring alerts"

### 🎓 **Knowledge Management**
"Create a knowledge base: index all documents, extract entities and relationships, enable semantic search, generate summaries, and provide intelligent recommendations"

## 🏃‍♂️ Quick Start

### 🤖 Running a Miner

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install MCP servers (requires Node.js)
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-fetch
npm install -g @modelcontextprotocol/server-memory
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-sqlite

# 3. Register on Bittensor subnet
btcli subnet register --netuid 

# 4. Run the miner
python neurons/miner.py --wallet.name your_wallet --wallet.hotkey your_hotkey
```

### 🔍 Running a Validator

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Register on Bittensor subnet  
btcli subnet register --netuid 

# 3. Run the validator
python neurons/validator.py --wallet.name your_wallet --wallet.hotkey your_hotkey
```

## 📋 Installation

### Prerequisites
- **Python 3.8+**
- **Node.js 18+** (for MCP servers)
- **Git**
- **Bittensor CLI**

### Full Installation

```bash
# 1. Clone the repository
git clone https://github.com/apeiron-tools/apeiron-tools.git
cd apeiron-tools

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install MCP servers
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-fetch
npm install -g @modelcontextprotocol/server-memory
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-sqlite

# 5. Install development tools (optional)
pip install -r requirements.txt[dev]
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY="your_anthropic_key"  # For MCP servers
export BITTENSOR_WALLET_NAME="your_wallet"
export BITTENSOR_HOTKEY_NAME="your_hotkey"

# Optional
export APEIRON_LOG_LEVEL="INFO"
export APEIRON_MAX_CONCURRENT_TOOLS="20"
export APEIRON_HEALTH_CHECK_INTERVAL="30"
```

### MCP Server Configuration

Edit `apeiron/registry.py` to customize your MCP server setup:

```python
# Add custom MCP servers
custom_servers = [
    MCPServerConfig(
        server_id="custom_api",
        name="Custom API Server", 
        category="api",
        host="localhost",
        port=8010,
        command="node",
        args=["path/to/your/mcp-server.js"],
        tools=["fetch_data", "process_response"]
    )
]
```

## 🎖️ Scoring System

Miners are evaluated across five key dimensions:

| Criteria | Weight | Description |
|----------|--------|-------------|
| 🎯 **Tool Coverage** | 30% | Relevant tool selection and comprehensive coverage |
| 💎 **Result Quality** | 25% | Accuracy, completeness, and relevance of responses |
| ⚡ **Execution Speed** | 20% | Response time and parallel processing efficiency |
| 🛡️ **Error Handling** | 15% | Graceful failure management and recovery |
| 🔧 **Resource Efficiency** | 10% | Optimal tool selection and server utilization |

## 🧪 Example Queries

### Simple Query
```
"List all Python files in this repository and show their sizes"
```

### Medium Complexity  
```
"Analyze this GitHub repository: check code quality, identify dependencies, 
generate documentation, and create a deployment checklist"
```

### Complex Orchestration
```
"Research AI startups: gather data from Crunchbase, AngelList, and news sources,
analyze funding trends, identify market gaps, cross-reference with patent data,
generate competitive landscape report, and create investment recommendations"
```

## 🏆 Performance Targets

- **⚡ Response Time**: <5 seconds average
- **🎯 Success Rate**: >90% query completion
- **🔧 Tool Support**: 50+ tools per query (scaling to 1000+)
- **🌐 Network Coverage**: Support for multi-domain queries
- **⚖️ Decentralization**: 1,024+ active miners

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=apeiron --cov-report=html

# Run specific test categories
pytest tests/test_protocol.py
pytest tests/test_orchestrator.py -v
```

### Code Quality

```bash
# Format code
black apeiron/ neurons/ tests/
isort apeiron/ neurons/ tests/

# Lint code
flake8 apeiron/ neurons/
mypy apeiron/ neurons/

# All quality checks
make quality  # If Makefile is present
```

### Adding New MCP Servers

1. **Install the MCP server** (usually via npm)
2. **Add configuration** in `apeiron/registry.py`
3. **Update tool mappings** in `apeiron/orchestrator.py`
4. **Add tests** in `tests/test_integration.py`

## 📊 Monitoring

### Miner Status
```bash
# Check miner health
curl http://localhost:8091/health

# View server status
tail -f miner.log | grep "MCP SERVER STATUS"
```

### Validator Metrics
```bash
# Check validation results
tail -f validator.log | grep "LEADERBOARD"

# Monitor network health
python scripts/network_monitor.py
```

## 🤝 Contributing

We welcome contributions to the infinite AI tool ecosystem!

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-tool`)
3. **Commit** your changes (`git commit -m 'Add amazing tool'`)
4. **Push** to the branch (`git push origin feature/amazing-tool`)
5. **Open** a Pull Request

### Development Guidelines

- Follow **PEP 8** style guidelines
- Add **comprehensive tests** for new features
- Include **docstrings** for all public methods
- Update **documentation** for API changes
- Ensure **async/await** patterns for performance

## 🆘 Troubleshooting

### Common Issues

**❌ MCP servers not starting**
```bash
# Check Node.js version
node --version  # Should be 18+

# Reinstall MCP servers
npm uninstall -g @modelcontextprotocol/server-*
npm install -g @modelcontextprotocol/server-filesystem
```

**❌ Bittensor registration fails**
```bash
# Check wallet balance
btcli wallet balance

# Verify subnet exists
btcli subnet list | grep 
```

**❌ Memory issues with large queries**
```bash
# Reduce concurrent tools
export APEIRON_MAX_CONCURRENT_TOOLS="10"

# Increase timeout
export APEIRON_DEFAULT_TIMEOUT="60"
```

### Getting Help

- 💬 **Discord**: [Join our community](https://discord.gg/apeiron-tools)
- 🐛 **Issues**: [GitHub Issues](https://github.com/apeiron-tools/apeiron-tools/issues)
- 📧 **Email**: contact@apeiron.tools
- 🐦 **Twitter**: [@ApeironTools](https://twitter.com/ApeironTools)

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Bittensor Foundation** for the decentralized AI infrastructure
- **Anthropic** for the Model Context Protocol (MCP)
- **The AI community** for pushing the boundaries of what's possible

---

## 🌟 Join the Revolution

**Apeiron Tools isn't just another AI project - it's the future of unlimited AI capability.**

Break free from context windows. Join the network that makes AI truly infinite.

```bash
git clone https://github.com/apeiron-tools/apeiron-tools.git
cd apeiron-tools
python neurons/miner.py --wallet.name your_wallet
```

**The infinite starts now. 🚀**

---

*Built with ❤️ by the Apeiron Tools team*

*"In the beginning was the Word, and the Word was unlimited." - Apeiron Manifesto*
