# Contract Document Analyzer API

A sophisticated AI-powered contract analysis system built with FastAPI and multi-agent architecture for comprehensive legal document processing and risk assessment.

## 🚀 Features

- **Multi-Agent Analysis System**: Employs specialized AI agents for different aspects of contract analysis
- **Real-time Risk Assessment**: Calculates risk scores and identifies high-risk clauses
- **Intelligent Document Classification**: Automatically determines if a document is a contract
- **Party Extraction**: Identifies all parties and their roles, stakes, and obligations
- **Financial Terms Analysis**: Extracts payment terms, deadlines, and monetary obligations
- **Compliance Gap Detection**: Identifies potential legal and regulatory issues
- **Async Processing**: Supports both synchronous and asynchronous analysis
- **File Upload Support**: Direct file upload and processing
- **RESTful API**: Full REST API with comprehensive documentation

## 🏗️ Architecture

### Multi-Agent System

The system uses a sophisticated multi-agent architecture:

1. **DocumentClassificationAgent**: Determines if the document is a legal contract
2. **ClauseAnalysisAgent**: Analyzes individual clauses for risk and compliance
3. **InsightsAgent**: Extracts key contract insights and metadata
4. **PartiesAgent**: Identifies parties and their relationships
5. **MetricsAgent**: Calculates comprehensive risk metrics
6. **FinalReportAgent**: Generates structured analysis reports

### Technology Stack

- **Framework**: FastAPI (Python 3.8+)
- **AI/ML**: 
  - LangChain for LLM orchestration
  - Azure OpenAI for language processing
  - HuggingFace Transformers for embeddings
- **Vector Database**: ChromaDB for semantic search
- **Text Processing**: Dynamic chunking with RecursiveCharacterTextSplitter
- **Async Processing**: Python asyncio for concurrent operations
- **API Documentation**: Auto-generated OpenAPI/Swagger docs

## 📋 Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access with GPT-4 or GPT-3.5-turbo deployment
- 4GB+ RAM (for embedding models)
- 2GB+ disk space (for model storage)

## 🔧 Installation

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd contract-document-analyzer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

4. **Run the setup script**
```bash
python setup_and_run.py
```

### Manual Setup

1. **Create virtual environment**
```bash
python -m venv contract_analyzer_env
source contract_analyzer_env/bin/activate  # On Windows: contract_analyzer_env\Scripts\activate
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
Create a `.env` file with the following:
```env
# Azure OpenAI Configuration (Required)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=your-gpt-deployment-name

# Optional Configuration
DEBUG=True
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=10
MAX_DOCUMENT_LENGTH=100000
CHROMA_PERSIST_DIRECTORY=./chroma_db
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

4. **Start the server**
```bash
uvicorn contract_analyzer_api:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Setup

1. **Using Docker Compose**
```bash
docker-compose up -d
```

2. **Using Docker directly**
```bash
docker build -t contract-analyzer .
docker run -p 8000:8000 --env-file .env contract-analyzer
```

## 🚀 Usage

### API Endpoints

The API will be available at `http://localhost:8000` with the following endpoints:

#### Core Analysis
- `POST /analyze` - Synchronous contract analysis
- `POST /analyze-async` - Start asynchronous analysis
- `GET /status/{document_id}` - Check analysis status
- `GET /result/{document_id}` - Get analysis results

#### File Operations
- `POST /upload-analyze` - Upload and analyze file

#### Management
- `GET /health` - Health check
- `DELETE /cleanup/{document_id}` - Clean up analysis data

### API Documentation

Once the server is running, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Request Examples

#### Synchronous Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "Your contract text here...",
    "analysis_depth": "standard"
  }'
```

#### Asynchronous Analysis
```bash
# Start analysis
curl -X POST "http://localhost:8000/analyze-async" \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "Your contract text here...",
    "analysis_depth": "comprehensive"
  }'

# Check status
curl "http://localhost:8000/status/{document_id}"

# Get results
curl "http://localhost:8000/result/{document_id}"
```

#### File Upload
```bash
curl -X POST "http://localhost:8000/upload-analyze" \
  -F "file=@contract.txt"
```

### Python Client Example

```python
import requests

# Initialize client
client = requests.Session()
base_url = "http://localhost:8000"

# Analyze contract
contract_text = """
SERVICE AGREEMENT
This agreement is between Company A and Company B...
"""

response = client.post(f"{base_url}/analyze", json={
    "document_text": contract_text,
    "analysis_depth": "standard"
})

result = response.json()
print(f"Risk Score: {result['risk_assessment']['overall_risk_score']}")
print(f"Contract Type: {result['insights']['contract_type']}")
```

## 📊 Response Format

### Analysis Response Structure

```json
{
  "document_id": "contract_20241128_143022_a1b2c3d4",
  "timestamp": "2024-11-28T14:30:22.123456",
  "is_contract": true,
  "confidence_score": 0.95,
  "processing_time_seconds": 12.34,
  "risk_assessment": {
    "overall_risk_score": 0.65,
    "risk_distribution": {
      "HIGH": 2,
      "MEDIUM": 5,
      "LOW": 3
    },
    "high_risk_areas": ["Payment Terms", "Termination Clause"],
    "compliance_gaps": ["Missing data protection clause"],
    "mitigation_strategies": ["Review high-risk clauses with legal team"]
  },
  "insights": {
    "contract_type": "Service Agreement",
    "key_themes": ["payment terms", "service delivery", "termination"],
    "critical_dates": ["2024-12-31", "2025-06-01"],
    "financial_terms": ["$50,000 monthly fee", "2% late penalty"],
    "jurisdiction": "New York, USA",
    "termination_clauses": ["30 days notice required"]
  },
  "parties": [
    {
      "name": "ABC Corp",
      "role": "Service Provider",
      "stakes": ["receive payment", "maintain service quality"],
      "obligations": ["provide monthly reports", "24/7 support"],
      "rights": ["terminate for non-payment"]
    }
  ],
  "clause_analyses": [
    {
      "clause_id": "Section_1_a1b2c3d4",
      "section": "Payment Terms",
      "content": "Client agrees to pay...",
      "risk_level": "HIGH",
      "risk_score": 0.8,
      "ambiguity_flags": ["unclear payment schedule"],
      "compliance_issues": ["potential late payment disputes"],
      "recommendations": ["specify exact payment dates"]
    }
  ],
  "executive_summary": "Service Agreement with overall medium risk...",
  "recommendations": ["Clarify payment terms", "Add termination procedures"]
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint | - | Yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | - | Yes |
| `AZURE_OPENAI_API_VERSION` | API version | `2024-02-01` | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Model deployment name | - | Yes |
| `DEBUG` | Enable debug mode | `False` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `MAX_FILE_SIZE_MB` | Max upload file size | `10` | No |
| `MAX_DOCUMENT_LENGTH` | Max document length | `100000` | No |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | `./chroma_db` | No |
| `EMBEDDINGS_MODEL` | HuggingFace model name | `sentence-transformers/all-MiniLM-L6-v2` | No |

### Analysis Depth Options

- `quick`: Fast analysis with basic risk assessment
- `standard`: Comprehensive analysis with detailed insights
- `comprehensive`: Full analysis with extensive party and clause breakdown

## 🧪 Testing

### Automated Tests

Run the test suite:
```bash
python test_api.py
```

### Manual Testing with cURL

```bash
# Make the script executable
chmod +x curl_examples.sh

# Run examples
./curl_examples.sh
```

### Postman Collection

Import the provided Postman collection (`contract_analyzer_postman_collection.json`) for interactive testing.

### Sample Test Documents

The repository includes sample contracts for testing:
- Simple service agreement
- Complex master service agreement  
- Employment contract
- Non-contract document (for classification testing)

## 🔍 Key Features Deep Dive

### Dynamic Text Chunking

The system uses an intelligent chunking algorithm that adapts to document size:

```python
chunk_size = min(C_max, max(C_min, int(alpha * (N ** 0.6))))
```

Where:
- `N` = document length
- `C_min` = minimum chunk size (400)
- `C_max` = maximum chunk size (2000)
- `alpha` = scaling factor (30)
- `beta` = overlap ratio (0.2)

### Risk Scoring Algorithm

Risk scores are calculated using multiple factors:
- **Clause complexity**: Length and legal terminology density
- **Ambiguity detection**: Vague or unclear language
- **Compliance gaps**: Missing standard clauses
- **Financial risk**: Payment terms and penalty analysis
- **Termination risk**: Exit clause analysis

### JSON Serialization Safety

The system includes robust JSON handling to prevent serialization errors:
- Custom JSON encoder for numpy types
- Automatic conversion of Python objects
- Safe JSON parsing with error recovery
- Validation of all response data

## 🚨 Troubleshooting

### Common Issues

1. **"Service unavailable: LLM or embeddings not properly configured"**
   - Check your `.env` file has correct Azure OpenAI credentials
   - Verify the deployment name matches your Azure OpenAI deployment
   - Ensure your Azure OpenAI service is running

2. **"Module not found" errors**
   - Run `pip install -r requirements.txt`
   - Check you're using the correct Python environment

3. **Out of memory errors**
   - Reduce document size or chunking parameters
   - Use `analysis_depth: "quick"` for large documents
   - Increase system RAM if possible

4. **Slow analysis times**
   - Use asynchronous endpoints for large documents
   - Consider reducing chunk size for faster processing
   - Check Azure OpenAI quota limits

5. **JSON serialization errors**
   - The system includes automatic fixes for common JSON issues
   - If problems persist, check the document encoding

### Performance Optimization

- **Concurrent Processing**: Use async endpoints for multiple documents
- **Caching**: Results are cached temporarily for repeat requests
- **Chunking**: Optimize chunk size based on document characteristics
- **Rate Limiting**: Implement client-side rate limiting for API calls

### Monitoring and Logging

Enable detailed logging by setting:
```env
DEBUG=True
LOG_LEVEL=DEBUG
```

Logs include:
- Request/response timing
- Error stack traces
- Model inference metrics
- Memory usage statistics

## 🚀 Production Deployment

### Scaling Considerations

1. **Horizontal Scaling**: Deploy multiple API instances behind a load balancer
2. **Database**: Use external vector database (Pinecone, Weaviate) for production
3. **Caching**: Implement Redis for result caching
4. **Queue System**: Use Celery for heavy async processing
5. **Monitoring**: Add comprehensive monitoring (Prometheus, Grafana)

### Security

- **API Keys**: Use proper API key management
- **Rate Limiting**: Implement rate limiting per client
- **Input Validation**: Validate all input documents
- **HTTPS**: Use SSL/TLS in production
- **CORS**: Configure CORS settings appropriately

### Docker Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  contract-analyzer:
    build: .
    restart: unless-stopped
    environment:
      - DEBUG=False
      - LOG_LEVEL=INFO
    ports:
      - "8000:8000"
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

## 📝 API Versioning

Current API version: `v1.0.0`

The API follows semantic versioning:
- Major version: Breaking changes
- Minor version: New features (backward compatible)
- Patch version: Bug fixes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 contract_analyzer_api.py

# Run type checking
mypy contract_analyzer_api.py

# Format code
black contract_analyzer_api.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the API docs at `/docs` endpoint
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Join community discussions
- **Email**: Contact support team

## 🔮 Roadmap

### Upcoming Features

- [ ] PDF and DOCX file support
- [ ] Batch processing for multiple contracts
- [ ] Custom risk scoring models
- [ ] Integration with legal databases
- [ ] Multi-language support
- [ ] Advanced compliance checking (GDPR, CCPA)
- [ ] Contract comparison and diff analysis
- [ ] Integration with popular contract management systems

### Version History

- **v1.0.0** (Current): Initial release with full multi-agent analysis
- **v0.9.0**: Beta release with core functionality
- **v0.8.0**: Alpha release with basic analysis

---

## Quick Start Commands

```bash
# Clone and setup
git clone <repo-url>
cd contract-document-analyzer
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials

# Run server
uvicorn contract_analyzer_api:app --reload

# Test API
curl http://localhost:8000/health

# View documentation
open http://localhost:8000/docs
```

For detailed examples and advanced usage, see the `/examples` directory and API documentation.