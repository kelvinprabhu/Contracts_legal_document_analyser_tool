# contract_analyzer_api.py

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
import json
import numpy as np
import uuid
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import docx2txt
import tempfile
import os
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

# ================= CONFIGURATION =================
app = FastAPI(
    title="Contract Document Analyzer",
    description="AI-powered contract analysis with multi-agent system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CUSTOM JSON ENCODER =================
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and other non-serializable objects"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def safe_json_serialize(data):
    """Safely serialize data to JSON string"""
    return json.dumps(data, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)

def safe_json_loads(data_str):
    """Safely load JSON with error handling"""
    try:
        return json.loads(data_str)
    except json.JSONDecodeError as e:
        # Try to fix common issues
        fixed_str = data_str.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
        try:
            return json.loads(fixed_str)
        except:
            raise ValueError(f"Invalid JSON format: {e}")

# ================= PYDANTIC MODELS =================
class RiskLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class AnalysisRequest(BaseModel):
    document_text: str = Field(..., description="The contract document text to analyze")
    analysis_depth: Optional[str] = Field(default="standard", description="Analysis depth: quick, standard, or comprehensive")

class ClauseAnalysisResponse(BaseModel):
    clause_id: str
    section: str
    content: str
    risk_level: RiskLevel
    risk_score: float
    ambiguity_flags: List[str]
    compliance_issues: List[str]
    recommendations: List[str]

class ContractInsightsResponse(BaseModel):
    contract_type: str
    key_themes: List[str]
    critical_dates: List[str]
    financial_terms: List[str]
    jurisdiction: str
    termination_clauses: List[str]

class PartyInfoResponse(BaseModel):
    name: str
    role: str
    stakes: List[str]
    obligations: List[str]
    rights: List[str]

class RiskAssessmentResponse(BaseModel):
    overall_risk_score: float
    risk_distribution: Dict[str, int]
    high_risk_areas: List[str]
    compliance_gaps: List[str]
    mitigation_strategies: List[str]

class AnalysisResponse(BaseModel):
    document_id: str
    timestamp: str
    is_contract: bool
    confidence_score: float
    risk_assessment: RiskAssessmentResponse
    insights: ContractInsightsResponse
    parties: List[PartyInfoResponse]
    clause_analyses: List[ClauseAnalysisResponse]
    executive_summary: str
    recommendations: List[str]
    processing_time_seconds: float

class StatusResponse(BaseModel):
    status: str
    message: str
    document_id: Optional[str] = None

# ================= GLOBAL VARIABLES =================
# Store analysis results temporarily (in production, use Redis or database)
analysis_cache = {}
processing_status = {}

# Initialize embeddings (load once)
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
except Exception as e:
    print(f"Warning: Could not load embeddings: {e}")
    embeddings = None

# Initialize LLM (configure with your Azure OpenAI credentials)
try:
    llm = AzureChatOpenAI(
        deployment_name="neostats_hackathon_api_v1",
        model="gpt-4.1",
        temperature=0,
        api_version="2024-05-01-preview",
        api_key="Azure_Open_API",  # Replace with your actual key
        azure_endpoint="Https://neoaihackathon.cognitiveservices.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview" 
    )
except Exception as e:
    print(f"Warning: Could not initialize LLM: {e}")
    llm = None

# ================= CORE FUNCTIONS =================
def dynamic_chunker(text: str, C_min=400, C_max=2000, alpha=30, beta=0.2):
    """Dynamic text chunking with formula-based sizing"""
    N = len(text)
    chunk_size = min(C_max, max(C_min, int(alpha * (N ** 0.6))))
    overlap = int(beta * chunk_size)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    return chunks, chunk_size, overlap

# ================= MULTI-AGENT CLASSES =================
class DocumentClassificationAgent:
    """Agent 1: Determines if document is a contract"""
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

    async def analyze(self, text: str):
        prompt = f"""
        Determine if the following document is a contract or legal agreement:
        {text[:2000]}...

        Respond with valid JSON only:
        {{
            "is_contract": true,
            "confidence": 0.95,
            "reasoning": "This appears to be a legal contract because..."
        }}
        """
        messages = [
            SystemMessage(content="You are a legal document classifier. Respond only with valid JSON."),
            HumanMessage(content=prompt)
        ]
        try:
            response = await self.llm.ainvoke(messages)
            return safe_json_loads(response.content)
        except Exception as e:
            return {"is_contract": True, "confidence": 0.5, "reasoning": f"Error in classification: {e}"}

class ClauseAnalysisAgent:
    """Agent 2.1: Risk analysis per clause"""
    def __init__(self, llm: AzureChatOpenAI, retriever):
        self.llm = llm
        self.retriever = retriever

    async def analyze_clause(self, section_title: str, content: str):
        try:
            # Retrieve relevant context
            context_docs = self.retriever.get_relevant_documents(content)
            context_text = "\n".join([doc.page_content for doc in context_docs[:3]])

            prompt = f"""
            Analyze this contract clause for risk and compliance:
            Section: {section_title}
            Content: {content[:1000]}
            Context: {context_text[:1000]}

            Respond with valid JSON only:
            {{
                "risk_level": "HIGH",
                "risk_score": 0.8,
                "ambiguity_flags": ["unclear payment terms", "missing deadline"],
                "compliance_issues": ["potential GDPR violation"],
                "recommendations": ["clarify payment schedule", "add termination clause"]
            }}
            """
            messages = [
                SystemMessage(content="You are a contract risk analyst. Respond only with valid JSON."),
                HumanMessage(content=prompt)
            ]
            response = await self.llm.ainvoke(messages)
            result = safe_json_loads(response.content)
            
            return {
                "clause_id": f"{section_title[:10]}_{uuid.uuid4().hex[:8]}",
                "section": section_title,
                "content": content[:500] + "..." if len(content) > 500 else content,
                "risk_level": result.get("risk_level", "MEDIUM"),
                "risk_score": float(result.get("risk_score", 0.5)),
                "ambiguity_flags": result.get("ambiguity_flags", []),
                "compliance_issues": result.get("compliance_issues", []),
                "recommendations": result.get("recommendations", [])
            }
        except Exception as e:
            return {
                "clause_id": f"{section_title[:10]}_{uuid.uuid4().hex[:8]}",
                "section": section_title,
                "content": content[:500],
                "risk_level": "MEDIUM",
                "risk_score": 0.5,
                "ambiguity_flags": [],
                "compliance_issues": [f"Analysis error: {str(e)}"],
                "recommendations": ["Manual review required"]
            }

class InsightsAgent:
    """Agent 2.2: Extracts contract insights"""
    def __init__(self, llm: AzureChatOpenAI, retriever):
        self.llm = llm
        self.retriever = retriever

    async def analyze(self, full_text: str):
        try:
            context_docs = self.retriever.get_relevant_documents(full_text[:2000])
            context_text = "\n".join([doc.page_content for doc in context_docs[:3]])

            prompt = f"""
            Extract key insights from this contract:
            Contract: {full_text[:2000]}
            Context: {context_text[:1000]}

            Respond with valid JSON only:
            {{
                "contract_type": "Service Agreement",
                "key_themes": ["payment terms", "intellectual property", "termination"],
                "critical_dates": ["2024-12-31", "2025-06-01"],
                "financial_terms": ["$50,000 monthly fee", "2% late payment penalty"],
                "jurisdiction": "New York, USA",
                "termination_clauses": ["30 days notice required", "immediate termination for breach"]
            }}
            """
            messages = [
                SystemMessage(content="You are a contract insights expert. Respond only with valid JSON."),
                HumanMessage(content=prompt)
            ]
            response = await self.llm.ainvoke(messages)
            return safe_json_loads(response.content)
        except Exception as e:
            return {
                "contract_type": "Unknown",
                "key_themes": [],
                "critical_dates": [],
                "financial_terms": [],
                "jurisdiction": "Unknown",
                "termination_clauses": []
            }

class PartiesAgent:
    """Agent 2.3: Extract parties info"""
    def __init__(self, llm: AzureChatOpenAI, retriever):
        self.llm = llm
        self.retriever = retriever

    async def analyze(self, full_text: str):
        try:
            context_docs = self.retriever.get_relevant_documents(full_text[:2000])
            context_text = "\n".join([doc.page_content for doc in context_docs[:3]])
            
            prompt = f"""
            Identify all parties and their details from this contract:
            Contract: {full_text[:2000]}
            Context: {context_text[:1000]}

            Respond with valid JSON only (array of parties):
            [
                {{
                    "name": "Company ABC Inc.",
                    "role": "Service Provider",
                    "stakes": ["receive payment", "maintain service quality"],
                    "obligations": ["provide monthly reports", "24/7 support"],
                    "rights": ["terminate for non-payment", "intellectual property ownership"]
                }}
            ]
            """
            messages = [
                SystemMessage(content="You are an expert in contract parties analysis. Respond only with valid JSON array."),
                HumanMessage(content=prompt)
            ]
            response = await self.llm.ainvoke(messages)
            return safe_json_loads(response.content)
        except Exception as e:
            return [{"name": "Unknown", "role": "Unknown", "stakes": [], "obligations": [], "rights": []}]

class MetricsAgent:
    """Agent 3.1: Calculates metrics"""
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

    async def analyze(self, clause_analyses: List[Dict]):
        try:
            scores = [float(c.get("risk_score", 0.5)) for c in clause_analyses]
            overall_score = float(np.mean(scores)) if scores else 0.5
            
            risk_dist = {
                "HIGH": len([c for c in clause_analyses if c.get("risk_level") == "HIGH"]),
                "MEDIUM": len([c for c in clause_analyses if c.get("risk_level") == "MEDIUM"]),
                "LOW": len([c for c in clause_analyses if c.get("risk_level") == "LOW"])
            }
            
            high_risk_areas = [c["section"] for c in clause_analyses if c.get("risk_level") == "HIGH"]
            compliance_gaps = []
            for c in clause_analyses:
                compliance_gaps.extend(c.get("compliance_issues", []))
            
            mitigation_strategies = [
                "Review high-risk clauses with legal team",
                "Clarify ambiguous terms",
                "Add missing compliance requirements",
                "Update termination clauses",
                "Establish clear payment terms"
            ]
            
            return {
                "overall_risk_score": overall_score,
                "risk_distribution": risk_dist,
                "high_risk_areas": high_risk_areas[:10],
                "compliance_gaps": compliance_gaps[:10],
                "mitigation_strategies": mitigation_strategies
            }
        except Exception as e:
            return {
                "overall_risk_score": 0.5,
                "risk_distribution": {"HIGH": 0, "MEDIUM": 1, "LOW": 0},
                "high_risk_areas": [],
                "compliance_gaps": [f"Metrics calculation error: {str(e)}"],
                "mitigation_strategies": ["Manual review required"]
            }

class FinalReportAgent:
    """Agent 4: Generate final report"""
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm

    async def generate(self, analysis_components: Dict[str, Any]):
        insights = analysis_components['insights']
        risk_assessment = analysis_components['risk_assessment']
        
        summary = f"Contract Analysis Summary: Type: {insights.get('contract_type', 'Unknown')}, " \
                 f"Overall Risk Score: {risk_assessment['overall_risk_score']:.2f}, " \
                 f"High Risk Areas: {len(risk_assessment['high_risk_areas'])}"

        return {
            "document_id": f"contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "is_contract": analysis_components['is_contract'],
            "confidence_score": float(analysis_components['confidence']),
            "risk_assessment": risk_assessment,
            "insights": insights,
            "parties": analysis_components['parties'],
            "clause_analyses": analysis_components['clause_analyses'],
            "executive_summary": summary,
            "recommendations": risk_assessment['mitigation_strategies']
        }

# ================= MULTI-AGENT ORCHESTRATOR =================
class MultiAgentContractAnalyzer:
    def __init__(self, llm: AzureChatOpenAI, retriever):
        self.llm = llm
        self.retriever = retriever
        self.classifier = DocumentClassificationAgent(llm)
        self.clause_agent = ClauseAnalysisAgent(llm, retriever)
        self.insights_agent = InsightsAgent(llm, retriever)
        self.parties_agent = PartiesAgent(llm, retriever)
        self.metrics_agent = MetricsAgent(llm)
        self.reporter = FinalReportAgent(llm)

    async def analyze_contract(self, contract_text: str):
        try:
            # Step 1: classify
            classification = await self.classifier.analyze(contract_text)
            if not classification.get("is_contract", False) or classification.get("confidence", 0) < 0.3:
                return {
                    "error": "Document not recognized as contract", 
                    "confidence": classification.get("confidence", 0)
                }

            # Step 2: split into sections (improved section detection)
            sections = {}
            paragraphs = contract_text.split("\n\n")
            for i, chunk in enumerate(paragraphs):
                if len(chunk.strip()) > 50:  # Skip very short sections
                    title = f"Section_{i+1}"
                    # Try to extract actual section titles
                    first_line = chunk.split("\n")[0].strip()
                    if len(first_line) < 100 and any(char.isdigit() for char in first_line):
                        title = first_line[:50] + "..." if len(first_line) > 50 else first_line
                    sections[title] = chunk

            # Step 3: parallel analysis with error handling
            clause_tasks = [
                self.clause_agent.analyze_clause(title, content) 
                for title, content in list(sections.items())[:10]  # Limit to first 10 sections
            ]
            
            clause_results, insights, parties = await asyncio.gather(
                asyncio.gather(*clause_tasks, return_exceptions=True),
                self.insights_agent.analyze(contract_text),
                self.parties_agent.analyze(contract_text),
                return_exceptions=True
            )

            # Handle exceptions in clause results
            clause_results = [result for result in clause_results if not isinstance(result, Exception)]
            
            # Step 4: metrics
            risk_assessment = await self.metrics_agent.analyze(clause_results)

            # Step 5: final report
            analysis_components = {
                'is_contract': True,
                'confidence': classification.get("confidence", 0.8),
                'clause_analyses': clause_results,
                'insights': insights if not isinstance(insights, Exception) else {},
                'parties': parties if not isinstance(parties, Exception) else [],
                'risk_assessment': risk_assessment
            }
            
            final_report = await self.reporter.generate(analysis_components)
            return final_report

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

# ================= API ENDPOINTS =================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Health check endpoint"""
    return {
        "message": "Contract Document Analyzer API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Detailed health check"""
    status = {
        "api": "healthy",
        "embeddings": "loaded" if embeddings else "error",
        "llm": "loaded" if llm else "error",
        "timestamp": datetime.now().isoformat()
    }
    return status

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_contract(request: AnalysisRequest):
    """Analyze a contract document"""
    if not llm or not embeddings:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable: LLM or embeddings not properly configured"
        )
    
    start_time = datetime.now()
    
    try:
        # Create vectorstore for this document
        chunks, chunk_size, overlap = dynamic_chunker(request.document_text)
        
        # Create temporary collection name
        collection_name = f"contract_temp_{uuid.uuid4().hex[:8]}"
        
        vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=collection_name
        )
        
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Initialize analyzer
        analyzer = MultiAgentContractAnalyzer(llm, retriever)
        
        # Perform analysis
        result = await analyzer.analyze_contract(request.document_text)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Add processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        result["processing_time_seconds"] = processing_time
        
        # Clean up vectorstore
        try:
            vectordb.delete_collection()
        except:
            pass  # Ignore cleanup errors
        
        return JSONResponse(content=json.loads(safe_json_serialize(result)))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-async", response_model=StatusResponse)
async def analyze_contract_async(background_tasks: BackgroundTasks, request: AnalysisRequest):
    """Start asynchronous contract analysis"""
    if not llm or not embeddings:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable: LLM or embeddings not properly configured"
        )
    
    document_id = f"async_contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    processing_status[document_id] = {"status": "processing", "progress": 0}
    
    async def process_contract():
        try:
            processing_status[document_id]["progress"] = 10
            
            # Create vectorstore
            chunks, _, _ = dynamic_chunker(request.document_text)
            collection_name = f"contract_async_{uuid.uuid4().hex[:8]}"
            
            processing_status[document_id]["progress"] = 30
            
            vectordb = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                collection_name=collection_name
            )
            
            retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            processing_status[document_id]["progress"] = 50
            
            # Perform analysis
            analyzer = MultiAgentContractAnalyzer(llm, retriever)
            result = await analyzer.analyze_contract(request.document_text)
            
            processing_status[document_id]["progress"] = 90
            
            if "error" not in result:
                analysis_cache[document_id] = result
                processing_status[document_id] = {"status": "completed", "progress": 100}
            else:
                processing_status[document_id] = {"status": "error", "error": result["error"], "progress": 100}
            
            # Cleanup
            try:
                vectordb.delete_collection()
            except:
                pass
                
        except Exception as e:
            processing_status[document_id] = {"status": "error", "error": str(e), "progress": 100}
    
    background_tasks.add_task(process_contract)
    
    return StatusResponse(
        status="accepted",
        message="Analysis started",
        document_id=document_id
    )

@app.get("/status/{document_id}", response_model=Dict[str, Any])
async def get_analysis_status(document_id: str):
    """Get status of asynchronous analysis"""
    if document_id not in processing_status:
        raise HTTPException(status_code=404, detail="Document ID not found")
    
    status_info = processing_status[document_id].copy()
    
    if status_info["status"] == "completed" and document_id in analysis_cache:
        status_info["result_available"] = True
    
    return status_info

@app.get("/result/{document_id}", response_model=AnalysisResponse)
async def get_analysis_result(document_id: str):
    """Get result of completed analysis"""
    if document_id not in analysis_cache:
        if document_id in processing_status:
            status = processing_status[document_id]["status"]
            if status == "processing":
                raise HTTPException(status_code=202, detail="Analysis still in progress")
            elif status == "error":
                raise HTTPException(status_code=400, detail=processing_status[document_id].get("error", "Analysis failed"))
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = analysis_cache[document_id]
    return JSONResponse(content=json.loads(safe_json_serialize(result)))

@app.post("/upload-analyze", response_model=AnalysisResponse)
async def upload_and_analyze(file: UploadFile = File(...)):
    """Upload a document file (.txt, .pdf, .docx, .doc) and analyze it"""
    if not file.filename.lower().endswith(('.txt', '.pdf', '.docx', '.doc')):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extract text
        document_text = None

        if file.filename.lower().endswith('.txt'):
            with open(tmp_path, "r", encoding="utf-8") as f:
                document_text = f.read()

        elif file.filename.lower().endswith(('.docx', '.doc')):
            document_text = docx2txt.process(tmp_path)

        elif file.filename.lower().endswith('.pdf'):
            document_text = ""
            reader = PdfReader(tmp_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    document_text += text + "\n"

        if not document_text or document_text.strip() == "":
            raise HTTPException(status_code=400, detail="Could not extract text from document")

        # Analyze using existing endpoint logic
        request = AnalysisRequest(document_text=document_text)
        return await analyze_contract(request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
@app.delete("/cleanup/{document_id}")
async def cleanup_analysis(document_id: str):
    """Clean up stored analysis data"""
    removed_items = []
    
    if document_id in analysis_cache:
        del analysis_cache[document_id]
        removed_items.append("result")
    
    if document_id in processing_status:
        del processing_status[document_id]
        removed_items.append("status")
    
    if not removed_items:
        raise HTTPException(status_code=404, detail="Document ID not found")
    
    return {"message": f"Cleaned up: {', '.join(removed_items)}", "document_id": document_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)