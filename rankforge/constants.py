TIER1_RETRIEVAL = {
    "faiss", "pinecone", "qdrant", "milvus", "weaviate", "chroma", "pgvector",
    "sentence-transformers", "vector search", "dense retrieval",
    "semantic search", "embedding", "ann", "approximate nearest neighbor",
    "colbert", "bi-encoder", "cross-encoder", "vector database", "vector db",
    "neural search", "hybrid search", "hnsw", "ivf index"
}

TIER2_NLP_IR = {
    "nlp", "bm25", "elasticsearch", "information retrieval", "text ranking",
    "tfidf", "tf-idf", "inverted index", "lucene", "solr", "text search",
    "ranking model", "learning to rank", "ltr", "ndcg", "mrr", "recall",
    "query understanding", "passage retrieval", "reranking", "sparse retrieval"
}

TIER2_RECSYS = {
    "recommendation system", "collaborative filtering", "matrix factorization",
    "two-tower", "retrieval ranking", "a/b testing", "experimentation",
    "personalization", "candidate generation", "ranking pipeline", "rrf",
    "reciprocal rank fusion", "hybrid retrieval"
}

TIER3_LLM = {
    "llm", "large language model", "rag", "retrieval augmented", "fine-tuning",
    "lora", "qlora", "bert", "transformers", "huggingface", "prompt engineering",
    "langchain", "openai api", "gpt", "instruction tuning", "embeddings api"
}

TIER3_MLOPS = {
    "mlflow", "weights and biases", "wandb", "bentoml", "triton", "mlops",
    "model serving", "feature store", "kubeflow", "model monitoring",
    "experiment tracking", "dagster", "airflow ml"
}

NEGATIVE_CV_SPEECH = {
    "yolo", "object detection", "image classification", "opencv",
    "resnet", "efficientnet", "speech recognition", "asr", "tts",
    "text-to-speech", "whisper asr", "wav2vec", "pose estimation",
    "face detection", "image segmentation", "action recognition", "gan",
    "diffusion model", "stable diffusion"
}

BIG_TECH_GLOBAL = {
    "google", "meta", "facebook", "amazon", "microsoft", "apple",
    "netflix", "linkedin", "salesforce", "adobe", "nvidia", "intel",
    "ibm", "oracle", "twitter", "uber", "airbnb", "stripe",
    "atlassian", "spotify", "snap", "pinterest", "paypal",
    "shopify", "zoom", "slack", "dropbox", "workday",
    "servicenow", "twilio", "datadog", "mongodb inc", "elastic",
    "confluent", "databricks", "snowflake inc", "palantir",
    "samsung", "tencent", "alibaba", "bytedance", "baidu"
}

WITCH_COMPANIES = {
    "tcs", "wipro", "infosys", "hcl", "cognizant", "accenture",
    "tech mahindra", "mphasis", "hexaware", "ltimindtree", "capgemini"
}

PRODUCT_STARTUPS_INDIA = {
    "razorpay", "zomato", "swiggy", "paytm", "cred", "flipkart",
    "meesho", "ola", "phonepe", "zepto", "blinkit", "nykaa",
    "policybazaar", "freshworks", "zoho", "dream11", "unacademy",
    "groww", "slice", "sarvam ai", "krutrim", "rephrase.ai",
    "yellow.ai", "haptik", "verloop.io", "aganitha", "mad street den",
    "glance", "inmobi", "locobuzz", "niramai", "pharmeasy", "vedantu",
    "genpact ai", "upgrad", "linkedin india", "sharechat",
    "lenskart", "urban company", "cleartax", "browserstack", "postman"
}

PREFERRED_CITIES = {
    "hyderabad", "pune", "mumbai", "delhi", "noida", "gurgaon", "gurugram",
    "bengaluru", "bangalore", "delhi ncr", "new delhi", "chennai"
}

JD_TITLES = {
    "ai engineer", "machine learning engineer", "nlp engineer",
    "search engineer", "applied scientist", "ml engineer",
    "recommendation systems engineer", "ai researcher",
    "applied ml engineer", "ai specialist", "ai research engineer",
    "senior ai engineer", "lead ai engineer", "staff ml engineer",
    "data scientist", "senior data scientist", "research engineer"
}

NOTICE_THRESHOLDS = [
    (0, 1.05), (15, 1.02), (30, 1.00), (60, 0.85),
    (90, 0.70), (120, 0.55), (999, 0.40)
]

ALL_JD_SKILLS = TIER1_RETRIEVAL | TIER2_NLP_IR | TIER2_RECSYS | TIER3_LLM

TIER_WEIGHTS = {
    "tier1": 4.0,
    "tier2_nlp": 3.0,
    "tier2_recsys": 3.0,
    "tier3_llm": 2.5,
    "tier3_mlops": 2.0
}
