# IntelSync Example Workflow

This sample demonstrates how to orchestrate a simple, end-to-end multi-agent pipeline using the Agent Development Kit (ADK):

1. **WebScraperAgent** – Fetches live web articles from configured URLs  
2. **BigQueryLoaderAgent** – Writes raw JSON data into Google BigQuery  
3. **InsightGeneratorAgent** – Enriches stored data with sentiment analysis and key-entity extraction via Cloud Natural Language API  

## Usage

1. Navigate to this sample folder:
   ```bash
   cd contributing/samples/intel_sync_example
	```
2. Ensure your config/ folder is populated with valid YAML configsand GCP credentials.
3. Run the workflow:
	```bash
	python main.py
	```
_Created for the purposes of entering the Agent Development Kit Hackathon with Google Cloud. #adkhackathon_
