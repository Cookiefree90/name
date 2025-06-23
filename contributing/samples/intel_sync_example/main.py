"""
IntelSync Example workflow for ADK Hackathon entry.
Demonstrates multi-agent orchestration using ADK Workflow API.
"""

from google_adk import Workflow
from agents.web_scraper_agent import WebScraperAgent
from agents.bigquery_loader_agent import BigQueryLoaderAgent
from agents.insight_generator_agent import InsightGeneratorAgent

def build_intelsync():
    wf = Workflow("intel_sync_example")  # valid identifier
    wf.add_agent(WebScraperAgent("scraper", "config/scraper_config.yaml"))
    wf.add_agent(BigQueryLoaderAgent("loader",  "config/bq_config.yaml"))
    wf.add_agent(InsightGeneratorAgent("insights","config/insights_config.yaml"))
    wf.set_sequence(["scraper", "loader", "insights"])
    return wf

if __name__ == "__main__":
    build_intelsync().run()
