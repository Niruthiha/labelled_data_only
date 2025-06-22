# Raw LLM Application Data Extractor

This tool extracts **labeled GitHub issues** from real-world LLM-powered application repositories for taxonomy and research purposes. It is designed to collect only issues that have human-applied labels, making the dataset ideal for supervised learning, taxonomy building, or analysis of LLM app development.

## Repository Search Criteria

Uses a wide range of targeted GitHub search queries to find repositories that are likely to be real-world LLM-powered applications.

**Searches cover:**
- Document chat, PDF analysis, semantic search, RAG, summarization, information extraction
- Customer support bots (Discord, Telegram, Slack, helpdesk)
- Code generation, code assistant, code review, unit test generation, code explanation
- Web apps using Streamlit, Gradio, Flask, FastAPI, Django, Next.js, React, Express.js
- AI agents, workflow/task/email automation
- Apps using OpenAI, Google AI/Gemini, Cohere, Anthropic Claude, Together AI, Replicate, HuggingFace
- Both Python and JavaScript/TypeScript apps
- **Star thresholds:** Most queries require `stars:>3` (some JS/TS queries use `stars:>5`).

## Filtering Out Libraries/Frameworks

Excludes repositories that are likely to be SDKs, libraries, or frameworks by:
- Checking if the repo belongs to official orgs (e.g., `openai/`, `anthropic/`, `google/`, `langchain-ai/`, `huggingface/`, etc.)
- Filtering by repo name, description, and topics for patterns like sdk, client, wrapper, library, framework, toolkit, package, etc.
- Excludes repos with topics like sdk, api, library, framework, package, etc.
- Excludes repos with descriptions like python client, api wrapper, sdk for, library for, etc.
- **Exception:** If the repo name contains app indicators (e.g., chatbot, assistant, app, web, dashboard, demo, project, etc.), it is considered an application.

## Application Structure Verification

Checks the file and directory structure of each repo for application indicators:
- **Files:** app.py, main.py, server.py, requirements.txt, package.json, index.js, bot.py, chatbot.py, etc.
- **Directories:** templates, static, src, components, api, routes, agents, bots, etc.
- **Negative signals:** Too many framework/library files (e.g., setup.py, pyproject.toml, __init__.py, etc.) reduce the app score.
- **Scores:** At least 2 app files or 2 app directories, and less than 3 framework files.

## LLM API Usage Detection

Scans main files (app.py, main.py, server.py, requirements.txt) for LLM API usage:
- Looks for keywords: openai, anthropic, cohere, together, replicate, google.generativeai, gemini, claude, gpt-3.5, gpt-4, OPENAI_API_KEY, etc.
- **Score:** At least 1 LLM usage keyword must be present.

## GitHub Issues Requirements

- The repository must have issues enabled.
- Must have at least 2 open or closed issues (excluding pull requests).

## LLM-Related Issues Analysis

- Samples up to 20 recent issues (excluding PRs).
- Counts issues mentioning LLM-related keywords (API key, rate limit, OpenAI, GPT, Claude, Gemini, prompt, model, embedding, RAG, chatbot, etc.).
- **Threshold:** At least 1 LLM-related issue or ≥10% of issues are LLM-related.

## Issue Labeling Analysis

- Calculates:
  - Percentage of issues with labels
  - Average labels per issue
  - Total number of unique labels in the repo
- **Lenient thresholds:** At least 20% of issues labeled, or at least 3 unique labels, or average ≥0.3 labels per issue.

## Final Suitability Decision

A repo is considered suitable if:
- It is a real application (not a library/framework)
- It uses LLM APIs in code
- It has issues enabled and at least 2 issues
- It has at least 1 LLM-related issue (or ≥10% LLM-related)
- It meets at least one labeling threshold, or has ≥30% LLM-related issues (prioritizing LLM usage over perfect labeling)
- Reason for inclusion/exclusion is logged and saved.

## Extraction and Output

For each suitable repo:
- Extracts up to 200 issues from the last year (excluding PRs)
- Saves issues with metadata (title, author, state, labels, body, comments, etc.) to a JSON file
- Saves search results and detailed analysis to JSON files in the output directory.

---

In summary:
The script is highly selective, focusing on real, user-facing LLM-powered applications with active and relevant issue discussions, not libraries or frameworks. It uses a combination of search queries, structural checks, code content analysis, and issue/label statistics to ensure only high-quality, relevant repositories are included. All thresholds, scores, and logic are explicitly coded and logged for transparency.

## Features

- **Discovers real LLM application repos** (not frameworks/libraries) using targeted GitHub search queries.
- **Filters for repositories with sufficient labeled issues** (at least 2 labeled issues).
- **Extracts only labeled issues** (ignores unlabeled issues and pull requests).
- **Saves each repository’s labeled issues** in a separate JSON file.
- **Provides summary files** for easy navigation and statistics.

## Output Files

- `search_results.json`: All repositories found by the search queries.
- `filtered_repositories.json`: Final list of repositories with enough labeled issues (these are the ones extracted).
- `[repo_name]_labeled_issues.json`: Labeled issues for each extracted repository.
- `extraction_summary.json`: Summary of the extraction process, including stats and a list of all output files.

## Usage

```bash
python3 extract5.py --token YOUR_GITHUB_TOKEN --output_dir ./your_output_dir
```

- `YOUR_GITHUB_TOKEN` with your GitHub personal access token.
- `--output_dir` is optional (default: `./raw_llm_data`).

## Requirements

- Python 3.7+
- `requests`, `tqdm`

Install dependencies:
```bash
pip install requests tqdm
```

## Notes

- Only issues with at least one label are extracted.
- Each `[repo_name]_labeled_issues.json` file contains full issue metadata and a sample of comments.
- Use `filtered_repositories.json` and `extraction_summary.json` to find which repositories were successfully extracted.

---

**Perfect for building labeled datasets for LLM application research and taxonomy!**
