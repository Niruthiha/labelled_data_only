# Raw LLM Application Data Extractor

This tool extracts **labeled GitHub issues** from real-world LLM-powered application repositories for taxonomy and research purposes. It is designed to collect only issues that have human-applied labels, making the dataset ideal for supervised learning, taxonomy building, or analysis of LLM app development.

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
