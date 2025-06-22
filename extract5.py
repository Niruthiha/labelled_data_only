#!/usr/bin/env python3
"""
Raw LLM Application Data Extractor
Pure data extraction tool - no categorization, no filtering assumptions.
Extracts comprehensive issue data from LLM applications for taxonomy research.
"""

import os
import json
import time
import requests
import argparse
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RawLLMDataExtractor:
    def __init__(self, github_token: str, output_dir: str):
        if not github_token:
            raise ValueError("GitHub token cannot be empty.")
        self.token = github_token
        self.output_dir = output_dir
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        os.makedirs(self.output_dir, exist_ok=True)

        # Real LLM applications - much lower thresholds to capture authentic projects
        self.llm_app_searches = [
            # API key searches (real production apps, even small ones)
            'OPENAI_API_KEY language:python stars:>2 -tutorial -example',
            'ANTHROPIC_API_KEY language:python stars:>1',
            'GOOGLE_AI_KEY language:python stars:>1', 
            'COHERE_API_KEY language:python stars:>1',
            'REPLICATE_API_TOKEN language:python stars:>1',
            
            # Document processing (lower thresholds)
            'pdf chat openai language:python stars:>3 -tutorial',
            'document analysis openai language:python stars:>2',
            'RAG retrieval augmented language:python stars:>3',
            'semantic search openai language:python stars:>2',
            'pdf processing openai language:python stars:>2',
            
            # Chatbots and assistants (small teams often build these)
            'discord bot openai language:python stars:>2',
            'telegram bot gpt language:python stars:>2', 
            'slack bot openai language:python stars:>2',
            'chatbot openai language:python stars:>3 -tutorial',
            'ai assistant openai language:python stars:>2',
            
            # Code generation (often internal tools)
            'code generator openai language:python stars:>2 -tutorial',
            'code review openai language:python stars:>2',
            'unit test generator language:python stars:>2',
            'code assistant openai language:python stars:>2',
            
            # Business automation (small business tools)
            'email automation openai language:python stars:>2',
            'workflow automation llm language:python stars:>2',
            'content generation openai language:python stars:>3 -tutorial',
            'meeting transcription openai language:python stars:>2',
            
            # Web applications with LLM
            'streamlit openai language:python stars:>3 -tutorial',
            'flask openai language:python stars:>3 -example',
            'fastapi openai language:python stars:>3 -tutorial',
            'django openai language:python stars:>3',
            'gradio openai language:python stars:>2',
            
            # Multi-provider and switching (complex integration patterns)
            'model switching language:python stars:>1',
            'provider switching language:python stars:>1',
            'llm fallback language:python stars:>1',
            'openai anthropic language:python stars:>2',
            
            # Specialized applications
            'ai agent language:python stars:>3 -framework',
            'voice assistant openai language:python stars:>2',
            'image generation openai language:python stars:>2',
            
            # Cost and optimization focused (real business concerns)
            'token optimization openai language:python stars:>1',
            'api cost language:python stars:>1',
            'cost tracking openai language:python stars:>1',
            'usage monitoring llm language:python stars:>1',
            
            # Vector database integrations (often small projects)
            'pinecone openai language:python stars:>2',
            'chroma openai language:python stars:>2', 
            'qdrant openai language:python stars:>1',
            'weaviate openai language:python stars:>1',
            'faiss openai language:python stars:>2',
            
            # JavaScript/Node.js applications (lower bars)
            'openai nextjs javascript stars:>3 -tutorial',
            'openai react javascript stars:>3 -example',
            'openai express nodejs stars:>3 -tutorial',
            'openai vue javascript stars:>2',
            'OPENAI_API_KEY typescript stars:>3',
            
            # Other LLM providers (often smaller ecosystems)
            'together.ai language:python stars:>1',
            'huggingface transformers language:python stars:>3 -tutorial',
            'ollama language:python stars:>2',
            'local llm language:python stars:>2',
            
            # Application types that tend to be smaller but real
            'data analysis openai language:python stars:>2',
            'report generation openai language:python stars:>2',
            'text processing openai language:python stars:>2',
            'language translation openai language:python stars:>2',
            'sentiment analysis openai language:python stars:>2',
            
            # Broader searches with exclusions (capture variety)
            'openai language:python stars:>5 -tutorial -example -sdk -wrapper -langchain',
            'gpt language:python stars:>5 -tutorial -example -framework -langchain',
            'llm application language:python stars:>3 -tutorial -framework',
            'artificial intelligence language:python stars:>4 -tutorial -course -learning',
            
            # Even include some very small projects that might have unique patterns
            'openai api language:python stars:>1 created:>2023-01-01',
            'gpt integration language:python stars:>1 created:>2023-01-01',
            'llm integration language:python stars:>2 created:>2023-01-01'
        ]

    def _make_request(self, url: str, params: dict = None) -> dict:
        """Make API request with rate limiting and error handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                # Handle rate limiting
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining < 10:
                        reset_time = int(response.headers['X-RateLimit-Reset'])
                        sleep_duration = max(reset_time - time.time(), 60)
                        logger.warning(f"Rate limit low. Sleeping {sleep_duration:.0f}s")
                        time.sleep(sleep_duration)

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(15)
                else:
                    return None
        return None

    def search_llm_repositories(self) -> list:
        """Search for LLM application repositories."""
        logger.info("🔍 Searching for LLM application repositories...")
        
        found_repos = set()
        all_repos = []
        
        for i, query in enumerate(self.llm_app_searches, 1):
            logger.info(f"🔎 [{i}/{len(self.llm_app_searches)}] Query: {query}")
            
            search_url = f"{self.base_url}/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': 30
            }
            
            search_results = self._make_request(search_url, params)
            
            if not search_results or 'items' not in search_results:
                logger.warning(f"   ❌ No results")
                time.sleep(2)
                continue
            
            new_repos = 0
            for repo in search_results['items']:
                repo_name = repo['full_name']
                
                if repo_name in found_repos:
                    continue
                
                # Basic filtering - skip obvious frameworks/SDKs
                if self._is_likely_framework(repo):
                    continue
                
                found_repos.add(repo_name)
                all_repos.append({
                    'repo': repo_name,
                    'stars': repo['stargazers_count'],
                    'language': repo.get('language', ''),
                    'description': (repo.get('description') or '')[:200],
                    'url': repo['html_url'],
                    'topics': repo.get('topics', []),
                    'created_at': repo.get('created_at', ''),
                    'updated_at': repo.get('updated_at', ''),
                    'search_query': query
                })
                new_repos += 1
            
            logger.info(f"   ✅ Found {new_repos} new repos")
            time.sleep(3)
        
        # Sort by stars
        all_repos.sort(key=lambda x: x['stars'], reverse=True)
        
        logger.info(f"\n📊 Search completed: {len(all_repos)} unique repositories found")
        return all_repos

    def _is_likely_framework(self, repo: dict) -> bool:
        """Basic framework detection to avoid obvious SDKs/libraries."""
        repo_name = repo['full_name'].lower()
        description = (repo.get('description') or '').lower()
        
        # Official orgs
        framework_orgs = [
            'openai/', 'anthropic/', 'google/', 'microsoft/', 'cohere-ai/',
            'langchain-ai/', 'huggingface/', 'gradio-app/', 'streamlit/'
        ]
        
        for org in framework_orgs:
            if repo_name.startswith(org):
                return True
        
        # Framework indicators
        framework_keywords = [
            'python-sdk', '-sdk', 'api-client', '-client', '-wrapper', 
            'python-package', 'pip-package', 'library for', 'sdk for',
            'official client', 'client library'
        ]
        
        return any(keyword in repo_name or keyword in description 
                  for keyword in framework_keywords)

    def filter_repositories_with_labeled_issues(self, repo_list: list) -> list:
        """Filter repositories that have sufficient LABELED issues for taxonomy research."""
        logger.info("🏷️  Filtering repositories with labeled issues...")
        
        suitable_repos = []
        
        for i, repo_info in enumerate(repo_list, 1):
            repo_name = repo_info['repo']
            logger.info(f"📊 [{i}/{len(repo_list)}] Checking {repo_name}...")
            
            # Get repository data
            repo_data = self._make_request(f"{self.base_url}/repos/{repo_name}")
            if not repo_data:
                logger.info(f"   ❌ API error")
                continue
            
            # Check if issues are enabled
            if not repo_data.get('has_issues', False):
                logger.info(f"   ❌ Issues disabled")
                continue
            
            # Sample recent issues to check for labeled content
            labeled_count, total_sampled = self._count_labeled_issues(repo_name)
            
            if labeled_count < 2:  # Need at least 2 labeled issues
                logger.info(f"   ❌ Too few labeled issues ({labeled_count}/{total_sampled} sampled)")
                continue
            
            # Estimate total labeled issues
            if total_sampled > 0:
                labeling_rate = labeled_count / total_sampled
                total_issues_estimate = self._estimate_total_issues(repo_name)
                estimated_labeled = int(total_issues_estimate * labeling_rate)
            else:
                estimated_labeled = 0
            
            # Add detailed statistics to repo info
            repo_info.update({
                'total_issues_estimate': total_issues_estimate,
                'labeled_issues_estimate': estimated_labeled,
                'labeling_rate': round(labeling_rate * 100, 1),
                'sample_labeled': labeled_count,
                'sample_total': total_sampled,
                'has_issues': True,
                'repo_id': repo_data.get('id'),
                'default_branch': repo_data.get('default_branch', 'main'),
                'size': repo_data.get('size', 0)
            })
            
            suitable_repos.append(repo_info)
            logger.info(f"   ✅ Good labeling: {labeled_count}/{total_sampled} labeled ({labeling_rate*100:.1f}%), est. {estimated_labeled} total labeled")
            
            time.sleep(1)
        
        logger.info(f"\n📊 Filtering completed: {len(suitable_repos)} repositories with good labeling practices")
        return suitable_repos
    
    def _count_labeled_issues(self, repo_name: str) -> tuple:
        """Sample issues to count how many have labels."""
        issues_url = f"{self.base_url}/repos/{repo_name}/issues"
        params = {
            'state': 'all',
            'per_page': 50,  # Sample size
            'sort': 'updated',
            'direction': 'desc'
        }
        
        issues = self._make_request(issues_url, params)
        if not issues:
            return 0, 0
        
        # Filter out pull requests and count labeled issues
        actual_issues = [issue for issue in issues if 'pull_request' not in issue]
        labeled_issues = [issue for issue in actual_issues if issue.get('labels')]
        
        return len(labeled_issues), len(actual_issues)

    def _estimate_total_issues(self, repo_name: str) -> int:
        """Estimate total issue count by sampling."""
        issues_url = f"{self.base_url}/repos/{repo_name}/issues"
        params = {'state': 'all', 'per_page': 100, 'page': 1}
        
        issues = self._make_request(issues_url, params)
        if not issues:
            return 0
        
        # Filter out pull requests for actual issue count
        actual_issues = [issue for issue in issues if 'pull_request' not in issue]
        
        # If we got less than 100, this is likely the total
        if len(issues) < 100:
            return len(actual_issues)
        
        # Otherwise, estimate based on first page ratio
        issue_ratio = len(actual_issues) / len(issues) if issues else 0
        
        # Try to get second page to improve estimate
        params['page'] = 2
        issues_page2 = self._make_request(issues_url, params)
        
        if issues_page2 and len(issues_page2) > 0:
            # We have at least 100+ issues, estimate conservatively
            return max(len(actual_issues) * 2, 50)  # Conservative estimate
        else:
            return len(actual_issues)

    def extract_labeled_issues_only(self, repo_name: str, max_issues: int = 500) -> list:
        """Extract ONLY issues that have labels - focused on human-categorized data."""
        logger.info(f"🏷️  Extracting LABELED issues only from {repo_name}...")
        
        labeled_issues = []
        issues_url = f"{self.base_url}/repos/{repo_name}/issues"
        
        params = {
            'state': 'all',
            'per_page': 100,
            'sort': 'updated',
            'direction': 'desc'
        }
        
        page = 1
        while len(labeled_issues) < max_issues:
            params['page'] = page
            issues_batch = self._make_request(issues_url, params)
            
            if not issues_batch:
                break
            
            batch_labeled_count = 0
            for issue in issues_batch:
                if 'pull_request' in issue:
                    continue  # Skip pull requests
                
                # ONLY process issues with labels
                if not issue.get('labels'):
                    continue
                
                labels = [label['name'] for label in issue.get('labels', [])]
                
                # Extract comprehensive issue data
                issue_data = {
                    # Basic issue info
                    'issue_number': issue['number'],
                    'title': issue['title'],
                    'body': issue.get('body', ''),
                    'state': issue['state'],
                    'author': issue['user']['login'],
                    'author_type': issue['user']['type'],
                    
                    # Timestamps
                    'created_at': issue['created_at'],
                    'updated_at': issue['updated_at'],
                    'closed_at': issue.get('closed_at'),
                    
                    # Labels (the key data for your research!)
                    'labels': labels,
                    'label_count': len(labels),
                    'has_labels': True,  # Always true since we filter for this
                    
                    # Engagement metrics
                    'comments_count': issue.get('comments', 0),
                    'reactions': issue.get('reactions', {}),
                    
                    # Assignment and milestone
                    'assignees': [user['login'] for user in issue.get('assignees', [])],
                    'milestone': issue.get('milestone', {}).get('title') if issue.get('milestone') else None,
                    
                    # URLs for reference
                    'html_url': issue['html_url'],
                    'api_url': issue['url'],
                    
                    # Repository context
                    'repository': repo_name,
                    'extraction_date': datetime.now().isoformat()
                }
                
                # Get limited comments for context (only for labeled issues worth analyzing)
                if issue.get('comments', 0) > 0 and issue.get('comments', 0) <= 20:
                    comments = self._get_issue_comments(issue['comments_url'], limit=5)
                    issue_data['comments'] = comments
                else:
                    issue_data['comments'] = []
                
                labeled_issues.append(issue_data)
                batch_labeled_count += 1
                
                if len(labeled_issues) >= max_issues:
                    break
            
            logger.info(f"   Page {page}: +{batch_labeled_count} labeled issues (total: {len(labeled_issues)})")
            
            # Break if we got fewer than requested (last page) or no labeled issues in this batch
            if len(issues_batch) < 100 or batch_labeled_count == 0:
                break
            
            page += 1
            time.sleep(1)  # Rate limiting
        
        logger.info(f"✅ Extracted {len(labeled_issues)} LABELED issues from {repo_name}")
        return labeled_issues

    def _get_issue_comments(self, comments_url: str, limit: int = 5) -> list:
        """Get limited comments for additional context."""
        comments_data = self._make_request(comments_url)
        if not comments_data:
            return []
        
        return [
            {
                'author': comment['user']['login'],
                'body': comment['body'][:300],  # Truncate long comments
                'created_at': comment['created_at']
            }
            for comment in comments_data[:limit]
        ]

    def run_extraction(self):
        """Main extraction workflow - pure data collection."""
        logger.info("🚀 Starting Raw LLM Application Data Extraction...")
        
        # 1. Search for repositories
        repo_candidates = self.search_llm_repositories()
        
        if not repo_candidates:
            logger.error("❌ No repositories found!")
            return
        
        # Save search results
        search_file = os.path.join(self.output_dir, 'search_results.json')
        with open(search_file, 'w') as f:
            json.dump(repo_candidates, f, indent=2)
        logger.info(f"💾 Search results saved: {search_file}")
        
        # 2. Filter repositories with sufficient LABELED issues
        suitable_repos = self.filter_repositories_with_labeled_issues(repo_candidates)
        
        if not suitable_repos:
            logger.error("❌ No repositories with sufficient labeled issues!")
            return
        
        # Save filtered results
        filtered_file = os.path.join(self.output_dir, 'filtered_repositories.json')
        with open(filtered_file, 'w') as f:
            json.dump(suitable_repos, f, indent=2)
        logger.info(f"💾 Filtered repositories saved: {filtered_file}")
        
        # 3. Extract LABELED issues only from each repository
        logger.info(f"\n🏷️  Extracting LABELED issues from {len(suitable_repos)} repositories...")
        
        extraction_summary = {
            'extraction_date': datetime.now().isoformat(),
            'extraction_focus': 'LABELED_ISSUES_ONLY',
            'total_repositories': len(suitable_repos),
            'successful_extractions': 0,
            'total_labeled_issues_extracted': 0,
            'repositories': []
        }
        
        for i, repo_info in enumerate(suitable_repos, 1):
            repo_name = repo_info['repo']
            logger.info(f"\n📖 [{i}/{len(suitable_repos)}] Processing {repo_name}")
            
            # Extract LABELED issues only
            repo_issues = self.extract_labeled_issues_only(repo_name)
            
            if repo_issues:
                # Save individual repository data
                safe_filename = repo_name.replace('/', '_') + '_labeled_issues.json'
                repo_file = os.path.join(self.output_dir, safe_filename)
                
                repo_data = {
                    'repository': repo_name,
                    'repository_info': repo_info,
                    'extraction_date': datetime.now().isoformat(),
                    'extraction_type': 'LABELED_ISSUES_ONLY',
                    'total_labeled_issues': len(repo_issues),
                    'issues': repo_issues
                }
                
                with open(repo_file, 'w', encoding='utf-8') as f:
                    json.dump(repo_data, f, indent=2, ensure_ascii=False)
                
                # Update summary
                extraction_summary['successful_extractions'] += 1
                extraction_summary['total_labeled_issues_extracted'] += len(repo_issues)
                extraction_summary['repositories'].append({
                    'name': repo_name,
                    'labeled_issues_extracted': len(repo_issues),
                    'estimated_labeling_rate': repo_info.get('labeling_rate', 0),
                    'file': safe_filename
                })
                
                logger.info(f"💾 Saved {len(repo_issues)} LABELED issues to {safe_filename}")
            else:
                logger.info(f"⚠️  No labeled issues found in {repo_name}")
            
            time.sleep(2)  # Be respectful to API
        
        # Save extraction summary
        summary_file = os.path.join(self.output_dir, 'extraction_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(extraction_summary, f, indent=2)
        
        logger.info(f"\n🎉 LABELED ISSUE EXTRACTION COMPLETED!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   • Repositories processed: {extraction_summary['successful_extractions']}")
        logger.info(f"   • Total LABELED issues extracted: {extraction_summary['total_labeled_issues_extracted']}")
        logger.info(f"   • Average labeled issues per repo: {extraction_summary['total_labeled_issues_extracted']/max(extraction_summary['successful_extractions'],1):.1f}")
        logger.info(f"   • Data saved to: {self.output_dir}/")
        logger.info(f"   • Summary: {summary_file}")
        logger.info(f"   🏷️  ALL ISSUES HAVE LABELS - Perfect for taxonomy research!")

def main():
    parser = argparse.ArgumentParser(description='Raw LLM Application Data Extractor')
    parser.add_argument('--token', required=True, help='GitHub Personal Access Token')
    parser.add_argument('--output_dir', default='./raw_llm_data', help='Output directory')
    
    args = parser.parse_args()
    
    extractor = RawLLMDataExtractor(github_token=args.token, output_dir=args.output_dir)
    extractor.run_extraction()

if __name__ == "__main__":
    main()