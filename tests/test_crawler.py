import pytest
from pathlib import Path
import json

from src.crawler.crawl_cxd import crawl_cxd


def test_crawl_cxd(tmp_path):
    """Test idiom crawling functionality"""
    output_path = tmp_path / "cxd_idioms.jsonl"
    crawl_cxd(output_path)
    
    # Check if output file exists
    assert output_path.exists()
    
    # Check output file content
    items = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            items.append(item)
    
    assert len(items) > 0
    assert all("idiom" in item for item in items)