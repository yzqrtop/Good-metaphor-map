import json
from pathlib import Path
import argparse


def crawl_cxd(output_path: Path) -> None:
    """Call open-source API or local CXD database to crawl 14,032 idioms"""
    # Implement crawling logic here, currently using mock data
    # In actual projects, should call real CXD API or database
    idioms = []
    
    # Mock data - replace with real crawling in actual projects
    sample_idioms = [
        {"idiom": "井底之蛙", "explanation": "比喻见识短浅的人"},
        {"idiom": "守株待兔", "explanation": "比喻死守狭隘经验，不知变通"},
        {"idiom": "画蛇添足", "explanation": "比喻做了多余的事，非但无益，反而不合适"},
        {"idiom": "掩耳盗铃", "explanation": "比喻自己欺骗自己"},
        {"idiom": "亡羊补牢", "explanation": "比喻出了问题以后想办法补救，可以防止继续受损失"},
    ]
    
    idioms.extend(sample_idioms)
    
    # Write to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for idiom in idioms:
            json.dump(idiom, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Crawling completed, total {len(idioms)} idioms saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="Output file path")
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    crawl_cxd(args.out)