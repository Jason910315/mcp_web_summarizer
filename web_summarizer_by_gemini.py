import datetime
import json
from pathlib import Path
from typing import Any
from bs4 import BeautifulSoup
from transformers import pipeline
from mcp.server.fastmcp import FastMCP
import asyncio
from crawl4ai import *
import re
from openai import OpenAI
import os
from dotenv import load_dotenv
import tiktoken
import google.generativeai as genai

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key = gemini_key)

gemini_model = genai.GenerativeModel("gemini-2.0-flash")

mcp = FastMCP("web_summarizer_by_gemini")

# 初始化 tokenizer 以便計算 tokens 數
encoding = tiktoken.encoding_for_model("gpt-4")

# 爬取網頁內容並轉為 markdown 格式
async def fetch_url_content(url: str) -> str:
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url = url, raw = True)
            return result.markdown
    except Exception as e:
        return f"爬取失敗: {str(e)}"

# 計算實際的token數量
def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

# 切割爬蟲結果，使其區塊不超過最大 tokens 限制
def split_text(text: str, max_tokens: int = 3000, overlap: int = 100) -> list:
    chunks = []
    sentences = re.split(r"(。|\n)", text)   # 正規表達式
    current_chunk = ""
    
    for sentence in sentences:
        test_chunk = current_chunk + sentence
        if count_tokens(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # 加上重疊部分
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + sentence
            else:
                # 如果單個句子就超過限制，強制添加
                chunks.append(sentence)
                current_chunk = ""
    
    # 最後一段
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# 用 gpt 模型對一段文字做總結
def summarize_with_gemini(text: str,model = "gemini-2.0-flash",temperature = 0.3) -> str:
    system_prompt = """請你閱讀使用者給予的由網站轉換而來的 Markdown 內容，並根據使用者提出的需求進行判斷與回應。
        請依照以下規則操作：
        1. 忽略與主文無關的部分（如：導覽列、頁面標題、作者資訊、版權聲明、廣告、頁尾、網站連結、其他不完整的項目列表或空段落）。
        2. 首先，辨識主文內容是否與使用者需求相關。
        3. 接著，針對符合需求的主文部分，依照使用者需求進行以下一種或多種整理任務：
        - 摘要：整理出文章主題、重點內容與結論。
        - 分析：歸納事件背景、問題核心、可能影響與作者立場。
        - 教學：若為技術或知識型文章，請條列步驟並補足教學說明。
        - 觀點整理：整理內文中的評論、立場或主觀見解。
        4. 主文的特徵通常是：連貫的自然語言敘述、具有背景說明、事件發展、技術教學或觀點分析等段落。
        5. 輸出可以是一段清晰且流暢的說明文章、條列式的重點整理等。
    """
    full_prompt = f"{system_prompt}\n\n{text}"
    response = gemini_model.generate_content(
        full_prompt,
        generation_config={
            "temperature": temperature
        }
    )
    return response.text

def summarize_text(text: str, user_prompt: str) -> str:
    # 計算實際token數量
    text_tokens = count_tokens(text)
    
    # 如果文字長度超過 3000 tokens，則需要分段摘要（預留空間給prompt）
    if text_tokens >= 3000:
        chunks = split_text(text, max_tokens=3000, overlap=100)
        partial_summarizes = []
        
        for chunk in chunks:
            summary_prompt = (
                f"使用者的摘要指令如下：{user_prompt}\n\n"
                "請截取以下 markdown 內容並判斷內容與使用者指令要求的相關程度進行摘要，以流暢的段落文字顯示，且不要加上開頭語氣:\n\n"
                f"{chunk}"
            )
            summary = summarize_with_gemini(text = summary_prompt, model = "gemini-2.0-flash", temperature = 0.3)
            partial_summarizes.append(summary)
     
        combined_summarize = "\n\n".join(partial_summarizes)

        # 總結每段摘要，並統整為一篇結構清晰、語氣一致、無重複的總結
        final_prompt = (
                f"使用者的摘要指令如下：{user_prompt}\n\n"
                "以下是根據原始長文分段產出的摘要，請根據使用者指令統整為一篇結構清晰、語氣一致、無重複的總結：\n\n"
                f"{combined_summarize}"
            )
        final_summary = summarize_with_gemini(text = final_prompt)
        return final_summary

    else:
        summary_prompt = (
                f"使用者的摘要指令如下：{user_prompt}\n\n"
                "請截取以下 markdown 內容並判斷內容與使用者指令要求的相關程度進行摘要，以流暢的段落文字顯示:\n\n"
                f"{text}"
        )
        return summarize_with_gemini(summary_prompt)

@mcp.tool()
async def summarize_web_by_gemini(url: str, instruction: str) -> str:
    """
    Fetch a webpage and use the gemini model tosummarize it based on user instruction.

    Args:
        url: website url
        instruction: instruction for summarization
    Returns:
        A summary of the webpage content based on the user instruction.
    """
    markdown = await fetch_url_content(url)
    summary = summarize_text(markdown, instruction)

    # 儲存資料
    base_dir = Path.cwd()
    parent_dir = base_dir.parent
    save_dir = os.path.join(parent_dir, "AI-code", "summary_results.json")
    
    record = {
        "url": url,
        "instruction": instruction,
        "markdown": markdown,
        "summary": summary,
    }

    with open(save_dir, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    
    return summary
    
if __name__ == "__main__":
    mcp.run(transport = "stdio")
