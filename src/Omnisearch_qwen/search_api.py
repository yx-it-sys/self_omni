from mimetypes import guess_type
import base64
import json
import os
import time
from io import BytesIO

import requests
from PIL import Image
from serpapi import GoogleSearch

API_KEY = ""
retry_attempt = 3


def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def search_text_by_text(text):
    params = {
        "engine": "google",
        "q": text,
        "api_key": API_KEY,
        "num": 5,
    }
    for i in range(retry_attempt):
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            return results.get("organic_results", [])
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retry_attempt - 1:
                time.sleep(2)
            else:
                print("All retries failed.")
                return []


def search_image_by_text(text):
    params = {
        "engine": "google_images",
        "q": text,
        "api_key": API_KEY,
    }
    for i in range(retry_attempt):
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            images = results.get("images_results", [])
            return images[0] if images else {}
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < retry_attempt - 1:
                time.sleep(2)
            else:
                print("All retries failed.")
                return {}


def search_image_by_image_url(input_url):
    params = {
        "engine": "google_reverse_image",
        "image_url": input_url,
        "hl": "zh-CN",
        "gl": "CN",
        "api_key": API_KEY,
    }
    for i in range(retry_attempt):
        try:
            search = GoogleSearch(params)
            return search.get_dict()
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if "SSLError" in str(e):
                print("SSL error encountered.")
            elif "ConnectionError" in str(e):
                print("Network connection error.")
            if i < retry_attempt - 1:
                time.sleep(2)
            else:
                print("All retries failed. Returning empty result.")
                return {}


def parse_image_search_result_by_text(result, save_path, idx, conversation_num):
    images, texts = [], []
    url = result.get("thumbnail")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        fname = f"{idx}_{conversation_num}_{result.get('position','0')}.png"
        out = os.path.join(save_path, fname)
        img.save(out, format="PNG")
        images.append((url, out))
        texts.append(result.get("title", ""))
    except Exception as e:
        print(f"Failed to save thumbnail {url}: {e}")
    return images, texts


def parse_image_search_result_by_image(results, save_path, idx, conversation_num):
    images, texts = [], []
    kg = results.get("knowledge_graph", {})
    if "header_images" in kg:
        for item in kg["header_images"]:
            url = item.get("source")
            try:
                resp = requests.get(url)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
                fname = f"{idx}_{conversation_num}_header.png"
                out = os.path.join(save_path, fname)
                img.save(out, format="PNG")
                text = f"{kg.get('title','')}: {kg.get('description','')}"
                images.append((url, out))
                texts.append(text)
            except Exception as e:
                print(f"Failed to save header image {url}: {e}")
    elif "image_results" in results:
        for item in results["image_results"]:
            snippet = item.get("snippet")
            if snippet:
                texts.append(snippet)
    else:
        print("No 'knowledge_graph' or 'image_results' in response.")
    return images, texts


def fine_search(query, search_type, save_path, idx, conversation_num):
    if search_type == "text_search_text":
        results = search_text_by_text(query)
        texts = [item.get("title","") + item.get("snippet","") for item in results]
        return [], texts

    if search_type == "img_search_img":
        cache = os.path.join(save_path, f"image_search_res_{idx}.json")
        if os.path.exists(cache):
            with open(cache) as f:
                saved = json.load(f)
            imgs, txts = parse_image_search_result_by_image(saved, save_path, idx, conversation_num)
            if not txts:
                saved = search_image_by_image_url(query)
                imgs, txts = parse_image_search_result_by_image(saved, save_path, idx, conversation_num)
            return imgs, txts
        saved = search_image_by_image_url(query)
        return parse_image_search_result_by_image(saved, save_path, idx, conversation_num)

    if search_type == "text_search_img":
        result = search_image_by_text(query)
        return parse_image_search_result_by_text(result, save_path, idx, conversation_num)

    return [], []
