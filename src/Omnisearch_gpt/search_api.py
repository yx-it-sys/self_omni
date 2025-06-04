from mimetypes import guess_type
import base64
import json
import os
import time
from io import BytesIO
import requests
from cragmm_search.search import UnifiedSearchPipeline
from crag_image_loader import ImageLoader
from PIL import Image

# initiate both image and web search API
## validation
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
    text_model_name="BAAI/bge-large-en-v1.5",
    web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
)

## public_test
search_pipeline = UnifiedSearchPipeline(
    image_model_name="openai/clip-vit-large-patch14-336",
    image_hf_dataset_id="crag-mm-2025/image-search-index-public-test",
    text_model_name="BAAI/bge-large-en-v1.5",
    web_hf_dataset_id="crag-mm-2025/web-search-index-public-test",
)

retry_attempt = 3


def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

def search_text_by_text(text_query):
    results = search_pipeline(text_query, k=2)
    assert results is not None, "No results found."
    return results

def search_image_by_image_url(image_url):
    image = ImageLoader(image_url).get_image()
    results = search_pipeline(image, k=2)
    assert results is not None, "No results found."
    return results

def parse_image_search_result_by_image(results, save_path, idx, conversation_num):
    images, entities = [], []
    for result in results:
        url = result.get("url", "")
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            fname = f"{idx}_{conversation_num}.png"
            out = os.path.join(save_path, fname)
            img.save(out, format="PNG")
            entity = result.get("entities", "")
            images.append(url, out)
            entities.append(entity)
        except Exception as e:
            print(f"Failed to save image {url}: {e}")
    return images, entities        

def fine_search(query, search_type, save_path, dataset_name, idx, conversation_num):
    if search_type == "text_search_text":
        results = search_text_by_text(query)
        texts = [item.get("page_snippet","") for item in results]
        return [], texts

    if search_type == "img_search_img":
        cache = os.path.join(save_path, dataset_name, f"image_search_res_{idx}.json")
        if os.path.exists(cache):
            with open(cache) as f:
                saved = json.load(f)
            imgs, txts = parse_image_search_result_by_image(saved, save_path, idx, conversation_num)
            if not txts:
                saved = search_image_by_image_url(query)
                imgs, txts = parse_image_search_result_by_image(saved, save_path, idx, conversation_num)
            print("Search results:", txts)
            return imgs, txts
        saved = search_image_by_image_url(query)
        return parse_image_search_result_by_image(saved, save_path, idx, conversation_num)
    return [], []
