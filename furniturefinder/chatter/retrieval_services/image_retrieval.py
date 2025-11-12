import asyncio
import string
import json
import logging
import base64
from datetime import datetime
from random import choices
import requests
from tabulate import tabulate
from PIL import Image
from configparser import ConfigParser

from chatter.models import ChatMessage
from organization.utils import log_execution_time
from external_services.weaviate_services import WeaviateHandler 


cfg = ConfigParser()
cfg.read("conf/application.ini")

IMG_DESC_MODEL = cfg["OpenAI"]["image_descriptor_model"]
DESC_MAX_TOKENS = int(cfg["OpenAI"]["image_description_max_tokens"])
DESC_TEMP = float(cfg["OpenAI"]["image_description_temperature"])
DESC_SEED = int(cfg["OpenAI"]["image_description_seed"])

RESIZE_SHAPE = (512, 768)
DEFAULT_EMBED_MODEL = cfg["OpenAI"]["text_embedding_model"]


class ImageMatcher:
    def __init__(
        self,
        base_prompt,
        resize_shape=RESIZE_SHAPE,
        save_path="data",
        desc_model=IMG_DESC_MODEL,
        logger=None,
        embed_model=DEFAULT_EMBED_MODEL,
    ):
        self._prompt = base_prompt
        self._resize_shape = resize_shape
        self._save_path = save_path
        self._desc_model = desc_model
        self._embed_model = embed_model

        self._shortlist_cap = 7
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._used_model = ""

        self._log = logger if isinstance(logger, logging.Logger) else logging.getLogger("image_matcher")

    def token_usage(self):
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens,
            "model": self._used_model,
        }

    def _download(self, chat_message: ChatMessage) -> str:
        ts = datetime.strftime(chat_message.timestamp, "%Y%m%d_%H_%M_%S_%f")
        rand_tag = "".join(choices(string.ascii_lowercase, k=5))
        resp = requests.get(chat_message.image_url)
        fname = f"{ts}_{rand_tag}.png"
        with open(f"{self._save_path}/{fname}", "wb") as out_file:
            out_file.write(resp.content)
        return fname

    def _resize(self, tag: str, fname: str, shape: tuple = None) -> str:
        shape = shape or self._resize_shape
        with Image.open(f"{self._save_path}/{fname}") as img:
            img.thumbnail(shape)
            resized = f"{self._save_path}/r{fname}"
            img.save(resized)
        return resized

    def _parse_openai_json(self, api_response) -> dict:
        parsed = {"name": "Target Furniture Item in Image", "url": "", "price": ""}
        data_str = api_response.choices[0].message.content
        for key, val in json.loads(data_str).items():
            parsed[key] = val
        return parsed

    def _describe_image(self, tag: str, client, img_file: str) -> dict:
        with open(img_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        img_url = f"data:image/jpeg;base64,{encoded}"

        try:
            res = client.chat.completions.create(
                model=self._desc_model,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": self._prompt}]},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_url}}]},
                ],
                response_format={"type": "json_object"},
                temperature=DESC_TEMP,
                max_tokens=DESC_MAX_TOKENS,
                seed=DESC_SEED,
            )
            self._prompt_tokens += res.usage.prompt_tokens
            self._completion_tokens += res.usage.completion_tokens
            self._total_tokens += res.usage.total_tokens
            self._used_model += res.model
        except Exception as err:
            self._log.critical(f"{tag} | openai_image_description_failed", exc_info=True)
            raise err

        return self._parse_openai_json(res)

    def _generate_embedding(self, tag: str, client, desc: dict) -> dict:
        unique_terms = list({v for v in desc.values() if v})
        joined_text = "|".join(unique_terms)
        try:
            response = client.embeddings.create(input=joined_text, model=self._embed_model)
        except Exception as err:
            self._log.critical(f"{tag} | openai_embedding_error", exc_info=True)
            raise err

        self._prompt_tokens += response.usage.prompt_tokens
        self._total_tokens += response.usage.total_tokens
        self._used_model += response.model

        return {
            "vector": response.data[0].embedding,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": response.usage.total_tokens,
            "model": response.model,
        }

    def _fetch_weaviate_results(self, chat_message: ChatMessage, desc: dict) -> list:
        try:
            db = WeaviateHandler()
            result = db.query_data(chat_message.organization.name, json.dumps(desc), True)
            return self._create_shortlist_table(result, desc)
        except Exception:
            self._log.critical("weaviate_query_failed", exc_info=True)
            raise

    def _create_shortlist_table(self, query_results: list, target: dict) -> list:
        shortlist = []
        seen_urls = set()
        for row in query_results:
            url = row.get("url")
            name = row.get("name")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            entry = {"name": name, "url": url}
            desc_field = row.get("furnitureDescriptors") or row.get("furniture_descriptor")
            if desc_field:
                for k, v in json.loads(desc_field).items():
                    entry[k] = v

            shortlist.append(entry)
            if len(shortlist) >= self._shortlist_cap:
                break

        return [target] + shortlist

    @log_execution_time
    def run_image_match(self, client, chat_message: ChatMessage) -> str:
        tag = f"[{chat_message.request_id}] | [{chat_message.organization.id}] | [{chat_message.session_id}]"
        desc = asyncio.run(self._prepare_and_describe(tag, client, chat_message, is_chat=True))
        embedding_vector = self._generate_embedding(tag, client, desc).get("vector")
        shortlist = self._fetch_weaviate_results(chat_message, desc)
        result_table = tabulate(shortlist, headers="keys", tablefmt="github")
        return result_table

    async def _prepare_and_describe(self, tag: str, client, chat_message: ChatMessage, is_chat: bool = False) -> dict:
        img_name = self._download(chat_message)
        resized = self._resize(tag, img_name, (365, 365) if not is_chat else None)
        return self._describe_image(tag, client, resized)
