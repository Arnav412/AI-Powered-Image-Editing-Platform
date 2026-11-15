import io
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.base import Caption, EditPlan, ProviderError
from src.ai.factory import available_providers, get_provider
from src.ai.openrouter_provider import OpenRouterProvider


def make_png(color=(30, 90, 200), size=(32, 20)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class TestOpenRouterProvider(unittest.TestCase):
    def test_provider_initialization(self):
        p_gem = OpenRouterProvider(api_key="sk-or-test", flavor="gemini")
        self.assertEqual(p_gem.name, "gemini")
        self.assertEqual(p_gem.vision_model, "google/gemini-2.5-flash")
        self.assertEqual(p_gem.image_model, "google/gemini-2.5-flash-image")

        p_oai = OpenRouterProvider(api_key="sk-or-test", flavor="openai")
        self.assertEqual(p_oai.name, "openai")
        self.assertEqual(p_oai.vision_model, "openai/gpt-4o-mini")
        self.assertEqual(p_oai.image_model, "openai/gpt-image-1")

    def test_factory_with_openrouter(self):
        avail = available_providers()
        self.assertIn("gemini", avail)
        self.assertIn("openai", avail)
        self.assertIn("openrouter", avail)

        provider_gem = get_provider("gemini")
        self.assertIsInstance(provider_gem, OpenRouterProvider)
        self.assertEqual(provider_gem.name, "gemini")

        provider_oai = get_provider("openai")
        self.assertIsInstance(provider_oai, OpenRouterProvider)
        self.assertEqual(provider_oai.name, "openai")

    @patch("requests.post")
    def test_edit_image_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="}]
        }
        mock_post.return_value = mock_resp

        p = OpenRouterProvider(api_key="sk-or-test", flavor="gemini")
        result = p.edit_image(make_png(), "change background")
        self.assertTrue(len(result) > 0)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://openrouter.ai/api/v1/images")
        self.assertIn("input_references", kwargs["json"])
        self.assertEqual(kwargs["json"]["model"], "google/gemini-2.5-flash-image")

    @patch("openai.OpenAI")
    def test_vision_text_success(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = '{"caption": "A sunset beach", "tags": ["sunset", "beach"]}'
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        p = OpenRouterProvider(api_key="sk-or-test", flavor="gemini")
        p.client = mock_client
        cap = p.caption(make_png())
        self.assertEqual(cap.caption, "A sunset beach")
        self.assertEqual(cap.tags, ["sunset", "beach"])


if __name__ == "__main__":
    unittest.main()
