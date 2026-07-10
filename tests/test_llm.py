from __future__ import annotations

import unittest

from deepresearch.config import Settings
from deepresearch.llm import LLMConfigurationError, build_llm_agent, select_provider


class LLMProviderTests(unittest.TestCase):
    def test_select_provider_from_model_prefix(self) -> None:
        cases = {
            ("", "openai/gpt-4.1-mini"): ("openai", "gpt-4.1-mini"),
            ("", "gemini/gemini-2.5-flash"): ("gemini", "gemini-2.5-flash"),
            ("", "claude/claude-sonnet-4-20250514"): (
                "anthropic",
                "claude-sonnet-4-20250514",
            ),
            ("anthropic", "anthropic/claude-3-5-sonnet-latest"): (
                "anthropic",
                "claude-3-5-sonnet-latest",
            ),
            ("openai", "gpt-4.1-mini"): ("openai", "gpt-4.1-mini"),
        }

        for args, expected in cases.items():
            with self.subTest(args=args):
                selected = select_provider(*args)
                self.assertEqual((selected.provider, selected.model), expected)

    def test_ollama_provider_does_not_require_api_key(self) -> None:
        agent = build_llm_agent(
            Settings(
                model_name="ollama/llama3.1",
                api_keys={},
                base_urls={},
                llm_timeout=1,
            )
        )

        self.assertEqual(agent.model_identifier, "ollama/llama3.1")

    def test_remote_provider_requires_api_key(self) -> None:
        with self.assertRaisesRegex(LLMConfigurationError, "OPENAI_API_KEY"):
            build_llm_agent(
                Settings(
                    model_name="openai/gpt-4.1-mini",
                    api_keys={},
                    base_urls={},
                )
            )


if __name__ == "__main__":
    unittest.main()
