"""Model manager for loading and caching chat models."""

from typing import Dict, Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from src.log_util import logger


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """从完整指定的名称加载聊天模型。

    Args:
        fully_specified_name (str): 格式为 'provider/model' 的字符串。
    """
    if "/" in fully_specified_name:
        provider, model_name = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model_name = fully_specified_name

    model = init_chat_model(model_name, model_provider=provider)
    return model


class ModelManager:
    """A singleton class to manage and cache chat models."""

    _instance: Optional["ModelManager"] = None
    _models: Dict[str, BaseChatModel] = {}

    def __new__(cls) -> "ModelManager":
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.debug("ModelManager instance created")
        return cls._instance

    def get_model(self, fully_specified_name: str) -> BaseChatModel:
        """Get a chat model, loading it if necessary.

        Args:
            fully_specified_name: String in the format 'provider/model'.

        Returns:
            The requested chat model.
        """
        if fully_specified_name not in self._models:
            logger.info(f"Loading model: {fully_specified_name}")
            self._models[fully_specified_name] = load_chat_model(fully_specified_name)
        else:
            logger.debug(f"Using cached model: {fully_specified_name}")

        return self._models[fully_specified_name]

    def clear_cache(self) -> None:
        """Clear the model cache."""
        logger.info("Clearing model cache")
        self._models.clear()


model_manager = ModelManager()
