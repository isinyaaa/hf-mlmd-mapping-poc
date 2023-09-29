import os
import json
from pprint import pprint
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelCard
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2


def get_hf_api_handle():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    return HfApi(token=HF_TOKEN)


def get_mlmd_store() -> metadata_store.MetadataStore:
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.fake_database.SetInParent()
    return metadata_store.MetadataStore(connection_config)


class MLMDWrapper:
    _mlmd_map = {
        int: metadata_store_pb2.INT,
        str: metadata_store_pb2.STRING,
        float: metadata_store_pb2.DOUBLE,
        bool: metadata_store_pb2.BOOLEAN,
    }

    # _model_type_map = {
    #     "id": str,
    #     "description": str,
    #     "license": str,
    #     "sha": str,
    #     "tags": list[str],
    # }

    def __init__(self, store: metadata_store.MetadataStore, type_name: str):
        self._type_name = type_name
        self._store = store
        self._type = None

    @property
    def type(self) -> metadata_store_pb2.ArtifactType:
        return self._type

    def _create_artifact_type(self, type_map: Dict[str, Any]) -> int:
        try:
            _type = self._store.get_artifact_type(self._type_name).id
            print("Model type already exists")
        except:  # noqa: E722
            _type = metadata_store_pb2.ArtifactType()
            _type.name = self._type_name
            for k, v in type_map.items():
                _type.properties[k] = self._mlmd_map.get(type(v), metadata_store_pb2.STRING)
            _type = self._store.put_artifact_type(_type)
        self._type = _type
        return _type

    def register_artifact(self, type_map: Dict[str, Any]) -> int:
        """Register an artifact with the metadata store.

        Args:
            type_map (Dict[str, Any]): A dictionary of model metadata
        """
        artifact = metadata_store_pb2.Artifact()
        artifact.type_id = self._create_artifact_type(type_map)

        for k, v in type_map.items():
            if isinstance(v, str):
                artifact.properties[k].string_value = v
            elif isinstance(v, int):
                artifact.properties[k].int_value = v
            elif isinstance(v, float):
                artifact.properties[k].double_value = v
            elif isinstance(v, bool):
                artifact.properties[k].bool_value = v
            else:
                artifact.properties[k].string_value = json.dumps(v)
        return self._store.put_artifacts([artifact])[0]


if __name__ == "__main__":
    model = 'suno/bark-small'

    hf = get_hf_api_handle()
    # ds_info = hf.dataset_info(model)
    model_info = hf.model_info(model)
    model_card = ModelCard.load(model)
    # pprint(ds_info)
    # pprint(model_info)
    # pprint(model_card.data.to_dict())
    # pprint(model_card.text)
    # pprint(model_card.content)
    # hf.list_files_info()
    # hf.repo_info()
    # hf.space_info()

    model_info_md = {
        "id": model_info.modelId,
        "description": model_card.text,
        "license": model_card.data.license,
        "sha": model_info.sha,
        "tags": model_info.tags,
    }

    store = get_mlmd_store()

    hf_model_md = MLMDWrapper(store, "HuggingFace.model")
    model_id = hf_model_md.register_artifact(model_info_md)
    print("Model registered with id", model_id)
    pprint(store.get_artifacts_by_id([model_id]))
