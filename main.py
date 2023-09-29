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


class HuggingFaceModelMD:
    _mlmd_map = {
        int: metadata_store_pb2.INT,
        str: metadata_store_pb2.STRING,
        float: metadata_store_pb2.DOUBLE,
        bool: metadata_store_pb2.BOOLEAN,
    }

    _model_type_name = "HuggingFace.model"
    _model_type_map = {
        "id": str,
        "description": str,
        "license": str,
        "sha": str,
        "tags": list[str],
    }

    def __init__(self):
        connection_config = metadata_store_pb2.ConnectionConfig()
        connection_config.fake_database.SetInParent()
        self._store = metadata_store.MetadataStore(connection_config)
        # self._dataset_type_name = "HuggingFace.dataset"
        self._model_type = None
        # self._dataset_type = None

    @property
    def model_type(self) -> metadata_store_pb2.ArtifactType:
        return self._model_type

    def _create_artifact_type(self, type_map: Dict[str, Any]) -> int:
        try:
            model_type = self._store.get_artifact_type(self._model_type_name).id
            print("Model type already exists")
        except:  # noqa: E722
            model_type = metadata_store_pb2.ArtifactType()
            model_type.name = self._model_type_name
            for k, v in type_map.items():
                model_type.properties[k] = self._mlmd_map.get(type(v), metadata_store_pb2.STRING)
            model_type = self._store.put_artifact_type(model_type)
        self._model_type = model_type
        return model_type

    def register_model(self, type_map: Dict[str, Any]) -> int:
        """Register a model with the metadata store.

        Args:
            type_map (Dict[str, Any]): A dictionary of model metadata
        """
        artifact = metadata_store_pb2.Artifact()
        artifact.type_id = self._create_artifact_type(type_map)

        for k, v in type_map.items():
            # if k in self._model_type_map:
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
            # else:
            #     raise KeyError(f"Unknown property {k}")
        return self._store.put_artifacts([artifact])[0]

    # @property
    # def dataset_type(self) -> metadata_store_pb2.ArtifactType:
    #     if self._dataset_type is None:
    #         try:
    #             self._dataset_type = self._store.get_artifact_type(self._dataset_type_name)
    #         except:  # noqa: E722
    #             dataset_type = metadata_store_pb2.ArtifactType()
    #             dataset_type.name = self._dataset_type_name
    #             dataset_type.properties["name"] = metadata_store_pb2.STRING
    #             dataset_type.properties["description"] = metadata_store_pb2.STRING
    #             self._store.put_artifact_type(dataset_type)

    #     return self._dataset_type

    # def register_dataset(self, name: str, description: str) -> int:
    #     artifact = metadata_store_pb2.Artifact()
    #     artifact.properties["name"].string_value = name
    #     artifact.properties["description"].string_value = description
    #     artifact.type_id = self.dataset_type.id
    #     return self._store.put_artifact(artifact)


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

    hf_model_md = HuggingFaceModelMD()
    model_id = hf_model_md.register_model(model_info_md)
    print("Model registered with id", model_id)
    pprint(hf_model_md._store.get_artifacts_by_id([model_id]))
