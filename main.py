import os
import json
from pprint import pprint
from typing import Any, Dict, List, Optional
import inspect

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


def print_typemap(type_map: Dict[str, Any]):
    for k, v in type_map.items():
        if isinstance(v, list):
            print(f"{k}: List[{type(v[0])}]")
        else:
            print(f"{k}: {type(v)}")
        # print(f"{k}: {type(v)} = {v}")


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

    def _create_artifact_type(self, type_map: Dict[str, Any], new_mapping: Dict[str, str]) -> int:
        try:
            _type = self._store.get_artifact_type(self._type_name).id
            print("Model type already exists")
        except:  # noqa: E722
            _type = metadata_store_pb2.ArtifactType()
            _type.name = self._type_name
            for k, v in type_map.items():
                if k in new_mapping:
                    key = new_mapping[k]
                else:
                    key = k
                _type.properties[key] = self._mlmd_map.get(type(v), metadata_store_pb2.STRING)
            _type = self._store.put_artifact_type(_type)
        self._type = _type
        return _type

    def _repeated_scalar_to_str(self, repeated: List[Any]) -> str:
        return ','.join([str(x) for x in repeated])

    def register_artifact(self, type_map: Dict[str, Any]) -> int:
        """Register an artifact with the metadata store.

        Args:
            type_map (Dict[str, Any]): A dictionary of model metadata
        """
        artifact = metadata_store_pb2.Artifact()

        new_mapping: Dict[str, str] = dict()

        for k, v in type_map.items():
            if isinstance(v, str):
                artifact.properties[k].string_value = v
            # bool is a subclass of int, so check for bool first
            elif isinstance(v, bool):
                artifact.properties[k].bool_value = v
            elif isinstance(v, int):
                artifact.properties[k].int_value = v
            elif isinstance(v, float):
                artifact.properties[k].double_value = v
            elif v is None:
                artifact.properties[k].string_value = ""
            elif isinstance(v, list):
                if type(v[0]) in self._mlmd_map:
                    key = f"{k}.list"
                    new_mapping[k] = key
                    artifact.properties[key].string_value = self._repeated_scalar_to_str(v)
                elif inspect.isclass(v[0]):  # we're assuming homogeneous lists
                    key = f"{k}.key_list"
                    new_mapping[k] = key
                    key_list = list()
                    for item in v:
                        obj_id = MLMDWrapper(
                            self._store,
                            f"{self._type_name}.{item.__name__}"
                        ).register_artifact(item.__dict__)
                        key_list.append(str(obj_id))
                    artifact.properties[key].string_value = self._repeated_scalar_to_str(key_list)
                else:  # what if we're dealing with a list of lists?
                    # new_mapping[k] = ""
                    pass
            elif isinstance(v, dict):
                key = f"{k}.key"
                new_mapping[k] = key
                obj_id = MLMDWrapper(
                    self._store,
                    f"{self._type_name}.{k}"
                ).register_artifact(v)
                artifact.properties[key].string_value = str(obj_id)
            elif inspect.isclass(v):
                key = f"{k}.key"
                new_mapping[k] = key
                obj_id = MLMDWrapper(
                    self._store,
                    f"{self._type_name}.{v.__name__}"
                ).register_artifact(v.__dict__)
                artifact.properties[key].string_value = str(obj_id)
            else:
                print(v)
                # pprint(v)
                if inspect.isclass(v):
                    print("Class", v)
                raise NotImplementedError(f"Type {type(v)} (for key {k}) not supported")

        artifact.type_id = self._create_artifact_type(type_map, new_mapping)

        print("Artifact type map:")
        print_typemap(artifact.properties)
        print()
        return self._store.put_artifacts([artifact])[0]


if __name__ == "__main__":
    model = 'suno/bark-small'

    hf = get_hf_api_handle()
    model_info = hf.model_info(model)
    print("Model info type map:")
    print_typemap(model_info.__dict__)
    print(model_info)
    print()
    # pprint(model_info)
    # model_card = ModelCard.load(model)
    # pprint(model_card.data.to_dict())
    # pprint(model_card.text)
    # pprint(model_card.content)

    model_info_md = {
        "modelId": model_info.modelId,
        "sha": model_info.sha,
        "lastModified": model_info.lastModified,
        "tags": model_info.tags,
        "pipeline_tag": model_info.pipeline_tag,
        "private": model_info.private,
        "siblings": model_info.siblings,
        "author": model_info.author,
        "config": model_info.config,
        "securityStatus": model_info.securityStatus,
        "disabled": model_info.disabled,
        "gated": model_info.gated,
        "library_name": model_info.library_name,
        "cardData": model_info.cardData,
        "transformersInfo": model_info.transformersInfo,
        # "description": model_card.text,
        # "license": model_card.data.license,
    }

    # HF internal model info
    # _id: <class 'str'>
    # id: <class 'str'>
    # downloads: <class 'int'>
    # likes: <class 'int'>
    # model-index: <class 'NoneType'>
    # spaces: List[<class 'str'>]

    store = get_mlmd_store()

    hf_model_md = MLMDWrapper(store, "HuggingFace.model")
    model_id = hf_model_md.register_artifact(model_info_md)
    print("Model registered with id", model_id)
    pprint(store.get_artifacts_by_id([model_id]))
