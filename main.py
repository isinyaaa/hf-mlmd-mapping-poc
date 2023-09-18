import os

from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2


def setup_db():
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.filename_uri = os.path.join(os.getcwd(), "mlmddb")
    connection_config.sqlite.connection_mode = 3
    return metadata_store.MetadataStore(connection_config)


if __name__ == "__main__":
    store = setup_db()
