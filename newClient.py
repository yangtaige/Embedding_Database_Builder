from langchain_community.embeddings import OllamaEmbeddings
import weaviate
from weaviate.embedded import EmbeddedOptions
import pandas as pd
from time import sleep
import hashlib
import uuid
import requests

def call_copilot_api(prompt:str, url:str, headers:dict, **kwargs)->str:
    """
    调用Copilot API，发送提示词并获取回复

    Args:
        prompt: 用户提示词
        **kwargs: 其他参数，如employeeName、orgId等

    Returns:
        API返回的回答字符串
    """

    # Log.logger.info('Requesting Started')
    # Log.logger.info(f'Sending request to Copilot api: {prompt}...')

    data = {
        "messages": [
            {
                "content": prompt,
                "role": "User"
            }
        ],


    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  
    # Log.logger.info(f'Return data {response.json()}')
    # Log.logger.info('Requesting Finished')
    return response.json()['data']['text']

def custom_uuid_to_uuidv4(custom_uuid:str)->uuid.UUID:
    """
    weaviate-client只支持UUIDv4格式的uuid，
    该函数是自定义 UUID 到 UUIDv4 的单射

    Args:
        custom_uuid: 自定义 UUID 字符串

    Returns:
        生成的 UUIDv4 字符串
    """

    # 计算 SHA-256 哈希值
    sha256 = hashlib.sha256()
    sha256.update(custom_uuid.encode('utf-8'))
    hash_value = sha256.hexdigest()  # 获取哈希值的十六进制字符串

    # 截取前 32 位并转换为 UUIDv4 格式
    uuid_bytes = bytes.fromhex(hash_value[:32])  # 将十六进制字符串转换为字节串
    uuid_obj = uuid.UUID(bytes=uuid_bytes, version=4)  # 创建 UUIDv4 对象
    return uuid_obj

def creating_class(class_name:str,
                   client:weaviate.Client=weaviate.Client(embedded_options=EmbeddedOptions()))\
        ->None:
    client.schema.delete_class(class_name)
    # logger.info(f'{class_name} in weaviate-client deleted.')
    # 创建类
    try:
        # logger.info("Begin to create class: "+ class_name)
        client.schema.create({"classes": [
            {
                "class": class_name,
                "properties": [
                    {"name": "name", "dataType": ["text"]},
                    {"name": "category", "dataType": ["text"], "index": True},
                    {"name": "dataSource", "dataType": ["text"]},
                    {"name": "description", "dataType": ["text"]},
                    {"name": "vector", "dataType": ["number[]"],
                     "vectorIndexConfig": {"name": "embedding_index", "dimensions": 3584, "distance": "cosine"}},
                    {"name": "uuid", "dataType": ["text"], 'index': True},
                ],
            },
        ]})
        # logger.info(f'Class: {class_name} in weaviate-client created.')
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        # logger.error(f"Error creating class: {e}")
        raise e

def process_data(row:pd.Series,
                 name:str,
                 embeddings:OllamaEmbeddings)\
        ->tuple[str,list[float], str]:
    # logger.info(f"Processing data --- {row[name]}......")
    description = ''.join(c for c in call_copilot_api(
    f"请使用中文以格式“名称：， 定义：”给出医学相关中文数据元：'{row[name]}'的定义，定义只要一句话，名称直接填写数据元'{row[name]}'本身，直接返回结果，不要包含多余内容。")
                          if c not in '[]{} "\'“”’‘').strip()
    # logger.info(f'Got response: {description}')
    vector = embeddings.embed_query(description)
    # logger.info(f"Data processing complete.")
    return description, vector, str(row['UUID'])

def init_data_loading(data_path:str):
    # logger.info("Begin to init data loading")
    df = pd.read_csv(data_path)
    # logger.info(f"Source data: {data_path}")
    return df


def load_data(data_path:str, class_name:str, name:str, category:str, dataSource:str,
              client:weaviate.Client=weaviate.Client(embedded_options=EmbeddedOptions()),
              embeddings:OllamaEmbeddings=OllamaEmbeddings(model="gemma2"))->None:
    df = init_data_loading(data_path)
    # logger.info("Begin to load data.")
    cnt = 0
    # 导入数据
    for index, row in df.iterrows():
        description, vector, uuid = process_data(row, embeddings)
        # logger.info(f"Loading data {cnt}...")
        client.data_object.create(
            {
                "properties": {
                    "name": row[name],
                    "category": row[category],
                    "dataSource": row[dataSource],
                    "description": description,
                    "vector": vector,
                    "uuid": uuid
                }
            },
            class_name,
            uuid = custom_uuid_to_uuidv4(uuid),
            vector = vector
        )
        # logger.info(f"Data {cnt} loaded.")
        cnt += 1
        sleep(3) # Avoiding rate limit
    # logger.info('Loading complete.')

if __name__ == "__main__":
    new_class = "example_class"
    creating_class(new_class) # Create a new class will delete the old one with the same name
    load_data("example/file/path", new_class, "name", "category", "dataSource")

