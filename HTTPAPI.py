from flask import Flask, request, jsonify
from langchain_community.embeddings import OllamaEmbeddings
from weaviate import Client
from weaviate.embedded import EmbeddedOptions
from newClient import custom_uuid_to_uuidv4, call_copilot_api
import numpy as np

def cosine_similarity(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return np.abs(np.dot(v1, v2) / (norm_v1 * norm_v2))



app = Flask(__name__)

client = Client(embedded_options=EmbeddedOptions())
embeddings = OllamaEmbeddings(model="gemma2")

@app.route('/queryDataElement', methods=['GET'])
def query_data_element(class_name:str):
    # logger.info(f'/queryDataElement get request(GET) {request}')
    data = request.json
    uuid = data['uuid']
    standard_name = data['standardName']
    data_element_name = data['dataElementName']
    query_vector = embeddings.embed_query(''.join(c for c in call_copilot_api(
        f"请使用中文以格式“名称：， 定义：”给出医学相关中文数据元：'{data_element_name}'的定义，定义只要一句话，名称直接填写数据元'{data_element_name}'本身，直接返回结果，不要包含多余内容。")
                                                  if c not in '[]{}() "\'“”’‘').strip())

    result = client.query.get(class_name, "properties{name, uuid, vector}") \
        .with_near_vector({'vector': query_vector}).with_limit(5).do()  # 设置返回结果的数量，建议不要少于5
    # 过滤结果：同时考虑相似度和分类
    filtered_result = [item["properties"] for item in result["data"]["Get"][class_name]]

    response_data= [{'dataElementName': item['name'],
                        'uuid': item['uuid'],
                        'similarity': cosine_similarity(item['vector'], query_vector)}
                       for item in filtered_result]


    if not uuid is None:
        uuid_list = (client.query.get(class_name, 'properties{uuid, name, vector}')
            .with_near_object({'id': f'{custom_uuid_to_uuidv4(uuid)}'}).with_limit(1).do())\
            ['data']['Get'][class_name]
        if uuid_list is None:
            pass
        else:
            uuid_dic = uuid_list[0]['properties']
            sim = cosine_similarity(uuid_dic['vector'], query_vector)
            obj = {'dataElementName': uuid_dic['name'],
                                     'uuid': uuid_dic['uuid'],
                                     'similarity': sim}
            if obj in response_data:
                response_data.remove(obj)
            response_data.insert(0, obj)
    # logger.info(f'/queryDataElement response:{response_data}')
    return jsonify(response_data)

#createDateElement(uuid,standardName,dataEelementName) response(isSuccess,dataEleCategory,dataElaDesc)
#入参（uuid，规范名，数据字段名）返回json参数（是否成功0成功，1失败；数据元分类，数据元描述中文）
@app.route('/createDataElement', methods=['POST'])
def create_data_element():
    # logger.info(f'/createDataElement post request(POST) {request}')
    try:
        data = request.json
        uuid = data['uuid']
        standard_name = data['standardName']
        data_element_name = data['dataElementName']

        cat = ''.join(c for c in call_copilot_api(
            f'数据元"{data_element_name}"最匹配以下哪个数据元分类："..."，\
直接返回分类名称，不要包含多余字符')
            if c not in '“”‘’"\' ')
        description = ''.join(c for c in call_copilot_api(
            f"请使用中文以格式“名称：， 定义：”给出医学相关中文数据元：'{data_element_name}'的定义，定义只要一句话，名称直接填写数据元'{data_element_name}'本身，直接返回结果，不要包含多余内容。")
                                                      if c not in '[]{}() "\'“”’‘').strip()
        query_vector = embeddings.embed_query(description)

        client.data_object.create({
                    "properties": {
                        "name": data_element_name,
                        "category": cat,
                        "dataSource": standard_name,
                        "description": description,
                        "vector": query_vector,
                        "uuid": uuid
                    }
                },
                "WingptBusinessDataElement",
                uuid = custom_uuid_to_uuidv4(uuid),
                vector = query_vector
        )
        # logger.info(f'/createDataElement response:isSuccess:0, dataEleCategory:{cat}, dataEleDesc:{description}')
        return jsonify({'isSuccess': 0, 'dataEleCategory': cat, 'dataEleDesc': description})
    except Exception as e:
        # logger.error(f'/createDataElement error:{e}')
        return jsonify({'isSuccess': 1, 'dataEleCategory': None, 'dataEleDesc': None, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

    # 下面是使用示例

    # url_get = 'http://127.0.0.1:5000/queryDataElement'
    # url_post = 'http://127.0.0.1:5000/createDataElement'
    # response1 = requests.post(url_post,
    #                           json={'uuid': '', 'standardName': 'a', 'dataElementName': '', })
    # response2 = requests.post(url_post,
    #                           json={'uuid': '', 'standardName': 'a', 'dataElementName': '', })
    # response0 = requests.get(url_get, json={'uuid': '', 'standardName': 'a',
    #                                         'dataElementName': '', })
    # print(response1.json())
    # print(response2.json())
    # print(response0.json())