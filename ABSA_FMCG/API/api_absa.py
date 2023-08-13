from flask import Flask
from flask_restful import Resource, Api, reqparse
import json
from API.tbsa import main as TBSA_infer
from API.acd_acsa import main as ACD_ACSA_infer


def init_api(task= "TBSA"):
    app = Flask(__name__)
    api = Api(app)

    input_data = {"data": []}
    if task == "TBSA":
        from API.tbsa import MODEL_CONFIG, get_tbsa_phobert_model
        models = get_tbsa_phobert_model(MODEL_CONFIG)

        class TBSA(Resource):
            def get(self):
                return input_data, 200

            def post(self):
                parser = reqparse.RequestParser()
                parser.add_argument("text")
                params = parser.parse_args()

                input_data["data"].append(params["text"])
                results = TBSA_infer(input_data["data"], models)
                input_data["data"].clear()
                return results, 201
        
        api.add_resource(TBSA, '/fmcg/tbsa/')
    elif task == "ACD_ACSA":
        from API.acd_acsa import MODEL_CONFIG, get_acd_acsa_phobert_model
        models = get_acd_acsa_phobert_model(MODEL_CONFIG)

        class ACD_ACSA(Resource):
            def get(self):
                return input_data, 200

            def post(self):
                parser = reqparse.RequestParser()
                parser.add_argument("text")
                params = parser.parse_args()

                input_data["data"].append(params["text"])
                results = ACD_ACSA_infer(input_data["data"], models)
                input_data["data"].clear()
                return results, 201

        api.add_resource(ACD_ACSA, '/fmcg/acd_acsa/')
    else:
        from API.tbsa import MODEL_CONFIG, get_tbsa_phobert_model
        tbsa_models = get_tbsa_phobert_model(MODEL_CONFIG)

        class TBSA(Resource):
            def get(self):
                return input_data, 200

            def post(self):
                parser = reqparse.RequestParser()
                parser.add_argument("text")
                params = parser.parse_args()

                input_data["data"].append(params["text"])
                results = TBSA_infer(input_data["data"], tbsa_models)
                input_data["data"].clear()
                return results, 201
        
        api.add_resource(TBSA, '/fmcg/tbsa/')

        from API.acd_acsa import MODEL_CONFIG, get_acd_acsa_phobert_model
        acd_acsa_models = get_acd_acsa_phobert_model(MODEL_CONFIG)

        class ACD_ACSA(Resource):
            def get(self):
                return input_data, 200

            def post(self):
                parser = reqparse.RequestParser()
                parser.add_argument("text")
                params = parser.parse_args()

                input_data["data"].append(params["text"])
                results = ACD_ACSA_infer(input_data["data"], acd_acsa_models)
                input_data["data"].clear()
                return results, 201

        api.add_resource(ACD_ACSA, '/fmcg/acd_acsa/')
    return app