from API.acd_acsa import *
import json
from API.api_absa import init_api
import argparse

if __name__ == "__main__":
    # with open("API/test.json", "r") as f:
    #     input_data = json.load(f)["data"]
    # results = main(input_data)
    # print(json.dumps(results, indent= 4, ensure_ascii=False))

    parser = argparse.ArgumentParser(description='ACD ACSA API')
    parser.add_argument('--ip', type=str, default= "172.26.33.201", 
                        help='Server ip')
    parser.add_argument('--port', type=int, default= 2504, 
                        help='Port')
    parser.add_argument("--debug", action="store_true")  
    args = parser.parse_args()

    app = init_api(task= "ACD_ACSA")
    app.run(host= args.ip, port= args.port, debug= args.debug)