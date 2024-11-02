from __future__ import print_function
import numpy as np
from rwjson import json2np, writejson
import math
from argparse import ArgumentParser
import faiss
import yaml
from yaml.loader import SafeLoader
from distutils.log import debug
from flask import *  

app = Flask(__name__)  

 
def run_cpu(xq_json, json_out):
     
    xq = json2np(xq_json)
    d = xb.shape[1]

    index = faiss.IndexFlatL2(d)   # build the index
    index.train(xt)
    # print(index.is_trained)
    index.add(xb)                  # add vectors to the index
    # print(index.ntotal)

    D, I = index.search(xq, xb.shape[0]) 
    index = np.argwhere(D < max_distance)
    i_pre = math.inf
    
    # obj_id = math.inf
    with open(json_out, "w", encoding='utf-8') as outfile:
        outfile.write('[')
        length = len(index)
        dictionary = {
            "object_id": [],
            "results": []                    
        }
        for i in range(len(index)):
            result = {
                    "sample_id": [],
                    "distance":[]
                }
            
            result["sample_id"] = str(I[index[i][0],index[i][1]])
            result["distance"] = str(D[index[i][0],index[i][1]])
                
            if index[i][0] == i_pre:
                dictionary["object_id"] = str(index[i][0])
                dictionary["results"].append(result)
            elif not dictionary["object_id"]:
                dictionary["object_id"] = str(index[i][0])
                dictionary["results"].append(result)
            else:
                json_object = json.dumps(dictionary, sort_keys = False, indent=4,)   
                outfile.write(json_object)
                outfile.write(',')
                dictionary["object_id"]=[]
                dictionary["results"]=[]
                dictionary["object_id"] = str(index[i][0])
                dictionary["results"].append(result)
                
            i_pre = index[i][0]
        
            if(i == len(index)-1):
                json_object = json.dumps(dictionary, sort_keys = False, indent=4)   
                outfile.write(json_object)
                dictionary.clear()            
        outfile.write(']')   
     
    return json_out

def run_gpu(xq_json, json_out):  
    
    xq = json2np(xq_json)
    d = xt.shape[1]

    res = faiss.StandardGpuResources() 
    index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index

    # make it a flat GPU index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    # print(gpu_index_flat.is_trained)
    gpu_index_flat.train(xt)
    gpu_index_flat.add(xb)                  # add vectors to the index
    # print(gpu_index_flat.ntotal)

    D, I = gpu_index_flat.search(xq, 2048) 
    index = np.argwhere(D < max_distance)
    i_pre = math.inf
    # obj_id = math.inf
    with open(json_out, "w", encoding='utf-8') as outfile:
        outfile.write('[')
        length = len(index)
        dictionary = {
            "object_id": [],
            "results": []                    
        }
        for i in range(len(index)):
            result = {
                    "sample_id": [],
                    "distance":[]
                }
            
            result["sample_id"] = str(I[index[i][0],index[i][1]])
            result["distance"] = str(D[index[i][0],index[i][1]])
                
            if index[i][0] == i_pre:
                dictionary["object_id"] = str(index[i][0])
                dictionary["results"].append(result)
            elif not dictionary["object_id"]:
                dictionary["object_id"] = str(index[i][0])
                dictionary["results"].append(result)
            else:
                json_object = json.dumps(dictionary, sort_keys = False, indent=4,)   
                outfile.write(json_object)
                outfile.write(',')
                dictionary["object_id"]=[]
                dictionary["results"]=[]
                dictionary["object_id"] = str(index[i][0])
                dictionary["results"].append(result)
                
            i_pre = index[i][0]
        
            
                
            if(i == len(index)-1):
                json_object = json.dumps(dictionary, sort_keys = False, indent=4)   
                outfile.write(json_object)
                dictionary.clear()            
        outfile.write(']')   
     
    return json_out


@app.route('/')  
def main():  
    return render_template("index.html")  

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['filequery']
        f.save(f.filename) 
        name = 'result_' + f.filename 
        name_save = app.config['OUTPUT']  + 'result_' + f.filename
        if device == 'gpu':
            run_gpu(f.filename, name_save)
        else:
            run_cpu(f.filename, name_save)
        # send_from_directory(app.config["UPLOAD_FOLDER"], name)
        return render_template("Acknowledgement.html", name = name)  

# @app.route('/download/<name>', methods = ['GET','POST']) 
# def download_file(name):
#     return send_from_directory(app.config["OUTPUT"], name)

# app.add_url_rule(
#     "/download/<name>", endpoint="download_file", build_only=True
# )
    
if __name__ == '__main__':  
    argparser = ArgumentParser('-c config')
    args = argparser.add_argument('-c', '--config', default=None,)
    args = argparser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=SafeLoader)
    app.config['OUTPUT'] = configs['output_dir']
    xb =json2np(configs['sample_dataset'])
    xt =json2np(configs['train_dataset'])
    device = configs['divice']
    max_distance = configs['max_distance']
    app.run(host = configs['host'], port = config['port'], debug = False)