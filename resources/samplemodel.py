from flask_restful import Resource, reqparse,request
from models.modelmapping import ModelMapping
import os
from resources.Model_Predict import predict
#########################
from multiprocessing import Pool
from threading import Thread


class SampleModel(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('apikey',
                        type=str,
                        required=True,
                        help="This akikey cannot be blank."
                        )
    parser.add_argument('username',
                        type=str,
                        required=True,
                        help="This username cannot be blank."
                        )

    def post(self):
        print("fdshgf")
        print('Headers: %s', request.headers)
        data = SampleModel.parser.parse_args()
        user= ModelMapping.find_by_apikey(data['apikey'])
        print(user)
        if(user==None):
            return "invalid key"
        row=user
        print ("Name: ",row.username, "Address:",row.apikey, "Email:",row.modelName)
        modelname=row.modelName
        
        #print('Body: %s', request.get_data())
        #print('Body: %s', request.files())
        app_root = os.path.dirname(os.path.abspath(__file__))
        #data = SampleModel.parser.parse_args()
        target = os.path.join(app_root, 'images')
        # target = os.path.join(APP_ROOT, 'static/')
        print("tar"+target)
        #return {"message": "image received","apicode":"100"}, 201
        
        for upload in request.files.getlist("image"):
            #print(upload)
            print("{} is the file name".format(upload.filename))
            filename = upload.filename
            destination = "/".join([target, filename])
            print ("Accept incoming file:", filename)
            print ("Save it to:", destination)
            upload.save(destination)
             
        print("predict calling")
        #thread = Thread(target = predict, args = (filename,modelname))
        #thread.start()
        #thread.join()
        print ("calling predict")           # Start a worker processes.
        pool = Pool(processes=1)
        ans=pool.apply_async(predict, [filename,modelname])
        pool.close()
        pool.join()
        #ans=predict(filename,modelname)
        print("predict called")
        # return send_from_directory("images", filename, as_attachment=True)
        #return render_template("complete.html", image_name=filename)
        print(ans)
        print(ans.successful())
        print(ans.ready())
        li=ans.get()
        print(li)
        print(type(li))
        lii=['aaa']
        lii=li.tolist()
        return {"message": "image received","apicode":"100","persons":lii}, 201
    
    
    def get(self):
        return ModelMapping.find_all()