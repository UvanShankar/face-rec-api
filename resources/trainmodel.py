from flask_restful import Resource, reqparse,request
from models.modelmapping import ModelMapping
import os
import uuid
from multiprocessing import Pool
from resources.trainthedata import trainthedata


class TrainModel(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('modelname',
                        type=str,
                        required=True,
                        help="This akikey cannot be blank.",
                        location='form'
                        )
    parser.add_argument('username',
                        type=str,
                        required=True,
                        help="This username cannot be blank.",
                        location='form'
                        )

    def post(self):
        print("fdshgf")
        print('Headers: %s', request.headers)
        data = TrainModel.parser.parse_args()
        app_root = os.path.dirname(os.path.abspath(__file__))
        target = os.path.join(app_root, 'images')
        print("tar"+target)
        
        for upload in request.files.getlist("image"):
            #print(upload)
            print("{} is the file name".format(upload.filename))
            filename = upload.filename
            destination = "/".join([target, filename])
            print ("Accept incoming file:", filename)
            print ("Save it to:", destination)
            upload.save(destination)
            pool = Pool(processes=1)   
            print ("calling train the data")           # bStart a worker processes.
            pool.apply_async(trainthedata, [destination,data['modelname']+'model'])
            print(destination,data['modelname']+'model.sav')
            print("called the train the data")
            apikey=str(uuid.uuid4())
            user =ModelMapping(data['username'], apikey, data['modelname']+'model')
            user.save_to_db()
            break
        # return send_from_directory("images", filename, as_attachment=True)
        #return render_template("complete.html", image_name=filename)

        return {"message": "Data received","apikey":apikey,"description": "will work only from tomorrow 12am"}, 201
