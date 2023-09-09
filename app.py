from flask import Flask
from flask import render_template,request
from src.BigMartSalesPrediction.utils.common import load_model


app=Flask(__name__)

@app.route('/')
def home():
   return render_template("home.html")

@app.route('/predict',methods=['GET','POST'])
def result():
   if request.method == 'POST':
      item_weight =float( request.form['item_weight'])
      item_fat_content=float(request.form['item_fat_content'])
      item_visibility=float(request.form['item_visibility'])
      item_type=float(request.form['item_type'])
      item_mrp=float(request.form['item_mrp'])
      outlet_identifier=float(request.form['outlet_identifier'])
      outlet_size=float(request.form['outlet_size'])
      outlet_location_type=float(request.form['outlet_location_type'])
      outlet_type=float(request.form['outlet_type'])
      model=load_model('artifacts/model_trainer/model.pkl')
      sc=load_model('artifacts/data_transformation/scale.pkl')
      input_data=sc.transform([[item_weight,item_fat_content,item_visibility,item_type,item_mrp,outlet_identifier,outlet_size,outlet_location_type,outlet_type]])
      result=model.predict(input_data)
      result=result[0]
      result=round(result,2)
      return render_template("predict.html",result=result)

if __name__ == '__main__':
    #app.run(debug = True ,host="0.0.0.0", port=8080,debug=True)
    app.run(debug = True ,host="0.0.0.0", port=8080)