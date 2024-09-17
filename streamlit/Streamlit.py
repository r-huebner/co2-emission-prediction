import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import zipfile
from sklearn.metrics import classification_report, f1_score, accuracy_score, mean_squared_error, r2_score, confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from lime.lime_tabular import LimeTabularExplainer
from keras.models import load_model

css = """
<style>
[data-testid="stSidebar"] {
    background-color: #b2e3a8; 
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Get DataFrames
@st.cache_data
def get_df_from_zip(archiv, filename):
  zipped_file = zipfile.ZipFile(archiv)
  csv_file = zipped_file.open(filename)
  df = pd.read_csv(csv_file)
  return df

# Jörg
@st.cache_data
def get_df_from_hdf(filename):
  df = pd.read_hdf(filename, key='df')
  return df

# Get test set
@st.cache_data
def get_test_data(filename):
    df =pd.read_csv(filename, index_col=0)
    return df

# datasets
df_info = get_test_data('../data/df_info.csv')
X_test = get_test_data('../data/X_test.csv')
y_test_reg = get_test_data('../data/y_test_reg.csv')
y_test_clf = get_test_data('../data/y_test_clf.csv')
df_filtered01 = get_df_from_zip('../data/data_filtered01.zip','data_filtered01.csv')
df_filtered02 = get_df_from_zip('../data/data_filtered02.zip','data_filtered02.csv')
df_filtered = pd.concat([df_filtered01,df_filtered02], axis=0)
df_preprocessed = get_df_from_hdf('../data/data_preprocessed_targets.h5')

# Model name lists
classification_models = ['Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree','XGBoost', 'XGBoost optimized', 'Neural Network', 'Neural Network optimized']
regression_models = ['Linear Regression', 'Elastic Net', 'Decision Tree', 'XGBoost', 'XGBoost optimized', 'Neural Network', 'Neural Network optimized']

# Get models
@st.cache_data
def load_classifiers():
    models = {}
    models['Decision Tree'] = joblib.load("../models/Classification/DecisionTreeClassifier.sav")
    models['Logistic Regression'] = joblib.load("../models/Classification/LogisticRegression.sav")
    models['XGBoost'] = joblib.load("../models/Classification/XGBClassifier.sav")
    clf_grid = joblib.load("../models/Optimization/XGBClassifier optimized.sav")
    models['XGBoost optimized'] = clf_grid.best_estimator_
    models['Neural Network'] = joblib.load("../models/NN/NN Classification.sav")
    models['Neural Network optimized'] = load_model("../models/Optimization/best_model_clf.keras")
    return models

@st.cache_data
def load_regressors():
    models = {}
    models['Decision Tree'] = joblib.load("../models/Regression/DecisionTreeRegressor.sav")
    models['Elastic Net'] = joblib.load("../models/Regression/ElasticNet.sav")
    models['Linear Regression'] = joblib.load("../models/Regression/LinearRegression.sav")
    models['XGBoost'] = joblib.load("../models/Regression/XGBRegressor.sav")
    reg_grid = joblib.load("../models/Optimization/XGBRegressor optimized.sav")
    models['XGBoost optimized'] = reg_grid.best_estimator_
    models['Neural Network'] = joblib.load("../models/NN/NN Regression.sav")
    models['Neural Network optimized'] = load_model("../models/Optimization/best_model_reg.keras")
    return models

# Load models
classifiers = load_classifiers()
regressors = load_regressors()

# Cache predictions
@st.cache_data
def predict_clf(model_list): 
    pred = {}
    for model_name in model_list:
        if model_name == "K-Nearest Neighbors":
            continue
        model = classifiers[model_name]
        pred[model_name] = model.predict(X_test)
        if "Neural" in model_name:
            pred[model_name] =  np.argmax(pred[model_name], axis=1)
    return pred
@st.cache_data
def predict_reg(model_list): 
    pred = {}
    for model_name in model_list:
        model = regressors[model_name]
        pred[model_name] = model.predict(X_test)
    return pred

# Get predictions
get_predictions_reg = predict_reg(regression_models)
get_predictions_clf = predict_clf(classification_models)

print("predicitions")
print(get_predictions_reg)

# Get feature importance figure
def featimp_fig(featimp_score):
    col_names = X_test.columns
    df_featimp = pd.DataFrame(featimp_score,index=col_names[:len(featimp_score)],columns=["importance"])
    df_featimp = df_featimp.sort_values(by="importance")
    df_featimp_10 = df_featimp.tail(10)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.barh(range(len(df_featimp_10)), df_featimp_10['importance'], tick_label=df_featimp_10.index)
    ax.set_xlabel('Feature Importance Percentage')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance')
    fig.tight_layout()
    return fig

# Initialize the explainer
def initialize_explainer():
    return LimeTabularExplainer(X_test.values, mode='classification', feature_names=X_test.columns, verbose=True, discretize_continuous=True)

explainer = initialize_explainer()

# Cache summary DataFrames
@st.cache_data
def get_summary_clf():
    df_summary_clf = pd.DataFrame(columns=['Model', 'Accuracy', 'F1 Score'])
    for model_name in classification_models:
        if  model_name =="K-Nearest Neighbors":
            new_df = pd.DataFrame([{'Model': "K-Nearest Neighbors", 'Accuracy': 0.9825, 'F1 Score': 0.9825}])
        else:
            y_test_pred = get_predictions_clf[model_name]
            accuracy = accuracy_score(y_test_clf, y_test_pred)
            f1 = f1_score(y_test_clf, y_test_pred, average='weighted')
            new_df = pd.DataFrame([{'Model': model_name, 'Accuracy': accuracy, 'F1 Score': f1}])
        
        df_summary_clf = pd.concat([df_summary_clf, new_df], ignore_index=True)
    return df_summary_clf

@st.cache_data
def get_summary_reg():
    df_summary_reg = pd.DataFrame(columns=['Model', 'R2', 'RMSE'])
    for model_name in regression_models:
        y_test_pred = get_predictions_reg[model_name]
        accuracy = r2_score(y_test_reg, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred))
        new_df = pd.DataFrame([{'Model': model_name, 'R2': accuracy, 'RMSE': round(rmse, 4)}])
        df_summary_reg = pd.concat([df_summary_reg, new_df], ignore_index=True)
    return df_summary_reg

# Label Image
def get_label_image(class_val):
    image_path=""
    if class_val == 0:
        image_path="images/A.png"
    elif class_val == 1:
        image_path="images/B.png"
    elif class_val == 2:
        image_path="images/C.png"
    elif class_val == 3:
        image_path="images/D.png"
    elif class_val == 4:
        image_path="images/E.png"
    elif class_val == 5:
        image_path="images/F.png"
    elif class_val == 6:
        image_path="images/G.png"
    return image_path

## Streamlit

st.title("CO2 emissions by vehicles")
st.sidebar.title("Table of contents")

pages=["Home", "Data Exploration", "Preprocessing", "Modeling", "Interpretation"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0] : 
    st.write("""
        <h5 style="text-align: right; color:green">
            Project by Roberta Hübner, Mehdi Rahmati and Jörg Grünklee
        </h5>
        """, unsafe_allow_html=True)
    st.write("## Home")
    col1, col2 = st.columns([2,3])
    col1.write("")
    col1.write("")
    col1.write("")
    text = """
           We aim to identify the vehicles with the most CO2 emissions and find the cars contributing the most to air pollution.
        """
    col1.markdown(text)
    col2.image("images/cars_emissions.jpg")
    col2.write("""
        <div style="font-size: smaller; color: grey; text-align: right">
            © by vectorjuice on Freepik
        </div>
        """, unsafe_allow_html=True)
    
if page == pages[1] :
    st.header("Data Exploration")
    
    subpages=["Infos about data source", "Value counts","Visualization"]
    subpage=st.sidebar.radio("Choose a theme:", subpages)

    # Infos about data source
    if subpage==subpages[0]:
        st.write("We used the dataset “Monitoring of CO2 emissions from passenger cars” of 2021, distributed by the European Environment Agency." \
                "The data source has over 4,2 Million registered vehicles with 39 features for each vehicle.")

        selection=st.selectbox("Select column-type:",("none","100% NaN","One Value","Uniques","Country & Registration","Old definition","Only electric","Identication no."))
        if selection=="none":
            cols=[]
        elif selection=="100% NaN":
            cols=["MMS","Ernedc (g/km)","De","Vf"]
        elif selection=="One Value":
            cols=["r","Status","year"]
        elif selection=="Uniques":
            cols=["ID","Unnamed: 0"]
        elif selection=="Country & Registration":
            cols=["Date of registration","Country","year"]
        elif selection=="Old definition":
            cols=["Enedc (g/km)"]
        elif selection=="Only electric":
            cols=["z (Wh/km)" , "Electric range (km)"]
        elif selection=="Identication no.":
            cols=["VFN" , "Tan"]

        df_info_styled = df_info.style.applymap(lambda _: "background-color: CornflowerBlue;", subset=(cols, slice(None)))
        st.dataframe(df_info_styled)
    
    # Value counts
    if subpage==subpages[1]:
        selection=st.selectbox("Select a column:",df_filtered.select_dtypes('object').columns)
        st.write("Unique values:",df_filtered[selection].nunique())
        st.write(df_filtered[selection].value_counts())
    
    # For later use
    if subpage==subpages=="":
        with st.expander("Heatmap of numbers:"):
            st.write(' ')
            fig = plt.figure()
            sns.heatmap(df_filtered.drop(nan_cols+unique_cols+one_val_cols+c_r_cols+old_cols+electric_cols+ident_cols,axis=1).select_dtypes(include='number').corr(),center=0,annot=True, fmt='.1f')
            st.pyplot(fig)
    # Visualization
    if subpage==subpages[2]:
        selections = ["CO2 emissions of cars with different fuel types",
                      "Relationship between vehicles mass and their CO2 emissions"]
        selection = st.selectbox('Select a visualization',selections)

        if selection==selections[0]:
            st.image("images/emission of fuel-type.png")
            st.write("This graph shows that the emissions for cars using electric and hydrogen are set to zero, most likely due to the difficulty of calculating emission values for electricity usage and hydrogen manufacturing. The highest emission values are seen for vehicles using petrol, followed by diesel. The average emissions are around 100-150 grams of CO2 per kilometer, whereas hybrid cars have an average emission of around 40 grams of CO2 per kilometer. From the distribution of the data points, especially for petrol, we can see that there is a wide range of emission values with a substantial amount of cars with emissions up to 400 grams per kilometer with some outliers even going up to around 550 grams per kilometer. In summary, petrol and diesel, the most widely used fuels, are the fuels with the highest emissions overall.")
        
        if selection==selections[1]:
            st.image("images/vehicle mass emission.png")
            st.write("This graph shows that the emissions increase with the mass of the car in a linear fashion. For diesel fuel, we can also see that there is not one mass favored, as the distribution is quite homogenous and does not aggregate or dissipate at around a specific mass. For petrol, the emissions seem to be more unstable for higher mass vehicles, with fewer vehicles with a mass over 2000 kg compared to diesel. It seems that the dependency of hybrid cars emissions on vehicle mass is happening with a lower slope while for other cars it happens with a higher slope. The hybrid types have much less emissions, most likely due to the calculations that use weighted values. Cars running on gas (LPG, NG) are only seen weighing between 1200 and 1600 kg, with no exceptions. This could be either due to technological limitations or the fact that construction is not cost-effective. Overall we can see that diesel is the fuel used for light to heavy vehicles, with heavier vehicles having higher emissions, closely followed by petrol.")
    
if page == pages[2] : 
    st.write("## Preprocessing")

    subpages=["Data correlation", "Missing values","Renaming","One hot-encoding","Class distribution and Emission labels"]
    subpage=st.sidebar.radio("Choose a theme:", subpages)

    if subpage==subpages[0]:
        st.write("#### Data correlation:")
        st.write("###### Numerical features with Pearson method:")
        st.image("images/corr_num.png")
        st.write("###### Categorical features with CramersV method:")
        st.image("images/corr_cat.png")
        st.write("For data preprocessing, we have already identified the 16 useless features and removed them from the data source. In order to be able to make decisions about deleting further columns, we look at which features correlate particularly strongly with each other. A heat map with the correlated values between all features gives us a good overview. For the numerical values, we use the Pearson method. For the categorical features, the Cramers V method is used. Now we can decide whether there is a well-maintained correlating feature for poorly maintained features. And if that is the case, then the bad one is deleted. Of the 39 features, only 9 remain.")
        st.image("images/remaining features.png", width=300)

    if subpage==subpages[1]:    
        st.write("#### Handling missing values:")
        
        st.write("###### CO2-Emission (Ewltp (g/km))")
        st.write("We dropped all 6365 NaN rows of the column, as they are only a small portion ( 0.18%) of the data.")
        
        st.write("###### Fuel consumption")
        st.write("After dropping the emissions NaN rows, we are left with 0.74% of the values of Fuel Consumption being NaN values, which again is a small percentage and therefore we decided to drop them entirely.")
        
        st.write("###### Engine Power (ep (KW))")
        st.write("There were only two entries left with no data, so the rows were also dropped.")

        st.write("###### Emission Reduction(Erwltp (g/km)) from Innovative Technologies(IT)")
        st.write("Look for the rows where IT is specified but Erwltp (g/km) is nan and fill the missing values with the mean of the IT group.")
        st.write("Rows with no IT specified will have no emission reduction, so we fill the nan with zeros.")
    
    if subpage==subpages[2]:
        st.write("#### Renaming features and Values:")
        st.write("###### Fuel consumption")
        st.write("The Column name of 'fuel consumption ' has one space at the end, which we rectified.")
        st.write("###### Fuel type (Ft)")
        st.write("For the fuel type of ng-biomethane exists a second value NG, so we changed that value also in ng-biomethane.")
            
    if subpage==subpages[3]:
        st.write("#### One hot-encoding:")
        st.write("###### Inovative technology (IT)")
        st.write("The values of IT changed to 1 if an inovative technology was given and 0 if not.")
        st.write("###### Man, Cr and Fuel Type")
        st.write("The remaining three categorical features Man, Cr and Fuel Type were changed into indicator variables. Each value gets its own column.")

    if subpage==subpages[4]:
        st.write("#### Class distribution and Emission labels")
        col1, col2 = st.columns([2,2])
        value_counts = df_preprocessed["target_clf"].value_counts()
        df_target = value_counts.reset_index()
        df_target.columns = ['Class', 'Num of Instances']
        df_target=df_target.sort_values(by="Class")
    
        df_target["Label"] = np.array(["A","B","C","D","E","F","G"])
        df_target.set_index('Class', inplace=True)
        col1.dataframe(df_target)
        col2.image("images/classes.png")

if page == pages[3] :
    st.write("## Modeling")
    subpages=["Classification", "Regression", "All Models (Summary)"]
    subpage=st.sidebar.radio("Choose Model Type:", subpages)
                     
    if subpage == subpages[0]:
        st.write("### Classification")
        
        choice = st.selectbox('Choose the model', classification_models)

        if choice =="Logistic Regression":
            with st.expander("Mathematical background:"):
                st.write("##### Linear combination of input features:")
                st.latex(r'''
                z = \mathbf{w}^T \mathbf{x} + b
                ''')
                st.latex(r'''
                \mathbf{w}\text{: weights, } \mathbf{x} \text{: input features, } b \text{: bias}
                ''')
                st.write("##### Probability calculation using the softmax function (multiclass):")
                st.latex(r'''
                p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}
                ''')
                st.latex(r'''
                p_k \text{: probability of class } k \text{, } K \text{: total number of classes}
                ''')
                st.write("##### Cost function to minimize:")
                st.write(" Cross-entropy loss")
                st.latex(r'''
                J(\mathbf{w}) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_{i,k} \log(p_{i,k}) + \frac{1}{2C} \sum_{k=1}^K \|\mathbf{w}_k\|^2
                ''')
                st.latex(r'''
                y\text{: actual class, } p \text{: predicted probability, } C \text{: inverse of regularization strength}
                ''')

            with st.expander("Parameter:"):
                st.markdown('''
                            * penalty='l2'
                            * dual=False
                            * tol=0.0001
                            * C=1.0
                            * fit_intercept=True
                            * intercept_scaling=1
                            * class_weight=None
                            * random_state=None
                            * solver='sag'
                            * max_iter=1000
                            * multi_class='auto'
                            * verbose=0
                            * warm_start=False
                            * n_jobs=None
                            * l1_ratio=None
                            ''')

        elif choice == "K-Nearest Neighbors":
            with st.expander("Parameter:"):
                st.markdown('''
                            *   n_neighbors=3
                            *   weights='uniform'
                            *   algorithm='auto'
                            *   leaf_size=30
                            *   p=2
                            *   metric='minkowski'
                            *   metric_params=None
                            *   n_jobs=None
                            ''')
            st.write("Unfortunatly the model is too big to be used.")
            st.write("See All Models (Summary) for the computed scores.")
                
        elif choice == "Decision Tree":
            with st.expander("Parameter:"):
                st.markdown('''
                            * criterion='entropy'
                            * splitter='best'
                            * max_depth=4
                            * min_samples_split=2
                            * min_samples_leaf=1
                            * min_weight_fraction_leaf=0.0
                            * max_features=None
                            * random_state=123
                            * max_leaf_nodes=None
                            * min_impurity_decrease=0.0
                            * class_weight=None
                            * ccp_alpha=0.0
                            ''')
                
        elif choice == "XGBoost":
            with st.expander("Parameter:"):
                st.markdown('''
                            * max_depth=6
                            * learning_rate=0.3
                            * n_estimators=100
                            * verbosity=1
                            * objective='multi:softmax'
                            * num_class=7
                            * booster='gbtree'
                            * tree_method='auto'
                            * n_jobs=1
                            * gamma=0
                            * min_child_weight=1
                            * max_delta_step=0
                            * subsample=1
                            * colsample_bytree=1
                            * colsample_bylevel=1
                            * colsample_bynode=1
                            * reg_alpha=0
                            * reg_lambda=1
                            * scale_pos_weight=1
                            * base_score=0.5
                            * random_state=0
                            * missing=None
                            * importance_type='gain'
                            ''')
            
        elif choice=="XGBoost optimized":
            # Load clf
            clf = classifiers[choice]

            st.write("The XGBoost model was chosen for optimization, as it gives the best results.")
            with st.expander("Parameter:"):
                col1, col2 = st.columns(2)
                col1.write("##### Optimization parameter:")
                col1.dataframe({'learning_rate': [0.1, 0.01, 0.001],
                    'max_depth': [3, 5, 7],
                    'n_estimators': [100, 500, 1000]})

                col2.write("##### GridSearchCV parameter:")
                col2.dataframe({'objective': 'multi:softmax',
                    'cv': 3,
                    'scoring': 'accuracy'})
                
                st.write("##### Obtained best-paratmeters from GridSearchCV are:")
                params = clf.get_params()
                st.dataframe({'learning_rate': [params.get('learning_rate')],
                    'max_depth': [params.get('max_depth')],
                    'n_estimators': [params.get('n_estimators')]})
                    
            with st.expander("Feature importance"):        
                featimp_score = clf.feature_importances_
                st. write("Ft: Fuel type, Man: Manufacturer")
                st.pyplot(featimp_fig(featimp_score))
        
        elif choice=="Neural Network":

            with st.expander("Model Layout:"):
                st.image("images/NN_layers_clf.png")

            with st.expander("Parameter:"):
                st.write("##### Callback function parameter:")
                col1, col2 = st.columns(2)
                col1.write("###### EarlyStopping")
                col1.dataframe({'monitor': 'val_loss',
                'patience': 5,
                'min_delta': 0.01,
                'mode': 'min',
                'verbose': 1})
                col2.write("###### ReduceLROnPlateau")
                col2.dataframe({'monitor': 'val_loss',
                'factor' : 0.1,
                'patience': 5,
                'min_delta': 0.01,
                'cooldown' : 4,
                'verbose': 1})

                st.write("##### Compiling parameter:")
                st.dataframe({'loss': 'sparse_categorical_crossentropy',
                'optimizer': "adam",
                'metrics': 'accuracy'})

        elif choice=="Neural Network optimized":
            # Load clf
            clf = classifiers[choice]

            with st.expander("Model Layout:"):
                st.image("images/NN_layers_clf.png")
            
            with st.expander("Parameter:"):
                st.write("##### Callback function parameter:")
                col1, col2 = st.columns(2)
                col1.write("###### EarlyStopping")
                col1.dataframe({'monitor': 'val_loss',
                'patience': 5,
                'min_delta': 0.01,
                'mode': 'min',
                'verbose': 1})
                col2.write("###### ReduceLROnPlateau")
                col2.dataframe({'monitor': 'val_loss',
                'factor' : 0.1,
                'patience': 5,
                'min_delta': 0.01,
                'cooldown' : 4,
                'verbose': 1})

                st.write("##### Compiling parameter:")
                st.dataframe({'loss': 'sparse_categorical_crossentropy',
                'metrics': 'accuracy'})

                col1, col2 = st.columns(2)
                col1.write("##### Optimization parameter:")
                col1.dataframe({'learning_rate': [0.01,0.001,0.0001,"","","",""],
                    'optimizers': ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"],
                    'batch_size': [512, 256, 128,"","","",""]})
                
                col2.write("##### GridSearchCV parameter:")
                col2.dataframe({'cv': 3,
                'scoring': 'accuracy'})
                
                st.write("##### Obtained best-paratmeters from GridSearchCV are:")
                optimizer = clf.optimizer
                st.dataframe({'learning_rate': [f"{optimizer.learning_rate.numpy():.4f}"],
                    'optimizer': [optimizer.__class__.__name__],
                    'batch_size': [128]})

        if not choice == "K-Nearest Neighbors":
            y_test_pred = get_predictions_clf[choice]
            display = st.radio('What do you want to show?', ('Classification report','Classification report (imbalanced)', 'Confusion matrix'))
            if display == 'Classification report':
                st.dataframe(classification_report(y_test_clf, y_test_pred, output_dict=True))
            elif display == 'Classification report (imbalanced)':
                st.dataframe(classification_report_imbalanced(y_test_clf, y_test_pred, output_dict=True))
            elif display == 'Confusion matrix':
                st.dataframe(confusion_matrix(y_test_clf, y_test_pred))

    if subpage == subpages[1]:
        st.write("### Regression")
        choice = st.selectbox('Choose the model', regression_models)
        y_test_pred = get_predictions_reg[choice]

        if choice == "Linear Regression":
            with st.expander("Mathematical background:"):
                st.write("##### Linear combination of input features:")
                st.latex(r'''
                \hat{y} = \mathbf{w}^T \mathbf{x} + b
                ''')
                st.latex(r'''
                \mathbf{w}\text{: weights, } \mathbf{x} \text{: input features, } b \text{: bias, }\hat{y} \text{: predicted value}
                ''')
                st.write("##### Cost function to minimize:")
                st.write(" Mean Squared Error (MSE) ")
                st.latex(r'''
                J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)^2
                ''')
                st.latex(r'''
                y: \text{actual value, } \hat{y}: \text{predicted value}
                ''')

            with st.expander("Parameter:"):
                st.markdown('''
                            *   copy_X=True
                            *   fit_intercept=True
                            *   n_jobs=None
                            *   positive=False
                            ''')
                         
        elif choice == "Elastic Net":
            with st.expander("Mathematical background:"):
                st.write("##### Linear combination of input features:")
                st.latex(r'''
                z = \mathbf{w}^T \mathbf{x} + b
                ''')
                st.latex(r'''\mathbf{w}\text{: weights, } \mathbf{x} \text{: input features, } b \text{: bias}''')

                st.write("##### Cost Function:")
                st.write("Mean Squared Error (MSE) with L1 and L2 penalties")
                st.latex(r'''
                J(\mathbf{w}) = \frac{1}{2} MSE + \alpha \left( \text{l1\_ratio} \| \mathbf{w} \|_1 + \frac{1 - \text{l1\_ratio}}{2} \| \mathbf{w} \|_2^2 \right)
                \\
                \\
                y\text{: actual value, }
                \hat{y} \text{: predicted value, } 
                \alpha \text{: regularization strength, }  
                \\
                \text{l1\_ratio} \text{: balance between L1 and L2 penalties}
                \\
                \| \mathbf{w} \|_{1/2} \text{: L1/2 norm (sum of the absolute values of the weights)}         
                ''')

            with st.expander("Parameter:"):
                st.markdown('''
                            *   alpha=1.0
                            *   l1_ratio=0.5
                            *   fit_intercept=True
                            *   normalize='deprecated'
                            *   precompute=False
                            *   max_iter=1000
                            *   copy_X=True
                            *   tol=0.0001
                            *   warm_start=False
                            *   positive=False
                            *   random_state=None
                            *   selection='cyclic'
                            ''')
                
        elif choice == "Decision Tree":
            with st.expander("Parameter:"):
                st.markdown('''
                            * criterion='gini'
                            * splitter='best'
                            * max_depth=5
                            * min_samples_split=2
                            * min_samples_leaf=1
                            * min_weight_fraction_leaf=0.0
                            * max_features=None
                            * random_state=None
                            * max_leaf_nodes=None
                            * min_impurity_decrease=0.0
                            * class_weight=None
                            * ccp_alpha=0.0
                            ''')
                    
        elif choice == "XGBoost":
            with st.expander("Parameter:"):
                st.markdown('''
                            *   max_depth=6
                            *   learning_rate=0.3
                            *   n_estimators=100
                            *   verbosity=1
                            *   objective='reg:squarederror'
                            *   booster='gbtree'
                            *   tree_method='auto'
                            *   n_jobs=1
                            *   gamma=0
                            *   min_child_weight=1
                            *   max_delta_step=0
                            *   subsample=1
                            *   colsample_bytree=1
                            *   colsample_bylevel=1
                            *   colsample_bynode=1
                            *   reg_alpha=0
                            *   reg_lambda=1
                            *   scale_pos_weight=1
                            *   base_score=0.5
                            *   random_state=0
                            *   missing=None
                            *   importance_type='gain'
                            ''')
                
        elif choice == "XGBoost optimized":
            # Load reg
            reg = regressors[choice]
            
            with st.expander("Parameter:"):
                col1, col2 = st.columns(2)

                col1.write("##### Optimization parameter:")
                col1.dataframe({'learning_rate': [0.1, 0.01, 0.001],
                    'max_depth': [3, 5, 7],
                    'n_estimators': [100, 500, 1000]})

                col2.write("##### GridSearchCV parameter:")
                col2.dataframe({'objective': 'reg:squarederror',
                    'cv': 3,
                    'scoring': 'neg_mean_squared_error'})
                
                st.write("##### Obtained best-paratmeters from GridSearchCV are:")
                params = reg.get_params()
                st.dataframe({'learning_rate': [params.get('learning_rate')],
                    'max_depth': [params.get('max_depth')],
                    'n_estimators': [params.get('n_estimators')]})
                
            with st.expander("Feature importance:"): 
                featimp_score = reg.feature_importances_
                st. write("Ft: Fuel type, Man: Manufacturer, ep: Engine power")
                st.pyplot(featimp_fig(featimp_score))

        elif choice=="Neural Network":
            with st.expander("Model Layout:"):
                st.image("images/NN_layers_reg.png")

            with st.expander("Parameter:"):
                st.write("##### Callback function parameter:")
                col1, col2 = st.columns(2)
                col1.write("###### EarlyStopping")
                col1.dataframe({'monitor': 'val_loss',
                'patience': 5,
                'min_delta': 0.01,
                'mode': 'min',
                'verbose': 1})
                col2.write("###### ReduceLROnPlateau")
                col2.dataframe({'monitor': 'val_loss',
                'factor' : 0.1,
                'patience': 5,
                'min_delta': 0.01,
                'cooldown' : 4,
                'verbose': 1})

                st.write("##### Compiling parameter:")
                st.dataframe({'loss': 'mse',
                'optimizer': "adam",
                'metrics': 'mean_absolute_error'})

        elif choice=="Neural Network optimized":
            # Load reg
            reg = regressors[choice]
                
            with st.expander("Model Layout:"):
                st.image("images/NN_layers_reg.png")

            with st.expander("Parameter:"):
                st.write("##### Callback function parameter:")
                col1, col2 = st.columns(2)
                col1.write("###### EarlyStopping")
                col1.dataframe({'monitor': 'val_loss',
                'patience': 5,
                'min_delta': 0.01,
                'mode': 'min',
                'verbose': 1})
                col2.write("###### ReduceLROnPlateau")
                col2.dataframe({'monitor': 'val_loss',
                'factor' : 0.1,
                'patience': 5,
                'min_delta': 0.01,
                'cooldown' : 4,
                'verbose': 1})

                st.write("##### Compiling parameter:")
                st.dataframe({'loss': 'mse',
                'metrics': 'mean_absolute_error'})

                col1, col2 = st.columns(2)
                col1.write("##### Optimization parameter:")
                col1.dataframe({'learning_rate': [0.001,0.0001,"","","","",""],
                    'optimizers': ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"],
                    'batch_size': [512, 256, 128,"","","",""]})
                
                col2.write("##### GridSearchCV parameter:")
                col2.dataframe({'objective': 'reg:squarederror',
                'cv': 3,
                'scoring': 'neg_mean_squared_error'})
                
                st.write("##### Obtained best-paratmeters from GridSearchCV are:")
                optimizer = reg.optimizer
                st.dataframe({'learning_rate': [f"{optimizer.learning_rate.numpy():.4f}"],
                    'optimizer': [optimizer.__class__.__name__],
                    'batch_size': [128]})

        display = st.radio('What do you want to show?', ('R2 Score', 'RMSE'))
        if display == 'R2 Score':
            st.write(r2_score(y_test_reg, y_test_pred))
        elif display == 'RMSE':
            st.write(np.sqrt(mean_squared_error(y_test_reg, y_test_pred)))

    if subpage == subpages[2]:
        st.write("### Summary")

        st.write("##### Classification")
        col11, col12 = st.columns([1,8])

        # Display the DataFrame
        df_summary_clf = get_summary_clf()
        col12.dataframe(df_summary_clf.sort_values(by="F1 Score"))

        st.write("##### Regression")
        col21, col22 = st.columns([1,8])

        # Display the DataFrame
        df_summary_reg = get_summary_reg()
        col22.dataframe(df_summary_reg.sort_values(by="R2"))

if page == pages[4] : 
    st.write("## Interpretation")

    st.write("We use Lime to explain our optimized XGBoost Classifier.")
    
    # Define the function to explain
    @st.cache_data
    def proba_func(X):
        proba = classifiers['XGBoost optimized'].predict_proba(X)
        print(proba)
        return proba

    with st.expander("Choose from list of falsely predicted instances:"):
            # Find mismatch indices
            y_test_clf_ary = y_test_clf.values.flatten()
            y_test_pred = get_predictions_clf["XGBoost optimized"]
            y_test_pred = np.array(y_test_pred)
            mismatch_indices = np.where(y_test_clf_ary != y_test_pred)[0]

             # Display the mismatched instances and their predictions
            true_classes=[]
            predicted_classes=[]
            for idx in mismatch_indices:
                true_classes.append(y_test_clf_ary[idx])
                predicted_classes.append(y_test_pred[idx])
            df_mismatch = pd.DataFrame()
            df_mismatch["Index"]=mismatch_indices
            print(df_mismatch)
            df_mismatch["True Class"]=true_classes
            df_mismatch["Predicted Class"]=predicted_classes
            st.dataframe(df_mismatch)


    idx = st.text_input("Choose an instance to explain [0 - {}]:".format(len(X_test)), value="")
    if idx == "":
        st.write('No instance is selected for interpretation!')
    else:
        idx = int(idx)
        instance = X_test.iloc[idx].values.reshape(1, -1)

        # Explain the instance
        explanation = explainer.explain_instance(instance[0], proba_func, num_features=10)  

        col1, col2, col3, col4 = st.columns([3,2,2,2])
        col1.write("###### Predicted Class:")
        proba = np.round(proba_func(instance), 4)
        probability = pd.DataFrame()
        probability['Label'] = np.array(["A","B","C","D","E","F","G"])
        probability['Class'] = np.array([0,1,2,3,4,5,6])        
        probability['Probability'] = proba.T
        probability = probability.set_index('Label', drop=True)
        col1.dataframe(probability)

        col2.write("###### Predicted label:")
        pred_class = probability.loc[probability["Probability"].idxmax(),"Class"]
        col2.image(get_label_image(pred_class),width=40)

        col3.write("###### True class:")
        true_class = y_test_clf.iloc[idx].values[0]
        col3.text(true_class)

        col4.write("###### True label:")
        col4.image(get_label_image(true_class),width=40)

        # Display explanation
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)
        plt.clf()

        with st.expander("Data of instance"):
            st.write(f"###### Instance id = {idx}:v")
            st.dataframe(pd.DataFrame(instance, columns=X_test.columns))

            

            
