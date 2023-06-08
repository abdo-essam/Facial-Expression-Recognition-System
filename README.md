# Facial-Expression-Recognition-System
Facial Expression Recognition System using the JAFFE database. However, there are a few issues with the code that need to be addressed

1.	The paths for the JAFFE database and the shape predictor model are incorrect. You need to provide the correct paths to these files. Make sure the jaffe_dir_path and landmarks_predictor_model variables contain the correct paths to the respective files.
2.	The code is missing the definition of the detect_eyes function, which is used in the preprocessing function. You need to include the detect_eyes function in your code or import it from another module.
3.	The code assumes that the JAFFE database images are grayscale, but it's good to double-check the image format and ensure they are indeed grayscale. If the images are in color, you might need to convert them to grayscale using cv2.cvtColor function.
4.	The code includes a plotting section where it attempts to display the preprocessed images. However, the code is missing the necessary import statement for the matplotlib.pyplot module. You need to add import matplotlib.pyplot as plt at the beginning of the code.
5.	The code uses the pywt library for wavelet transform, but it seems that the from_2d_to_1d function is just reshaping the images into a 1D array without performing any wavelet transform. If you want to apply wavelet transform, you need to modify the code accordingly.
6.	The code fits a PCA model to the preprocessed data, but it's unclear how many components are selected (n_components=35). You might need to experiment with different values or use techniques like cross-validation to determine the optimal number of components.
7.	The code uses GridSearchCV to find the best parameters for the SVM model, but it seems that the param_grid dictionary is missing the 'kernel' parameter. You need to add 'kernel': ['linear'] to the param_grid dictionary.
8.	The code trains the SVM model and evaluates its performance using accuracy, confusion matrix, and classification report. However, it's essential to keep in mind that these metrics might not be the best choice for imbalanced datasets or when dealing with specific classification goals. It's recommended to consider other evaluation metrics or techniques depending on the specific requirements of your application.
Please review and update the code according to these suggestions before running it.

