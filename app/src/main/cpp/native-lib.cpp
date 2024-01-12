#include <jni.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video.hpp>
#include "android/bitmap.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <iostream> // Librería estándar para leer y escribir datos en la consola, obtener los errores y loggs
#include <cstdlib> // Librería estándar para generación de números aleatorios, manejo de memoria, etc.
#include <android/log.h>

#include <cstring> // Tiene métodos para el manejo de cadenas de texto
#include <cmath> // Define las funciones matemáticas

#include <random> // Librería para generación de números aleatorios
#include <vector> // Librería para definir arreglos dinámicos
#include <sstream> // Librería para conversión de datos y manejo de flujos
#include <fstream> // Librería para manejar flujos de datos (archivos)

#include <filesystem> // Librería que contiene las funciones para listar archivos y carpetas

// Librerías de OpenCV
#include <opencv2/core/core.hpp> // Contiene las definiciones básicas de las matrices que representan imágenes y otras estructuras
#include <opencv2/highgui/highgui.hpp> // Contiene las definiciones y funciones para crear GUIs
#include <opencv2/imgproc/imgproc.hpp> // Permite realizar procesamiento de imágenes
#include <opencv2/imgcodecs/imgcodecs.hpp> // Permite gestionar los códecs para lectura de formatos gráficos
#include <opencv2/video/video.hpp> // Permite reproducir archivos de vídeo
#include <opencv2/videoio/videoio.hpp> // Permite almacenar vídeos

// DNN Module
#include <opencv2/dnn/dnn.hpp>
//ONNX Module


cv::dnn::Net neuralNetwork;


std::vector<std::string> etiquetas;
const float IMG_WIDTH = 640.0;
const float IMG_HEIGHT = 640.0;
const float CLASS_PROBABILITY = 0.5;
const float NMS_THRESHOLD = 0.5;
const float CONFIDENCE_THRESHOLD = 0.5;
const int NUMBER_OF_OUTPUTS = 85;
cv::Scalar BLACK = cv::Scalar(0,0,0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0,0,255);

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

std::vector<std::string> loadCOCO(AAssetManager *assetManager, const char *filename, char sep = '\n') {
    std::vector<std::string> names;

    // Abre el archivo desde la carpeta "assets"
    AAsset *asset = AAssetManager_open(assetManager, filename, AASSET_MODE_BUFFER);
    if (asset == nullptr) {
        // Maneja el error si el archivo no se puede abrir
        std::cerr << "Error al abrir el archivo " << filename << " desde assets." << std::endl;
        return names;
    }

    // Obtiene un puntero al búfer de datos del archivo
    const void *buffer = AAsset_getBuffer(asset);

    // Crea un flujo de entrada a partir del búfer de datos
    std::istringstream bufferStream(static_cast<const char*>(buffer));

    std::string token;
    while (getline(bufferStream, token, sep)) {
        if (token.size() > 1) {
            names.push_back(token);
        }
    }
    // Cierra el archivo
    AAsset_close(asset);
    etiquetas=names;
    return names;
}

void draw_label(cv::Mat& input_image, std::string label, int left, int top){
    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = std::max(top, label_size.height);
    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle.
    rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

std::vector<cv::Mat> forwardNET(cv::Mat inputImage, cv::dnn::Net net){
    // Create a blob from the input image
    cv::Mat blob;
    cv::dnn::blobFromImage(inputImage, blob, 1./255., cv::Size(IMG_WIDTH, IMG_HEIGHT), cv::Scalar(), true, false);

    net.setInput(blob);

    // Forward pass.
    std::vector<cv::Mat> outputs;

    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}


cv::Mat filterDetections(cv::Mat inputImg, std::vector<cv::Mat> detections, const std::vector<std::string>& classNames) {
    cv::Mat inputImage = inputImg.clone();
    std::vector<int> classIDs;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    float x_factor = inputImage.cols / IMG_WIDTH;
    float y_factor = inputImage.rows / IMG_HEIGHT;
    float* pData = new float[NUMBER_OF_OUTPUTS];
    float confidence = 0.0;
    float* probValues;
    cv::Point classId;
    double maxClassProb = 0.0;
    cv::Mat probabilityClasses = cv::Mat::zeros(1, classNames.size(), CV_32FC1);
    int totalDetections = detections[0].total() / NUMBER_OF_OUTPUTS;

    for (int i = 0; i < totalDetections; ++i) {
        std::memcpy(pData, (float*) detections[0].data + (i * NUMBER_OF_OUTPUTS), NUMBER_OF_OUTPUTS * sizeof(float));
        confidence = pData[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            probValues = (pData + 5);
            probabilityClasses = cv::Mat::zeros(1, classNames.size(), CV_32FC1);
            std::memcpy(probabilityClasses.data, probValues, classNames.size() * sizeof(float));
            minMaxLoc(probabilityClasses, 0, &maxClassProb, 0, &classId);
            if (maxClassProb > CLASS_PROBABILITY) {
                confidences.push_back(confidence);
                classIDs.push_back(classId.x);
                boxes.push_back(cv::Rect(int((pData[0] - 0.5 * pData[2]) * x_factor), int((pData[1] - 0.5 * pData[3]) * y_factor),
                                         int(pData[2] * x_factor), int(pData[3] * y_factor)));
            }
        }
    }

    std::vector<int> indices;
    std::string label = "";
    cv::dnn::NMSBoxes(boxes, confidences, CLASS_PROBABILITY, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        cv::rectangle(inputImage, boxes[indices[i]], BLUE, 3 * THICKNESS);
        label = cv::format("%.2f", confidences[indices[idx]]);
        label = classNames[classIDs[indices[i]]] + ":" + label;
        draw_label(inputImage, label, boxes[indices[i]].x, boxes[indices[i]].y);
    }
    return inputImage;
}





extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_afinal_MainActivity_loadCOCO(JNIEnv *env, jobject instance,
                                                    jobject asset_manager) {const char *filename = "coco.names";

    // Obtiene el puntero al AssetManager desde el objeto Java
    AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);

    // Llama a la función loadLabelsCOCO para cargar las etiquetas desde assets
    std::vector<std::string> labels = loadCOCO(mgr, filename);

    // Convierte el vector de cadenas C++ a un array de cadenas Java
    jobjectArray result = env->NewObjectArray(labels.size(), env->FindClass("java/lang/String"), nullptr);
    for (size_t i = 0; i < labels.size(); ++i) {
        env->SetObjectArrayElement(result, i, env->NewStringUTF(labels[i].c_str()));
    }

    return result;
}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_MainActivity_Model(JNIEnv *env, jobject instance, jobject asset_manager,
                                            jstring model_path) {
    AAssetManager *mgr = AAssetManager_fromJava(env,asset_manager);
    const char *modelPathStr = env->GetStringUTFChars(model_path, nullptr);
    AAsset *modelAsset = AAssetManager_open(mgr, modelPathStr, AASSET_MODE_BUFFER);
    env->ReleaseStringUTFChars(model_path, modelPathStr);
    if (modelAsset != nullptr) {
        off_t modelSize = AAsset_getLength(modelAsset);
        const void *modelData = AAsset_getBuffer(modelAsset);
        const uchar* modelDataUChar = static_cast<const uchar*>(modelData);
        std::vector<uchar> modelBuffer(modelDataUChar, modelDataUChar + modelSize);
        AAsset_close(modelAsset);
        cv::Mat modelMat(modelBuffer);
        __android_log_print(ANDROID_LOG_INFO, "Elvis6", "Cargando el modelo");

        neuralNetwork = cv::dnn::readNetFromONNX(modelMat);
        if (!neuralNetwork.empty()) {
            // El modelo se cargó correctamente, puedes imprimir un mensaje
            __android_log_print(ANDROID_LOG_INFO, "Elvis6", "Cargo el modelo");
        } else {
            __android_log_print(ANDROID_LOG_ERROR, "Elvis6", "No carga modelo 2");
        }
    }  else {
        __android_log_print(ANDROID_LOG_ERROR, "Elvis6", "No carga modelo 1");
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_MainActivity_Detect(JNIEnv *env, jobject instance, jlong addr_rgba) {
    cv::Mat* inputFrame = (cv::Mat*)addr_rgba;

    //std::vector<std::string> nombresCapas = neuralNetwork.getLayerNames();

    //nombresCapas = neuralNetwork.getUnconnectedOutLayersNames();

    std::vector<cv::Mat> detections;
    detections = forwardNET(*inputFrame, neuralNetwork);

    filterDetections(*inputFrame,detections, etiquetas);

}