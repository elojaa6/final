#include <jni.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <android/log.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unistd.h>
using namespace std;
using namespace cv;
using namespace dnn;

// Constant definitions
const float IMG_WIDTH = 640.0;
const float IMG_HEIGHT = 640.0;
const float CLASS_PROBABILITY = 0.5;
const float NMS_THRESHOLD = 0.5;
const float CONFIDENCE_THRESHOLD = 0.5;
const int NUMBER_OF_OUTPUTS = 6;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Variables globales
Net neuralNetwork;
vector<string> clases;


Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);


vector<string> loadLabelsCOCO(const string& path, char sep = '\n') {
    vector<string> names;

    string token = "";
    ifstream buffer(path, ios::in);

    while (getline(buffer, token, sep)) {
        if (token.size() > 1)
            names.push_back(token);
    }

    return names;
}

void draw_label(Mat& input_image, string label, int left, int top){
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}


vector<Mat> forwardNET(Mat inputImage, Net net){

    Mat blob;
    blobFromImage(inputImage, blob, 1./255., Size(IMG_WIDTH, IMG_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    vector<Mat> outputs;

    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

Mat filterDetections(Mat inputImg, vector<Mat> detections, const vector<string> clases){
    // Initialize vectors to hold respective outputs while unwrapping detections.
    Mat inputImage=inputImg.clone();
    vector<int> classIDs;
    vector<float> confidences;
    vector<Rect> boxes;

    // Resizing factor.
    float x_factor = inputImage.cols / IMG_WIDTH;
    float y_factor = inputImage.rows / IMG_HEIGHT;
    float *pData = new float[NUMBER_OF_OUTPUTS]; // = (float *)detections[0].data;
    float confidence = 0.0;
    float *probValues;
    Point classId;
    double maxClassProb = 0.0;

    //Mat scores(1, clases.size(), CV_32FC1, classes_scores);
    Mat probabilityClasses = Mat::zeros(1, clases.size(), CV_32FC1);

    // Calculate the number of detections. It is important to note that the NUMBER_OF_OUTPUTS
    // depends of the outputs of the neural network. In the case of YOLOv5 is 85 outpus (variables).
    int totalDetections = detections[0].total() / NUMBER_OF_OUTPUTS;
    cout << "Total of detections = " << totalDetections << endl;

    // According to the documentation, the total of detections for YOLOv5 is 25200
    // for default image size 640.

    // Iterate through 25200 detections.
    for (int i = 0; i < totalDetections; ++i){
        std::memcpy(pData, (float *) detections[0].data+(i*NUMBER_OF_OUTPUTS), NUMBER_OF_OUTPUTS*sizeof(float));
        confidence = pData[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD){
            probValues = (pData + 5);
            // Create a 1x85 Mat and store class scores of 80 classes.
            probabilityClasses = Mat::zeros(1, clases.size(), CV_32FC1);
            std::memcpy(probabilityClasses.data, probValues, clases.size()*sizeof(float));

            // Perform minMaxLoc and acquire the index of best class  score.
            minMaxLoc(probabilityClasses, 0, &maxClassProb, 0, &classId);
            // Continue if the class score is above the threshold.
            if (maxClassProb > CLASS_PROBABILITY){
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                classIDs.push_back(classId.x);
                boxes.push_back(Rect(int((pData[0]-0.5*pData[2])*x_factor),int((pData[1]-0.5*pData[3])*y_factor),
                                     int(pData[2]*x_factor), int(pData[3]*y_factor)));

            }
        }
    }


    // Perform Non-Maximum Suppression and draw predictions.
    vector<int> indices;
    string label = "";
    NMSBoxes(boxes, confidences, CLASS_PROBABILITY, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++){
        // Draw the bounding box arround detected object
        rectangle(inputImage, boxes[indices[i]], BLUE, 3*THICKNESS);
        // Get the label for the class name and its confidence.
        label = format("%.2f", confidences[indices[i]]);
        label = clases[classIDs[indices[i]]] + ":" + label;
        // Draw class labels.
        draw_label(inputImage, label, boxes[indices[i]].x, boxes[indices[i]].y);
    }
    return inputImage;
}

long getMemoryUsage() {
    ifstream statm("/proc/self/statm");
    if (!statm.is_open()) {
        return -1;  // Error al abrir el archivo
    }

    long size, resident, share, text, lib, data, dt;
    statm >> size >> resident >> share >> text >> lib >> data >> dt;
    statm.close();

    return resident * sysconf(_SC_PAGESIZE) / 1024;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_MainActivity_processCameraFrame(JNIEnv *env, jobject thiz, jlong inputMatAddr) {

    // Obtener el objeto Mat desde la dirección pasada como argumento
    Mat &frameMat = *(Mat *)inputMatAddr;
    cvtColor(frameMat, frameMat, cv::COLOR_RGBA2RGB);

    // Medir el tiempo de inicio para calcular FPS
    auto startTime = chrono::high_resolution_clock::now();

    // Procesar la imagen con la red
    vector<Mat> detections = forwardNET(frameMat, neuralNetwork);
    Mat processedImg = filterDetections(frameMat, detections, clases);

    // Medir el tiempo de finalización y calcular FPS
    auto endTime = chrono::high_resolution_clock::now();
    auto elapsedTime = chrono::duration_cast<chrono::duration<double>>(endTime - startTime).count();
    double fps = 1.0 / elapsedTime;  // FPS = 1 / tiempo transcurrido en segundos

    // Medir el uso de memoria RAM
    long ramUsage = getMemoryUsage();
    double ramUsageMB = static_cast<double>(ramUsage) / 1024.0;  // Convertir a MB

    // Mostrar el tiempo de inferencia
    string label = format("Inference time: %.2f s", elapsedTime);
    putText(processedImg, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

    // Mostrar los FPS con dos decimales
    string fpsLabel = format("FPS: %.2f", fps);
    putText(processedImg, fpsLabel, Point(20, 80), FONT_FACE, FONT_SCALE, RED);

    // Mostrar el uso de memoria RAM
    string ramLabel = format("RAM Usage: %.2f MB", ramUsageMB);
    putText(processedImg, ramLabel, Point(20, 120), FONT_FACE, FONT_SCALE, RED);

    // Copiar el resultado procesado de vuelta al objeto Mat original
    processedImg.copyTo(frameMat);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_MainActivity_loadNetwork(JNIEnv *env, jobject thiz, jstring absolute_path) {
    // Convertir jstring a const char*
    const char *path = env->GetStringUTFChars(absolute_path, 0);

    // Leer la red
    neuralNetwork = cv::dnn::readNet(path);

    // Liberar la memoria asignada por GetStringUTFChars
    env->ReleaseStringUTFChars(absolute_path, path);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_MainActivity_loadCoco(JNIEnv *env, jobject thiz, jstring absolute_path) {
    // Convertir jstring a const char*
    const char *path = env->GetStringUTFChars(absolute_path, 0);

    // Llamar a la función para cargar las clases COCO
    clases = loadLabelsCOCO(path);

    // Liberar la memoria asignada por GetStringUTFChars
    env->ReleaseStringUTFChars(absolute_path, path);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Frutilla_processCameraFrameFrutilla(JNIEnv *env, jobject thiz,
                                                            jlong input_mat_addr) {
    // Obtener el objeto Mat desde la dirección pasada como argumento
    Mat &frameMat = *(Mat *)input_mat_addr;
    cvtColor(frameMat, frameMat, cv::COLOR_RGBA2RGB);

    // Medir el tiempo de inicio para calcular FPS
    auto startTime = chrono::high_resolution_clock::now();

    // Procesar la imagen con la red
    vector<Mat> detections = forwardNET(frameMat, neuralNetwork);
    Mat processedImg = filterDetections(frameMat, detections, clases);

    // Medir el tiempo de finalización y calcular FPS
    auto endTime = chrono::high_resolution_clock::now();
    auto elapsedTime = chrono::duration_cast<chrono::duration<double>>(endTime - startTime).count();
    double fps = 1.0 / elapsedTime;  // FPS = 1 / tiempo transcurrido en segundos

    // Medir el uso de memoria RAM
    long ramUsage = getMemoryUsage();
    double ramUsageMB = static_cast<double>(ramUsage) / 1024.0;  // Convertir a MB

    // Mostrar el tiempo de inferencia
    string label = format("Inference time: %.2f s", elapsedTime);
    putText(processedImg, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

    // Mostrar los FPS con dos decimales
    string fpsLabel = format("FPS: %.2f", fps);
    putText(processedImg, fpsLabel, Point(20, 80), FONT_FACE, FONT_SCALE, RED);

    // Mostrar el uso de memoria RAM
    string ramLabel = format("RAM Usage: %.2f MB", ramUsageMB);
    putText(processedImg, ramLabel, Point(20, 120), FONT_FACE, FONT_SCALE, RED);

    // Copiar el resultado procesado de vuelta al objeto Mat original
    processedImg.copyTo(frameMat);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Frutilla_loadNetworkFrutilla(JNIEnv *env, jobject thiz,
                                                     jstring absolute_path) {
    // Convertir jstring a const char*
    const char *path = env->GetStringUTFChars(absolute_path, 0);

    // Leer la red
    neuralNetwork = cv::dnn::readNet(path);

    // Liberar la memoria asignada por GetStringUTFChars
    env->ReleaseStringUTFChars(absolute_path, path);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Frutilla_loadCocoFrutilla(JNIEnv *env, jobject thiz,
                                                  jstring absolute_path) {
    // Convertir jstring a const char*
    const char *path = env->GetStringUTFChars(absolute_path, 0);

    // Llamar a la función para cargar las clases COCO
    clases = loadLabelsCOCO(path);

    // Liberar la memoria asignada por GetStringUTFChars
    env->ReleaseStringUTFChars(absolute_path, path);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Banana_processCameraFrameBanana(JNIEnv *env, jobject thiz,
                                                        jlong input_mat_addr) {
    // Obtener el objeto Mat desde la dirección pasada como argumento
    Mat &frameMat = *(Mat *)input_mat_addr;
    cvtColor(frameMat, frameMat, cv::COLOR_RGBA2RGB);

    // Medir el tiempo de inicio para calcular FPS
    auto startTime = chrono::high_resolution_clock::now();

    // Procesar la imagen con la red
    vector<Mat> detections = forwardNET(frameMat, neuralNetwork);
    Mat processedImg = filterDetections(frameMat, detections, clases);

    // Medir el tiempo de finalización y calcular FPS
    auto endTime = chrono::high_resolution_clock::now();
    auto elapsedTime = chrono::duration_cast<chrono::duration<double>>(endTime - startTime).count();
    double fps = 1.0 / elapsedTime;  // FPS = 1 / tiempo transcurrido en segundos

    // Medir el uso de memoria RAM
    long ramUsage = getMemoryUsage();
    double ramUsageMB = static_cast<double>(ramUsage) / 1024.0;  // Convertir a MB

    // Mostrar el tiempo de inferencia
    string label = format("Inference time: %.2f s", elapsedTime);
    putText(processedImg, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

    // Mostrar los FPS con dos decimales
    string fpsLabel = format("FPS: %.2f", fps);
    putText(processedImg, fpsLabel, Point(20, 80), FONT_FACE, FONT_SCALE, RED);

    // Mostrar el uso de memoria RAM
    string ramLabel = format("RAM Usage: %.2f MB", ramUsageMB);
    putText(processedImg, ramLabel, Point(20, 120), FONT_FACE, FONT_SCALE, RED);

    // Copiar el resultado procesado de vuelta al objeto Mat original
    processedImg.copyTo(frameMat);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Banana_loadNetworkBanana(JNIEnv *env, jobject thiz, jstring absolute_path) {
    // Convertir jstring a const char*
    const char *path = env->GetStringUTFChars(absolute_path, 0);

    // Leer la red
    neuralNetwork = cv::dnn::readNet(path);

    // Liberar la memoria asignada por GetStringUTFChars
    env->ReleaseStringUTFChars(absolute_path, path);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Banana_loadCocoBanana(JNIEnv *env, jobject thiz, jstring absolute_path) {
    // Convertir jstring a const char*
    const char *path = env->GetStringUTFChars(absolute_path, 0);

    // Llamar a la función para cargar las clases COCO
    clases = loadLabelsCOCO(path);

    // Liberar la memoria asignada por GetStringUTFChars
    env->ReleaseStringUTFChars(absolute_path, path);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Uva_processCameraFrameUva(JNIEnv *env, jobject thiz, jlong input_mat_addr) {
    // Obtener el objeto Mat desde la dirección pasada como argumento
    Mat &frameMat = *(Mat *)input_mat_addr;
    cvtColor(frameMat, frameMat, cv::COLOR_RGBA2RGB);

    // Medir el tiempo de inicio para calcular FPS
    auto startTime = chrono::high_resolution_clock::now();

    // Procesar la imagen con la red
    vector<Mat> detections = forwardNET(frameMat, neuralNetwork);
    Mat processedImg = filterDetections(frameMat, detections, clases);

    // Medir el tiempo de finalización y calcular FPS
    auto endTime = chrono::high_resolution_clock::now();
    auto elapsedTime = chrono::duration_cast<chrono::duration<double>>(endTime - startTime).count();
    double fps = 1.0 / elapsedTime;  // FPS = 1 / tiempo transcurrido en segundos

    // Medir el uso de memoria RAM
    long ramUsage = getMemoryUsage();
    double ramUsageMB = static_cast<double>(ramUsage) / 1024.0;  // Convertir a MB

    // Mostrar el tiempo de inferencia
    string label = format("Inference time: %.2f s", elapsedTime);
    putText(processedImg, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

    // Mostrar los FPS con dos decimales
    string fpsLabel = format("FPS: %.2f", fps);
    putText(processedImg, fpsLabel, Point(20, 80), FONT_FACE, FONT_SCALE, RED);

    // Mostrar el uso de memoria RAM
    string ramLabel = format("RAM Usage: %.2f MB", ramUsageMB);
    putText(processedImg, ramLabel, Point(20, 120), FONT_FACE, FONT_SCALE, RED);

    // Copiar el resultado procesado de vuelta al objeto Mat original
    processedImg.copyTo(frameMat);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Uva_loadNetworkUva(JNIEnv *env, jobject thiz, jstring absolute_path) {
    // Convertir jstring a const char*
    const char *path = env->GetStringUTFChars(absolute_path, 0);

    // Leer la red
    neuralNetwork = cv::dnn::readNet(path);

    // Liberar la memoria asignada por GetStringUTFChars
    env->ReleaseStringUTFChars(absolute_path, path);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Uva_loadCocoUva(JNIEnv *env, jobject thiz, jstring absolute_path) {
    // Convertir jstring a const char*
    const char *path = env->GetStringUTFChars(absolute_path, 0);

    // Llamar a la función para cargar las clases COCO
    clases = loadLabelsCOCO(path);

    // Liberar la memoria asignada por GetStringUTFChars
    env->ReleaseStringUTFChars(absolute_path, path);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Mango_processCameraFrameMango(JNIEnv *env, jobject thiz,
                                                      jlong input_mat_addr) {
    // Obtener el objeto Mat desde la dirección pasada como argumento
    Mat &frameMat = *(Mat *)input_mat_addr;
    cvtColor(frameMat, frameMat, cv::COLOR_RGBA2RGB);

    // Medir el tiempo de inicio para calcular FPS
    auto startTime = chrono::high_resolution_clock::now();

    // Procesar la imagen con la red
    vector<Mat> detections = forwardNET(frameMat, neuralNetwork);
    Mat processedImg = filterDetections(frameMat, detections, clases);

    // Medir el tiempo de finalización y calcular FPS
    auto endTime = chrono::high_resolution_clock::now();
    auto elapsedTime = chrono::duration_cast<chrono::duration<double>>(endTime - startTime).count();
    double fps = 1.0 / elapsedTime;  // FPS = 1 / tiempo transcurrido en segundos

    // Medir el uso de memoria RAM
    long ramUsage = getMemoryUsage();
    double ramUsageMB = static_cast<double>(ramUsage) / 1024.0;  // Convertir a MB

    // Mostrar el tiempo de inferencia
    string label = format("Inference time: %.2f s", elapsedTime);
    putText(processedImg, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

    // Mostrar los FPS con dos decimales
    string fpsLabel = format("FPS: %.2f", fps);
    putText(processedImg, fpsLabel, Point(20, 80), FONT_FACE, FONT_SCALE, RED);

    // Mostrar el uso de memoria RAM
    string ramLabel = format("RAM Usage: %.2f MB", ramUsageMB);
    putText(processedImg, ramLabel, Point(20, 120), FONT_FACE, FONT_SCALE, RED);

    // Copiar el resultado procesado de vuelta al objeto Mat original
    processedImg.copyTo(frameMat);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Mango_loadNetworkMango(JNIEnv *env, jobject thiz, jstring absolute_path) {
    // Convertir jstring a const char*
    const char *path = env->GetStringUTFChars(absolute_path, 0);

    // Leer la red
    neuralNetwork = cv::dnn::readNet(path);

    // Liberar la memoria asignada por GetStringUTFChars
    env->ReleaseStringUTFChars(absolute_path, path);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_afinal_Mango_loadCocoMango(JNIEnv *env, jobject thiz, jstring absolute_path) {
    // Convertir jstring a const char*
    const char *path = env->GetStringUTFChars(absolute_path, 0);

    // Llamar a la función para cargar las clases COCO
    clases = loadLabelsCOCO(path);

    // Liberar la memoria asignada por GetStringUTFChars
    env->ReleaseStringUTFChars(absolute_path, path);
}