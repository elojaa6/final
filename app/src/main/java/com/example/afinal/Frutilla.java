package com.example.afinal;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;

import com.example.afinal.databinding.ActivityMainBinding;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

public class Frutilla extends CameraActivity {
    // Used to load the 'afinal' library on application startup.
    static {
        System.loadLibrary("afinal");
    }

    private static String LOGTAG = "OpenCV_Log";
    private CameraBridgeViewBase mOpenCvCameraView;
    private File cascadeFile;
    private BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface.SUCCESS:{
                    Log.v(LOGTAG, "OpenCV Loaded");
                    mOpenCvCameraView.enableView();
                }break;
                default:
                {
                    super.onManagerConnected(status);
                }break;
            }

        }
    };

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());

        fileOnnx();
        fileCoco();

        setContentView(binding.getRoot());
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(cvCameraViewListener);

    }

    private void fileOnnx(){

        try{
            cascadeFile = new File(getCacheDir(), "fresasb.onnx");
            if (!cascadeFile.exists()){
                InputStream inputStream = getAssets().open("fresasb.onnx");
                FileOutputStream outputStream = new FileOutputStream(cascadeFile);
                byte[] buffer = new byte[2048];
                int bytesRead = -1;
                while((bytesRead = inputStream.read(buffer)) != -1){
                    outputStream.write(buffer, 0, bytesRead);
                }
                inputStream.close();
                outputStream.close();
            }
            loadNetworkFrutilla(cascadeFile.getAbsolutePath());
        }catch (IOException e){
            e.printStackTrace();
        }
    }

    private void fileCoco(){

        try{
            cascadeFile = new File(getCacheDir(), "frutilla.names");
            if (!cascadeFile.exists()){
                InputStream inputStream = getAssets().open("frutilla.names");
                FileOutputStream outputStream = new FileOutputStream(cascadeFile);
                byte[] buffer = new byte[2048];
                int bytesRead = -1;
                while((bytesRead = inputStream.read(buffer)) != -1){
                    outputStream.write(buffer, 0, bytesRead);
                }
                inputStream.close();
                outputStream.close();
            }
            loadCocoFrutilla(cascadeFile.getAbsolutePath());
        }catch (IOException e){
            e.printStackTrace();
        }
    }


    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList(){
        return Arrays.asList(mOpenCvCameraView);
    }

    private CameraBridgeViewBase.CvCameraViewListener2 cvCameraViewListener = new CameraBridgeViewBase.CvCameraViewListener2(){

        @Override
        public void onCameraViewStarted(int width, int height){
        }

        @Override
        public void onCameraViewStopped(){

        }

        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
            Mat mRgba = inputFrame.rgba();
            processCameraFrameFrutilla(mRgba.getNativeObjAddr());

            return mRgba;
        }
    };

    public void onPause(){
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    public void onResume(){
        super.onResume();
        if (!OpenCVLoader.initDebug()){
            Log.i(LOGTAG, "OpenCV not found, Initializing");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallBack);
        }else{
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy(){
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }
    public native void processCameraFrameFrutilla(long inputMatAddr);
    public native void loadNetworkFrutilla(String absolutePath);
    public native void loadCocoFrutilla(String absolutePath);
}