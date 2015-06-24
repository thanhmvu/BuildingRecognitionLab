package com.thanh.photodetector;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import com.example.photodetector.R;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.MediaStore.Images;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;


public class MainActivity extends Activity implements CvCameraViewListener2 {
	// tag of messages printed to LogCat
    protected static final String TAG = "MainActivity";
    
    // tag of Error messages printed to LogCat
    protected static final String ERROR = "Error in MainActivity";
    
    // Whether an asynchronous menu action is in progress.
 	// If so, menu interaction should be disabled.
 	private boolean mIsMenuLocked;
 	
 	// Whether the next camera frame should be saved as a photo.   
 	private boolean mIsTakingPhoto;  

 	// Whether the object in next camera frame should be detected
 	private boolean mIsObjectDetecting;
 	
 	// Whether the library of training images is being loaded
 	private boolean mIsLoadingLib;
 	
 	// A matrix that is used when saving photos.
 	private Mat mBgr;

 	// A camera object that allows the app to access the device's camera
    private CameraBridgeViewBase mOpenCvCameraView;    
    
    private ImageDetector detector;
    
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
        			mBgr= new Mat();
        			
        			detector = new ImageDetector();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.photo_detector_layout);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.PhotoDetectorView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.main_activity, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
    	if(mIsMenuLocked){
    		Log.i(TAG, "called onOptionsItemSelected. mIsMenuLocked:" + mIsMenuLocked);
    		return true;
    	}
    	Log.i(TAG, "onOptionsSelected.menu is not locked");
        int id = item.getItemId();
        switch(id){
        case R.id.action_settings:
            return true;
        case R.id.menu_take_photo:
        	mIsMenuLocked= true;        	
        	//Next frame, take the photo
        	mIsTakingPhoto = true;        	
        	return true;    
        case R.id.menu_detect_object:	
        	mIsMenuLocked= true;      	
        	//Next frame, detect the photo
        	mIsObjectDetecting =true;
        	return true;
        case R.id.menu_load_library:
        	mIsLoadingLib= true;
        	return true;
        default:
        	return super.onOptionsItemSelected(item);
        }
    }
    
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
    	Mat rgba= inputFrame.rgba();
    	
    	if(mIsTakingPhoto){
    		mIsTakingPhoto= false;
    		// Save the image and retrieve its URI
    		savePhoto(rgba);
    	}    	
    	if(mIsObjectDetecting){
    		mIsObjectDetecting=false;
    		runExperiment();
    	}
    	if(mIsLoadingLib){
    		mIsLoadingLib=false;    		
    	}
    	return rgba;
    }
    
    // Method that save the give image to open storage in the device
    private void savePhoto(final Mat rgba) { 
		// Determine the path and metadata for the photo. 
		final long currentTimeMillis = System.currentTimeMillis(); 
		final String appName = getString(R.string.app_name); 
		final String galleryPath = 
				Environment.getExternalStoragePublicDirectory( 
						Environment.DIRECTORY_PICTURES).toString(); 
		final String albumPath = galleryPath + File.separator + appName; 
		final String photoPath = albumPath + File.separator + 
				currentTimeMillis + ".png"; 
		final ContentValues values = new ContentValues();  
		values.put(MediaStore.MediaColumns.DATA, photoPath); 
		values.put(Images.Media.TITLE, appName); 
		values.put(Images.Media.DESCRIPTION, appName); 
		values.put(Images.Media.DATE_TAKEN, currentTimeMillis); 
		
		// Ensure that the album directory exists. 
		File album = new File(albumPath); 
		if (!album.isDirectory() && !album.mkdirs()) { 
			Log.e(TAG, "Failed to create album directory at"+ 
					albumPath); 
			onSavePhotoFailed(); 
			return; 
		}  
		
		// Try to create the photo. 
		Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR, 3); 
		if (!Imgcodecs.imwrite(photoPath, mBgr)) {
			Log.e(TAG, "Failed to save photo to " + photoPath);
			onSavePhotoFailed(); 
		} 
		Log.d(TAG, "Photo saved successfully to " + photoPath);
		
		// Try to insert the photo into the MediaStore.
		Uri uri;
		try { 
			uri = getContentResolver().insert( 
					Images.Media.EXTERNAL_CONTENT_URI, values); 
		} catch (final Exception e) { 
			Log.e(TAG, "Failed to insert photo into MediaStore"); 
			e.printStackTrace(); 
			
			// Since the insertion failed, delete the photo.
			File photo = new File (photoPath);
			if (!photo.delete()) {
				Log.e(TAG, "Failed to delete non-inserted photo"); 
			}			
			onSavePhotoFailed(); 
			return;
		}
		
		// Open the photo in DisplayResultActivity.
        final Intent intent = new Intent(this, DisplayResultActivity.class);
        intent.putExtra(DisplayResultActivity.EXTRA_PHOTO_URI,uri);
        intent.putExtra(DisplayResultActivity.EXTRA_PHOTO_PATH,photoPath);
        startActivity(intent);
	}
	
	private void onSavePhotoFailed() { 
		mIsMenuLocked = false; 
		// Show an error message. 
		final String errorMessage = 
				getString(R.string.photo_error_message); 
		runOnUiThread(new Runnable() { 
			@Override public void run() { 
				Toast.makeText(MainActivity.this, errorMessage,
						Toast.LENGTH_SHORT).show(); 
			}
		});
	}
	
	
    @Override
    public void onPause()
    {
        super.onPause();
        Log.i(TAG, "called onPause");
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        Log.i(TAG, "called onResume");
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, 
        		this, mLoaderCallback);
        // reopen menu in case it was locked
		mIsMenuLocked = false; 
    }
    
    public void onDestroy() {
        super.onDestroy();
        Log.i(TAG, "called onDestroy");
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public void runExperiment()
    {
		try
        {
            File root = new File(Environment.getExternalStorageDirectory(), "Research");
            if (!root.exists()) {
                root.mkdirs();
            }
            File gpxfile = new File(root, "data_0623.txt");
            FileWriter writer = new FileWriter(gpxfile);
            
	    	int number_of_buildings =10;
	    	int number_of_angles =5;
	    	int variation_of_distance=4;
	    	
	    	//// Build the library    	
	    	long start= System.currentTimeMillis();
	    	// load using image paths from device
	    	for (int a = 0; a < 1 ; a++) {
		    	for (int b = 0; b < number_of_buildings ; b++) {
		    		int d=1;
					String fileName= b+"_"+a+"_"+d+".jpg";
					String photoPath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)+
							"/Research/database/" + fileName;
					detector.addToLibrary(photoPath, b);
				}    	
	    	}
	    	long done_building_lib= System.currentTimeMillis();
	    	
	    	Log.i(TAG, "Runtime to build library: "+ (done_building_lib - start)); 
	    	writer.append("Runtime to build library: "+ (done_building_lib - start)+ "\n");
	    	
	    	//// Detect photos 
			for (int a = 0; a < number_of_angles ; a++) {
				for (int d = 0; d < 1 ; d++) {
			    	int countCorrectMatch =0;
			    	for (int b = 0; b < number_of_buildings ; b++) {
			    		// load the query image
			    		long startD = System.currentTimeMillis();
			    		
				    	String fileName= b+"_"+a+"_"+d+".jpg";
						String query_path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)+
								"/Research/database/" + fileName;
						TrainingImage result = detector.detectPhoto(query_path);

				    	if(result.tourID() == b){
				    		countCorrectMatch++;
				    	}else{
				    		String matchName = new File(result.pathID()).getName();
				    		Log.i(TAG, "Mismatched: "+fileName+" with "+matchName);
				    		writer.append( fileName+"|"+matchName +"; ");
				    	}
				    	
				    	long endD =System.currentTimeMillis();
				    	Log.i(TAG, "Runtime to detect 1 image: "+(endD-startD));    
			    	}
			    	double accuracy = (double)countCorrectMatch*100/number_of_buildings ;
			    	Log.i(TAG, "a"+a+"_d"+d+", accuracy: "+accuracy+"%");    
			    	writer.append("\n"+"a"+a+"_d"+d+" "+accuracy+"%"+"\n");
			    	writer.flush();
				}
			}
			writer.close();
//          Toast.makeText(this, "Saved", Toast.LENGTH_SHORT).show();	        
        }
        catch(IOException e)
        {
             e.printStackTrace();
        }
    }

    // Method that displays a given photo on the screen,
    // passing the URI to DisplayResultActivity
    public void displayPhoto(Uri photoUri)
    {
    	// Open the photo in DisplayResultActivity.
        final Intent intent = new Intent(this, DisplayResultActivity.class);
        intent.putExtra(DisplayResultActivity.EXTRA_PHOTO_URI,photoUri);
        startActivity(intent);
    }
    
    // Method that displays a given photo on the screen,
    // passing the ID to DisplayResultActivity
    public void displayPhoto(int ID)
    {
    	// Open the photo in DisplayResultActivity.
        final Intent intent = new Intent(this, DisplayResultActivity.class);
        intent.putExtra(DisplayResultActivity.EXTRA_PHOTO_ID,ID);
        startActivity(intent);
    }
    
    // Method that displays a given photo on the screen,
    // passing the path to DisplayResultActivity
    public void displayPhoto(String photoPath)
    {
    	// Open the photo in DisplayResultActivity.
        final Intent intent = new Intent(this, DisplayResultActivity.class);
        intent.putExtra(DisplayResultActivity.EXTRA_PHOTO_PATH,photoPath);
        startActivity(intent);
    }
    
    // Method that returns the path from an URI
    // (URL Source) http://stackoverflow.com/questions/20067508/get-real-path-from-uri-android-kitkat-new-storage-access-framework    
    public String getPath(Uri uri) 
        {
            String[] projection = { MediaStore.Images.Media.DATA };
            Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
            if (cursor == null) return null;
            int column_index =             
            		cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            cursor.moveToFirst();
            String s=cursor.getString(column_index);
            cursor.close();
            return s;
        }
    
}
