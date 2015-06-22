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
    
    // Declare objects that support the process of images detecting
    private FeatureDetector fDetector;
    private DescriptorExtractor dExtractor;
    private DescriptorMatcher dMatcher;

    // A list of all training photos
    private List<Mat> photoLib= new ArrayList<Mat>();
    
    // A hash map that stores the URI of training images
    // is used if library is loaded from a list of training URI
    private HashMap<Mat,Uri> uriLib= new HashMap<Mat,Uri>();   

    // A hash map that stores the ID (from resource folder) of training images
    // is used if library is loaded from a list of training ID
    private HashMap<Mat,Integer> resIDLib= new HashMap<Mat,Integer>();    

    // A hash map that stores the path of training images
    // is used if library is loaded from a list of training paths
    private HashMap<Mat,String> pathLib= new HashMap<Mat,String>();
    
    // The number of photos in the library
    private int lib_size;
    
    // The index of training image used to query
    // used for testing
    private int query_index;
         	
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
        			mBgr= new Mat();
        			
        			fDetector = FeatureDetector.create
        					(FeatureDetector.FAST);
        			dExtractor = DescriptorExtractor.create
        					(DescriptorExtractor.ORB);
        			dMatcher= DescriptorMatcher.create
        					(DescriptorMatcher.BRUTEFORCE_HAMMING);
        			
        			lib_size= 10;
        			query_index=3; // query an image in the library
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
//    		// Detecting the query image
//    		// The query image is initialized at BaseLoaderCallback 
//    		Mat result = detectPhoto(photoLib.get(query_index)); 
//    		displayPhoto(pathLib.get(result));
    		runExperiment();
    	}
    	if(mIsLoadingLib){
    		mIsLoadingLib=false;        
//    		// Load training images from some sources
//        	long start= System.currentTimeMillis();    		
//    		loadLibFromDevice();        
//        	long end= System.currentTimeMillis();
//        	Log.i(TAG, "Runtime to load photoLib: "+ (end-start));    		
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
            File gpxfile = new File(root, "data_scale.txt");
            FileWriter writer = new FileWriter(gpxfile);
            
	    	HashMap<Mat, String> nameLib = new HashMap<Mat,String>();
	    	int number_of_buildings =10;
	    	int number_of_angles =5;
	    	int variation_of_distance=4;
	    	MatOfDMatch matches= new MatOfDMatch();
	    	
	    	//// Build the library    	
	    	long start= System.currentTimeMillis();
	    	// load using image paths from device
	    	for (int a = 0; a < 1 ; a++) {
		    	for (int b = 0; b < number_of_buildings ; b++) {
		    		int d=1;
					String fileName= b+"_"+a+"_"+d+".jpg";
					String photoPath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)+
							"/Research/database/" + fileName;
					Mat img= Imgcodecs.imread(photoPath);

			    	// scale down images
			    	Mat resized_img= new Mat();
			    	Size size= new Size(img.size().width/2, img.size().height/2);
			    	Imgproc.resize(img, resized_img, size);
			    	
					photoLib.add(resized_img);
					nameLib.put(resized_img, fileName);
				}    	
	    	}
	    	// get the list of descriptors for the list of images
	    	List<Mat> descriptor_list= descriptorList(photoLib);    	
	    	// add descriptors to train a descriptor collection
	    	dMatcher.add(descriptor_list);      
	    	long done_building_lib= System.currentTimeMillis();
	    	
	    	Log.i(TAG, "Runtime to build library: "+ (done_building_lib - start)); 
	    	writer.append("Runtime to build library: "+ (done_building_lib - start)+ "\n");
	    	
	    	//// Detect photos 
			for (int a = 0; a < number_of_angles ; a++) {
				for (int d = 0; d < variation_of_distance ; d++) {
			    	int countCorrectMatch =0;
			    	for (int b = 0; b < number_of_buildings ; b++) {
			    		// load the query image
			    		long startD = System.currentTimeMillis();
			    		
				    	String fileName= b+"_"+a+"_"+d+".jpg";
						String photoPath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)+
								"/Research/database/" + fileName;
						Mat img= Imgcodecs.imread(photoPath);
						
						Mat queryImg= new Mat();
				    	Size size= new Size(img.size().width/2, img.size().height/2);
				    	Imgproc.resize(img, queryImg, size);
						
				    	// get descriptors of the query image    	
				    	Mat query_descriptors = imgDescriptor(queryImg);		
				    	// Match the descriptors of a query image
				    	dMatcher.match(query_descriptors, matches);    	
				    	// filter good matches
				    	List<DMatch> good_matches = filterGoodMatches(matches.toList());    	
				    	// find the image that matches the most
				    	Mat bestMatch = findBestMatch(good_matches);  
				    	
				    	String matchName = nameLib.get(bestMatch);
				    	String[] buildingNo= matchName.split("_");
				    	int bNo = Integer.parseInt(buildingNo[0]);
				    	
				    	if(bNo == b){
				    		countCorrectMatch++;
				    	}else{
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
    
    // Method that displays the image and its features 
    // on the device's screen
    public void drawFeatures(Mat rgba){
    	MatOfKeyPoint keyPoints = new MatOfKeyPoint();    	
    	Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_RGBA2RGB);
    	fDetector.detect(rgba, keyPoints);
    	Features2d.drawKeypoints(rgba,keyPoints,rgba);
    	Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_RGB2RGBA);
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
    
    // Method that loads training library from resource folder,
    // required restricted naming of image files
    public void loadLibFromRes()
    {
    	// load by image ID from resource folder
	    try {
			for (int j = 0; j < lib_size; j++) {
				int id= getResources().getIdentifier
						("jpg_photo_"+j, "drawable", getPackageName());
				Mat img= Utils.loadResource(this, id);
				photoLib.add(img);
				resIDLib.put(img,id);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
    }
    
    // Method that loads training library from device storage,
    // required restricted naming of image files
    public void loadLibFromDevice()
    {
    	// load using image paths
    	for (int j = 0; j < lib_size; j++) {
			String photoPath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)+
					"/research/test_photo_" +j+".jpg";
			Mat img= Imgcodecs.imread(photoPath);
			Log.i(TAG, "img size" + img.size());
			photoLib.add(img);
			pathLib.put(img,photoPath);
		}
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
    
    // Method that detects a given image based on the training library    
    public Mat detectPhoto(Mat rgbaQuery){
    	Log.i(TAG, "called detectFeatures");
    	MatOfDMatch matches= new MatOfDMatch();
//    	List<MatOfDMatch> match_list = new ArrayList<MatOfDMatch>();
    	
    	long start= System.currentTimeMillis();    
    	// get the list of descriptors for the list of images
    	List<Mat> descriptor_list= descriptorList(photoLib);
    	
    	// add descriptors to train a descriptor collection
    	dMatcher.add(descriptor_list);      	
//    	Log.i(TAG, "DescriptorMatcher train collection size:  "+ 
//    			dMatcher.getTrainDescriptors().size());  
    		
    	long done_building_lib= System.currentTimeMillis();
//    	Log.i(TAG, "Runtime to build dcrptLib: "+ (done_building_lib-start));
    	
    	// get descriptors of the query image
    	// detect the matrix of key points of that image
    	Mat query_descriptors = imgDescriptor(rgbaQuery);
//		Log.i(TAG, "query img ID:  "+ rgbaQuery);
//		Log.i(TAG, "query img descriptors:  "+ query_descriptors.size());
		
    	// Match the descriptors of a query image 
    	// to descriptors in the training collection.
    	dMatcher.match(query_descriptors, matches);
//    	Log.i(TAG, "matrix of matches size:  "+ matches.size());
    	
//    	// match with ratio test
//    	dMatcher.knnMatch(query_descriptors, match_list, 2);
//    	Log.i(TAG, "knnMatch, k=2, match_list size:  "+  match_list.size());
//    	List<DMatch> tested_dMatch = new ArrayList<DMatch>();
//    	for(MatOfDMatch mat: match_list)
//    	{
//    		List<DMatch> dMatches = mat.toList();
//    		Log.i(TAG, "dMatches.size:  "+  dMatches.size());
//    		if(dMatches.size() <2)
//    		{
//    			tested_dMatch.add(dMatches.get(0));
//    		}else{
//	    		Log.i(TAG, "dMatches get (0):  "+  
//	    				dMatches.get(0).distance +"  "+
//	    				dMatches.get(0).imgIdx +"  "+ 
//	    				dMatches.get(0).queryIdx +"  "+ 
//	    				dMatches.get(0).trainIdx);
//	    		Log.i(TAG, "dMatches get (1):  "+  
//	    				dMatches.get(1).distance +"  "+
//	    				dMatches.get(1).imgIdx +"  "+ 
//	    				dMatches.get(1).queryIdx +"  "+ 
//	    				dMatches.get(1).trainIdx);
//	    		if(dMatches.get(0).distance < 0.75 * dMatches.get(1).distance)
//	    		{
//	    			tested_dMatch.add(dMatches.get(0));
//	    		}
//    		}
//    	}
//    	// filter good matches
//    	List<DMatch> total_matches = tested_dMatch;
//    	List<DMatch> good_matches = tested_dMatch;
    	
    	// filter good matches
    	List<DMatch> total_matches = matches.toList();
    	List<DMatch> good_matches = filterGoodMatches(total_matches);
//    	Log.i(TAG, "list of all matches size:  "+ total_matches.size());
//    	Log.i(TAG, "list of good matches size:  "+ good_matches.size());
    	
    	// find the image that matches the most
    	Mat bestMatch = findBestMatch(good_matches);   
//    	Log.i(TAG, "bestMatch img:  "+ bestMatch);   
    	
    	long done_matching= System.currentTimeMillis();
//    	Log.i(TAG, "Runtime to match: "+ (done_matching - done_building_lib));
//    	Log.i(TAG, "finishing detectFeatures");    	
    	return bestMatch;    	
    }
        
    
    // Method that returns the list of descriptor matrices
    // associating with a given list of photos 
    public List<Mat> descriptorList(List<Mat> photoLib){
    	List<Mat> descriptor_list= new ArrayList<Mat>();
    	
    	// for each image in photoLib
    	// do the following tasks
    	for(int i=0; i<photoLib.size(); i++){
    		// detect the matrix of key points of that image
    		Mat imgDescriptor = imgDescriptor(photoLib.get(i));    		
    		descriptor_list.add(imgDescriptor); 
//    		Log.i(TAG, "descriptor_list,one img descripters size:  "+ imgDescriptor.size());
    	}    	    	
//    	Log.i(TAG, "descriptor_list size:  "+ descriptor_list.size());
    	
    	return descriptor_list;
    }
        
    // Method that returns a matrix of descriptors for a given image
    public Mat imgDescriptor(Mat img)
    {
    	Mat imgDescriptor = new Mat();
    	// detect the matrix of key points of that image
		MatOfKeyPoint imgKeyPoints = new MatOfKeyPoint();
		fDetector.detect(img, imgKeyPoints);
		
//		// set threshold to 100 (instead of 1) to reduce the number of key points
//		// not work for new opencv
//		try {
//			File outputDir = getCacheDir(); // If in an Activity (otherwise getActivity.getCacheDir();
//			File outputFile = File.createTempFile("orbDetectorParams", ".YAML", outputDir);
//			writeToFile(outputFile, "%YAML:1.0\nthreshold: 100 \nnonmaxSupression: true\n");
//			fDetector.read(outputFile.getPath());
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
		// filter the best key points
		imgKeyPoints= topKeyPoints(imgKeyPoints, 500);
		
		// compute the descriptor from those key points
		dExtractor.compute(img,imgKeyPoints, imgDescriptor);
		return imgDescriptor;
    }
    
    // Method that returns the top 'n' best key points from 
    public MatOfKeyPoint topKeyPoints(MatOfKeyPoint imgKeyPoints, int n)
    {
		Log.i(TAG, "imgKeyPoints size:  "+ imgKeyPoints.size());
		// Sort and select 500 best key points
		List<KeyPoint> listOfKeypoints = imgKeyPoints.toList();
		if(listOfKeypoints.size()<n){
			Log.i(ERROR, "The requested number of key points is less than that of given key points");
			return imgKeyPoints;
		}		
		Collections.sort(listOfKeypoints, new Comparator<KeyPoint>() {
		    @Override
		    public int compare(KeyPoint kp1, KeyPoint kp2) {
		        // Sort them in descending order, so the best response KPs will come first
		        return (int) (kp2.response - kp1.response);
		    }
		});
//		Log.i(TAG, "listOfKeypoints size:  "+ listOfKeypoints.size());
		List<KeyPoint> bestImgKeyPoints = listOfKeypoints.subList(0,n);
//		Log.i(TAG, "bestImgKeyPoints size:  "+ bestImgKeyPoints.size());
		
		MatOfKeyPoint result = new MatOfKeyPoint();
		result.fromList(bestImgKeyPoints); 
		return result;
    }
    
    // (URL Source) http://answers.opencv.org/question/3167/java-how-to-set-parameters-to-orb-featuredetector/?answer=17296#post-id-17296
    private void writeToFile(File file, String data) {
        try {
			FileOutputStream stream = new FileOutputStream(file);
			OutputStreamWriter outputStreamWriter = new OutputStreamWriter(stream);
			outputStreamWriter.write(data);
			outputStreamWriter.close();
			stream.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }

    // Method that filters good matches from given list of matches,
    // using arbitrary bounds
    private List<DMatch> filterGoodMatches(List<DMatch> total_matches)
    {
    	List<DMatch> goodMatches = new ArrayList<DMatch>();
    	double max_dist = 0; double min_dist = 100;
    	// calculate max and min distances between keypoints
    	for( DMatch dm: total_matches)
    	{ 
    		double dist = dm.distance;
    	    if( dist < min_dist ) min_dist = dist;
    	    if( dist > max_dist ) max_dist = dist;
    	}
    	for(DMatch aMatch: total_matches){
    		// *note: arbitrary constants 3 & 0.02
    		if( aMatch.distance <= Math.max(3*min_dist, 0.02)){
    			goodMatches.add(aMatch);
    		}
    	}
    	return goodMatches;
    }
    
    // Method that finds the best match from a list of matches
    private Mat findBestMatch(List<DMatch> good_matches)
    {
    	HashMap<Mat,Integer> hm= new HashMap<Mat, Integer>();
    	// count the images matched
    	for(DMatch aMatch: good_matches){    		
    		Mat trainImg = photoLib.get(aMatch.imgIdx);   
    		if(hm.get(trainImg)==null){
    			hm.put(trainImg,1);
    		}else{
    			hm.put(trainImg, hm.get(trainImg)+1);
    		}
    	}
    	
    	// search for the image that matches the largest number of descriptors.
    	Mat bestMatch= null;
    	Integer greatestCount=0;
//    	Log.i(TAG, "hashmap of matches size:  "+ hm.size());
    	for(Mat trainImg: hm.keySet()){
//    		Log.i(TAG, "train img:  "+ trainImg);
    		Integer count=hm.get(trainImg);
    		if(count> greatestCount){
    			greatestCount= count;
    			bestMatch= trainImg;
    		}
    	}
    	
    	// print result
//    	for(Mat trainImg: hm.keySet()){
//    		Log.i(TAG, "Matched img result:  "+ trainImg +
//    				", numOfMatches: "+hm.get(trainImg));
//    	}    	
    	return bestMatch;
    }
    
}
