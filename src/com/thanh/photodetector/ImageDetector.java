package com.thanh.photodetector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

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

import android.util.Log;

public class ImageDetector {
	// Declare objects that support the process of images detecting
    private FeatureDetector fDetector;
    private DescriptorExtractor dExtractor;
    private DescriptorMatcher dMatcher;
    
    private double multiplier;
    private int number_of_key_points;
    
    // tag of messages printed to LogCat
    protected static final String TAG = "ImageDetector";
    
    // tag of Error messages printed to LogCat
    protected static final String ERROR = "Error in ImageDetector";
    
    // A list of all training photos
    private List<TrainingImage> training_library;
    
//    public ImageDetector()
//    {
//    	fDetector = FeatureDetector.create
//				(FeatureDetector.FAST);
//		dExtractor = DescriptorExtractor.create
//				(DescriptorExtractor.ORB);
//		dMatcher= DescriptorMatcher.create
//				(DescriptorMatcher.BRUTEFORCE_HAMMING);
//		training_library= new ArrayList<TrainingImage>();
//		multiplier = 0.5;
//    }
    
    public ImageDetector(int detector_type, int extractor_type, int matcher_type)
    {
    	fDetector = FeatureDetector.create
				(detector_type);
		dExtractor = DescriptorExtractor.create
				(extractor_type);
		dMatcher= DescriptorMatcher.create
				(matcher_type);
		training_library= new ArrayList<TrainingImage>();
		multiplier = 0.5;
		number_of_key_points = 1000;
    }
    
    public void addToLibrary(String image_path, long tour_item_id)
    {
    	Mat img = Imgcodecs.imread(image_path);
    	Mat resized_img = resize(img);  // scale down the image	
    	TrainingImage training_img= new TrainingImage(image_path, tour_item_id, resized_img);
    	Mat imgDescriptor = imgDescriptor(training_img);  
    	
    	// add image to dMacher's internal training library
    	dMatcher.add(Arrays.asList(imgDescriptor));
    	
    	// add image to training_library    	
    	training_library.add(training_img);
    }

    public void clearLibrary()
    {
    	// clear ImageDetector's library
    	training_library= new ArrayList<TrainingImage>();
    	// clear dMatcher's internal library
    	dMatcher.clear();
    }
    
    public long identifyObject(String image_path)
    {
    	TrainingImage result = detectPhoto(image_path);
    	return result.tourID();
    }

    public Mat resize(Mat src_img)
    {
		// scale down images
		Mat resized_img= new Mat();
		Size size= new Size(src_img.size().width*multiplier, src_img.size().height*multiplier);
		Imgproc.resize(src_img, resized_img, size);
		return resized_img;
    }
	
    // variables for drawCurrentMatches method
    TrainingImage CURRENT_QUERY_IMAGE;
    TrainingImage CURRENT_RESULT_IMAGE;
    MatOfDMatch CURRENT_GOOD_MATCHES;
    
    // Method that detects a given image based on the training library    
    public TrainingImage detectPhoto(String query_path){
//    	Log.i(TAG, "called detectFeatures");   
    	
    	Mat img = Imgcodecs.imread(query_path);
    	Mat resized_img = resize(img); // scale down the query image
    	TrainingImage query_image = new TrainingImage(query_path,0,resized_img);
    	
    	// get descriptors of the query image
    	// detect the matrix of key points of that image
    	Mat query_descriptors = imgDescriptor(query_image);
//		Log.i(TAG, "query image descriptors:  "+ query_descriptors.size());
		
    	// Match the descriptors of a query image 
    	// to descriptors in the training collection.
    	MatOfDMatch matches= new MatOfDMatch();
    	dMatcher.match(query_descriptors, matches);
//    	Log.i(TAG, "matrix of matches size:  "+ matches.size());
    	
    	// filter good matches
    	List<DMatch> total_matches = matches.toList();
    	List<DMatch> good_matches = total_matches;
//    	List<DMatch> good_matches = filterGoodMatches(total_matches);
//    	Log.i(TAG, "list of all matches size:  "+ total_matches.size());
//    	Log.i(TAG, "list of good matches size:  "+ good_matches.size());

    	// find the image that matches the most
    	TrainingImage bestMatch = findBestMatch_noFilter(good_matches); 
//    	Log.i(TAG, "bestMatch image:  "+ bestMatch.pathID());   

    	// update variables for drawCurrentMatches method
    	CURRENT_QUERY_IMAGE = query_image;  
    	CURRENT_RESULT_IMAGE = bestMatch;
    	CURRENT_GOOD_MATCHES = getCurrentGoodMatches(good_matches, bestMatch);
    	
//    	Log.i(TAG, "finishing detectFeatures");    	
    	return bestMatch;    	
    }

    private MatOfDMatch getCurrentGoodMatches(List<DMatch> good_matches,TrainingImage bestMatch)
    {
    	List<DMatch> matches_of_bestMatch = new ArrayList<DMatch>();
    	// loop to filter matches of train images, which are not the bestMatch image
    	for(DMatch aMatch: good_matches){    		
    		TrainingImage trainImg = training_library.get(aMatch.imgIdx);   
    		if (trainImg == bestMatch)
    		{
    			matches_of_bestMatch.add(aMatch);
    		}
    	}
    	MatOfDMatch result = new MatOfDMatch();
    	result.fromList(matches_of_bestMatch);
    	return result;
    }
    
    public Mat drawCurrentMatches(int n)
    {
    	Mat img1 = CURRENT_QUERY_IMAGE.image();
    	MatOfKeyPoint kp1= CURRENT_QUERY_IMAGE.keyPoints();
    	Mat img2 = CURRENT_RESULT_IMAGE.image();
    	MatOfKeyPoint kp2= CURRENT_RESULT_IMAGE.keyPoints();
    	Mat result = new Mat();
    	
    	Features2d.drawMatches(img1, kp1, img2, kp2, 
    			sortedKMatches(CURRENT_GOOD_MATCHES,0,n), result);
    	return result;
    }

    public MatOfDMatch sortedKMatches(MatOfDMatch matches, int start, int end)
    {
    	List<DMatch> list = matches.toList();
    	Collections.sort(list, new Comparator<DMatch>() {
            @Override
            public int compare(final DMatch object1, final DMatch object2) {
                return (int)(object1.distance - object2.distance);
            }
           } );
    	if(list.size()<end){
    		Log.i(TAG,"Only found "+list.size()+" matches. Can't return "+end);
    		end = list.size();
    	}
    	List<DMatch> subllist = list.subList(start, end);
    	MatOfDMatch result = new MatOfDMatch();
    	result.fromList(subllist);
    	return result;
    }
    
    // Method that returns a matrix of descriptors for a given image
    public Mat imgDescriptor(TrainingImage train_img)
    {
    	Mat img = train_img.image();
    	Mat imgDescriptor = new Mat();
    	// detect the matrix of key points of that image
		MatOfKeyPoint imgKeyPoints = new MatOfKeyPoint();
		fDetector.detect(img, imgKeyPoints);

		// filter the best key points
		imgKeyPoints= topKeyPoints(imgKeyPoints, number_of_key_points);

		// compute the descriptor from those key points
		dExtractor.compute(img,imgKeyPoints, imgDescriptor);
		train_img.setKeyPoints(imgKeyPoints);
		train_img.setDescriptors(imgDescriptor);
		return imgDescriptor;
    }
    
    // Method that returns the top 'n' best key points 
    private MatOfKeyPoint topKeyPoints(MatOfKeyPoint imgKeyPoints, int n)
    {
//		Log.i(TAG, "imgKeyPoints size:  "+ imgKeyPoints.size());
		// Sort and select n best key points
		List<KeyPoint> listOfKeypoints = imgKeyPoints.toList();
		if(listOfKeypoints.size()<n){
			Log.i(ERROR, "There are not enough "+n+" key points, only "+listOfKeypoints.size());
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
    		//	(!) WARNING:	hard code, arbitrary constants 3 & 0.02
    		if( aMatch.distance <= Math.max(3*min_dist, 0.02)){
    			goodMatches.add(aMatch);
    		}
    	}
    	return goodMatches;
    }
    
    HashMap<TrainingImage, Integer> CURRENT_MATCH_FREQUENCY;
    
    private TrainingImage findBestMatch_noFilter(List<DMatch> good_matches)
    {
    	HashMap<TrainingImage,Integer> hm= new HashMap<TrainingImage, Integer>();
    	// count the images matched
    	for(DMatch aMatch: good_matches){    		
    		TrainingImage trainImg = training_library.get(aMatch.imgIdx);   
    		if(hm.get(trainImg)==null){
    			hm.put(trainImg,1);
    		}else{
    			hm.put(trainImg, hm.get(trainImg)+1);
    		}
    	}
    	CURRENT_MATCH_FREQUENCY = hm;
    	// search for the image that matches the largest number of descriptors.
    	TrainingImage bestMatch= null;
    	Integer greatestCount=0;
//    	Log.i(TAG, "hashmap of matches size:  "+ hm.size());
    	for(TrainingImage trainImg: hm.keySet()){
//    		Log.i(TAG, "train img:  "+ trainImg);
    		Integer count=hm.get(trainImg);
    		if(count> greatestCount){
    			greatestCount= count;
    			bestMatch= trainImg;
    		}
    	}
    	
    	// print result
//    	for(TrainingImage trainImg: hm.keySet()){
//    		Log.i(TAG, "Matched img result:  "+ trainImg.pathID() +
//    				", numOfMatches: "+hm.get(trainImg));
//    	}    
    	
    	return bestMatch;    	
    }    
    
    // Method that finds the best match from a list of matches
    private TrainingImage findBestMatch(List<DMatch> good_matches)
    {
    	HashMap<TrainingImage,Integer> hm= new HashMap<TrainingImage, Integer>();
    	// count the images matched
    	for(DMatch aMatch: good_matches){    		
    		TrainingImage trainImg = training_library.get(aMatch.imgIdx);   
    		if(hm.get(trainImg)==null){
    			hm.put(trainImg,1);
    		}else{
    			hm.put(trainImg, hm.get(trainImg)+1);
    		}
    	}
    	CURRENT_MATCH_FREQUENCY = hm;
    	// search for the image that matches the largest number of descriptors.
    	TrainingImage bestMatch= null;
    	Integer greatestCount=0;
    	Integer secondGreatestCount=0;
//    	Log.i(TAG, "hashmap of matches size:  "+ hm.size());
    	for(TrainingImage trainImg: hm.keySet()){
//    		Log.i(TAG, "train img:  "+ trainImg);
    		Integer count=hm.get(trainImg);
    		if(count> greatestCount){
    			secondGreatestCount = greatestCount;
    			greatestCount= count;
    			bestMatch= trainImg;
    		}else if(count> secondGreatestCount){
    			secondGreatestCount = count;
    		}
    	}
    	
    	// print result
//    	for(TrainingImage trainImg: hm.keySet()){
//    		Log.i(TAG, "Matched img result:  "+ trainImg.pathID() +
//    				", numOfMatches: "+hm.get(trainImg));
//    	}    	
    	if (greatestCount > 1.5*secondGreatestCount){
    		return bestMatch;
    	}else{
    		Log.i(TAG, "Found no best match for the query image!");
    		return null;
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
    
    public void drawFeatures2(Mat rgba,MatOfKeyPoint keyPoints){
    	Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_RGBA2RGB);
    	Features2d.drawKeypoints(rgba,keyPoints,rgba);
    	Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_RGB2RGBA);
    }    
}
