package com.thanh.photodetector;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import android.util.Log;

public class ImageDetector {
	// Declare objects that support the process of images detecting
    private FeatureDetector fDetector;
    private DescriptorExtractor dExtractor;
    private DescriptorMatcher dMatcher;
    
    // tag of messages printed to LogCat
    protected static final String TAG = "ImageDetector";
    
    // tag of Error messages printed to LogCat
    protected static final String ERROR = "Error in ImageDetector";
    
    // A list of all training photos
    private List<Mat> photoLib= new ArrayList<Mat>();
    
    public ImageDetector()
    {
    	fDetector = FeatureDetector.create
				(FeatureDetector.FAST);
		dExtractor = DescriptorExtractor.create
				(DescriptorExtractor.ORB);
		dMatcher= DescriptorMatcher.create
				(DescriptorMatcher.BRUTEFORCE_HAMMING);
    }
    

    // Method that detects a given image based on the training library    
    public Mat detectPhoto(Mat rgbaQuery){
    	Log.i(TAG, "called detectFeatures");
    	MatOfDMatch matches= new MatOfDMatch();

    	long start= System.currentTimeMillis();    
    	// get the list of descriptors for the list of images
    	List<Mat> descriptor_list= descriptorList(photoLib);
    	
    	// add descriptors to train a descriptor collection
    	dMatcher.add(descriptor_list);      	
    	Log.i(TAG, "DescriptorMatcher train collection size:  "+ 
    			dMatcher.getTrainDescriptors().size());  
    		
    	long done_building_lib= System.currentTimeMillis();
    	Log.i(TAG, "Runtime to build dcrptLib: "+ (done_building_lib-start));
    	
    	// get descriptors of the query image
    	// detect the matrix of key points of that image
    	Mat query_descriptors = imgDescriptor(rgbaQuery);
		Log.i(TAG, "query img ID:  "+ rgbaQuery);
		Log.i(TAG, "query img descriptors:  "+ query_descriptors.size());
		
    	// Match the descriptors of a query image 
    	// to descriptors in the training collection.
    	dMatcher.match(query_descriptors, matches);
    	Log.i(TAG, "matrix of matches size:  "+ matches.size());
    	
    	// filter good matches
    	List<DMatch> total_matches = matches.toList();
    	List<DMatch> good_matches = filterGoodMatches(total_matches);
    	Log.i(TAG, "list of all matches size:  "+ total_matches.size());
    	Log.i(TAG, "list of good matches size:  "+ good_matches.size());
    	
    	// find the image that matches the most
    	Mat bestMatch = findBestMatch(good_matches);   
    	Log.i(TAG, "bestMatch img:  "+ bestMatch);   
    	
    	long done_matching= System.currentTimeMillis();
    	Log.i(TAG, "Runtime to match: "+ (done_matching - done_building_lib));
    	Log.i(TAG, "finishing detectFeatures");    	
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
    		Log.i(TAG, "descriptor_list,one img descripters size:  "+ imgDescriptor.size());
    	}    	    	
    	Log.i(TAG, "descriptor_list size:  "+ descriptor_list.size());
    	
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
		Log.i(TAG, "listOfKeypoints size:  "+ listOfKeypoints.size());
		List<KeyPoint> bestImgKeyPoints = listOfKeypoints.subList(0,n);
		Log.i(TAG, "bestImgKeyPoints size:  "+ bestImgKeyPoints.size());
		
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
    	Log.i(TAG, "hashmap of matches size:  "+ hm.size());
    	for(Mat trainImg: hm.keySet()){
    		Log.i(TAG, "train img:  "+ trainImg);
    		Integer count=hm.get(trainImg);
    		if(count> greatestCount){
    			greatestCount= count;
    			bestMatch= trainImg;
    		}
    	}
    	
    	// print result
    	for(Mat trainImg: hm.keySet()){
    		Log.i(TAG, "Matched img result:  "+ trainImg +
    				", numOfMatches: "+hm.get(trainImg));
    	}    	
    	return bestMatch;
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
    
    
}
