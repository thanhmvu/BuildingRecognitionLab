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
import org.opencv.imgcodecs.Imgcodecs;
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
    private List<TrainingImage> traing_library;
    
    public ImageDetector()
    {
    	fDetector = FeatureDetector.create
				(FeatureDetector.FAST);
		dExtractor = DescriptorExtractor.create
				(DescriptorExtractor.ORB);
		dMatcher= DescriptorMatcher.create
				(DescriptorMatcher.BRUTEFORCE_HAMMING);
		traing_library= new ArrayList<TrainingImage>();
    }
    
    public void addToLibrary(String image_path, long tour_item_id)
    {
    	TrainingImage img= new TrainingImage(image_path,tour_item_id);
    	Mat imgDescriptor = imgDescriptor(img.mat());  
    	// update descriptors of the image
    	img.setDescriptors(imgDescriptor); 
    	// add image to dMacher's internal training library
    	List<Mat> descriptor_list= new ArrayList<Mat>();
    	descriptor_list.add(imgDescriptor);
    	dMatcher.add(descriptor_list); 
    	// add image to traing_library
    	traing_library.add(img);
    }

    public void clearLibrary()
    {
    	// clear ImageDetector's library
    	traing_library= new ArrayList<TrainingImage>();
    	// clear dMatcher's internal library
    	dMatcher.clear();
    }
    
    public long identifyObject(String image_path)
    {
    	TrainingImage result = detectPhoto(image_path);
    	return result.tourID();
    }
    
    // Method that detects a given image based on the training library    
    public TrainingImage detectPhoto(String query_path){
    	Log.i(TAG, "called detectFeatures");
    	long start= System.currentTimeMillis();    
    	
    	MatOfDMatch matches= new MatOfDMatch();
    	Mat rgbaQuery = Imgcodecs.imread(query_path);

    	// get descriptors of the query image
    	// detect the matrix of key points of that image
    	Mat query_descriptors = imgDescriptor(rgbaQuery);
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
    	TrainingImage bestMatch = findBestMatch(good_matches);   
    	Log.i(TAG, "bestMatch img:  "+ bestMatch.path());   
    	
    	long done_matching= System.currentTimeMillis();
    	Log.i(TAG, "Runtime to match: "+ (done_matching - start));
    	Log.i(TAG, "finishing detectFeatures");    	
    	return bestMatch;    	
    }
        
    // Method that returns a matrix of descriptors for a given image
    public Mat imgDescriptor(Mat img)
    {
    	Mat imgDescriptor = new Mat();
    	// detect the matrix of key points of that image
		MatOfKeyPoint imgKeyPoints = new MatOfKeyPoint();
		fDetector.detect(img, imgKeyPoints);

		// filter the best key points
		imgKeyPoints= topKeyPoints(imgKeyPoints, 500);
		
		// compute the descriptor from those key points
		dExtractor.compute(img,imgKeyPoints, imgDescriptor);
		return imgDescriptor;
    }
    
    // Method that returns the top 'n' best key points 
    private MatOfKeyPoint topKeyPoints(MatOfKeyPoint imgKeyPoints, int n)
    {
		Log.i(TAG, "imgKeyPoints size:  "+ imgKeyPoints.size());
		// Sort and select n best key points
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
    private TrainingImage findBestMatch(List<DMatch> good_matches)
    {
    	HashMap<TrainingImage,Integer> hm= new HashMap<TrainingImage, Integer>();
    	// count the images matched
    	for(DMatch aMatch: good_matches){    		
    		TrainingImage trainImg = traing_library.get(aMatch.imgIdx);   
    		if(hm.get(trainImg)==null){
    			hm.put(trainImg,1);
    		}else{
    			hm.put(trainImg, hm.get(trainImg)+1);
    		}
    	}
    	
    	// search for the image that matches the largest number of descriptors.
    	TrainingImage bestMatch= null;
    	Integer greatestCount=0;
    	Log.i(TAG, "hashmap of matches size:  "+ hm.size());
    	for(TrainingImage trainImg: hm.keySet()){
    		Log.i(TAG, "train img:  "+ trainImg);
    		Integer count=hm.get(trainImg);
    		if(count> greatestCount){
    			greatestCount= count;
    			bestMatch= trainImg;
    		}
    	}
    	
    	// print result
    	for(TrainingImage trainImg: hm.keySet()){
    		Log.i(TAG, "Matched img result:  "+ trainImg.path() +
    				", numOfMatches: "+hm.get(trainImg));
    	}    	
    	return bestMatch;
    }    
}
