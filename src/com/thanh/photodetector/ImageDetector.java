package com.thanh.photodetector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import android.location.Location;
import android.util.Log;

public class ImageDetector {
	// Declare objects that support the process of images detecting
    private FeatureDetector fDetector;
    private DescriptorExtractor dExtractor;
    private DescriptorMatcher dMatcher;
    
    int max_side;
    double filter_ratio;
    private int number_of_key_points;
    
    // tag of messages printed to LogCat
    protected static final String TAG = "ImageDetector";
    
    // tag of Error messages printed to LogCat
    protected static final String ERROR = "Error in ImageDetector";
    
    // A list of all training photos
    private List<TrainingImage> training_library;
    
    public ImageDetector(int detector_type, int extractor_type, int matcher_type)
    {
    	fDetector = FeatureDetector.create
				(detector_type);
		dExtractor = DescriptorExtractor.create
				(extractor_type);
		dMatcher= DescriptorMatcher.create
				(matcher_type);
		training_library= new ArrayList<TrainingImage>();
		max_side = 300;
		number_of_key_points = 1000;
		filter_ratio = 5;
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
		double h = src_img.size().height;
		double w = src_img.size().width;
		double multiplier = max_side/Math.max(h,w);
		Size size= new Size(w*multiplier, h*multiplier);
		Imgproc.resize(src_img, src_img, size);
		return src_img;
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
    	TrainingImage bestMatch = findBestMatch(good_matches, query_image); 
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
//		imgKeyPoints= topKeyPoints(imgKeyPoints, number_of_key_points);

		Log.i(TAG, "imgKeyPoints size:  "+ imgKeyPoints.size());
		CURRENT_NUMBER_OF_FEATURES = (int)imgKeyPoints.size().height;
		
		// compute the descriptor from those key points
		dExtractor.compute(img,imgKeyPoints, imgDescriptor);
		train_img.setKeyPoints(imgKeyPoints);
		train_img.setDescriptors(imgDescriptor);
		return imgDescriptor;
    }
    
    public Mat imgDescriptor_rgb(TrainingImage train_img)
    {
    	Mat img = train_img.image();
    	Mat imgDescriptor = new Mat();
    	// detect the matrix of key points of that image
		MatOfKeyPoint imgKeyPoints = new MatOfKeyPoint();
		fDetector.detect(img, imgKeyPoints);

		// filter the best key points
//		imgKeyPoints= topKeyPoints(imgKeyPoints, number_of_key_points);

		Log.i(TAG, "imgKeyPoints size:  "+ imgKeyPoints.size());
		CURRENT_NUMBER_OF_FEATURES = (int)imgKeyPoints.size().height;

		// compute the descriptor from those key points
		//Using RGB channels to describe
		Mat img_r = new Mat(img.rows(), img.cols(), CvType.CV_8UC1);
		Mat img_g = new Mat(img.rows(), img.cols(), CvType.CV_8UC1);
		Mat img_b = new Mat(img.rows(), img.cols(), CvType.CV_8UC1);
		double[] rgb;
		for(int x=0; x < img.cols();x++){
			for(int y=0; y < img.rows(); y++){
				rgb = img.get(y,x);
				img_r.put(y, x, new double[]{rgb[0]});
				img_g.put(y, x, new double[]{rgb[1]});
				img_b.put(y, x, new double[]{rgb[2]});				
			}
		}

    	Mat imgDescriptor_r = new Mat();
    	Mat imgDescriptor_g = new Mat();
    	Mat imgDescriptor_b = new Mat();
		dExtractor.compute(img_r,imgKeyPoints, imgDescriptor_r);
		dExtractor.compute(img_g,imgKeyPoints, imgDescriptor_g);
		dExtractor.compute(img_b,imgKeyPoints, imgDescriptor_b);
//		Log.i(TAG, "imgDescriptor_r size:  "+ imgDescriptor_r.size());
//		Log.i(TAG, "imgDescriptor_g size:  "+ imgDescriptor_g.size());
//		Log.i(TAG, "imgDescriptor_b size:  "+ imgDescriptor_b.size());

		Mat imgDescriptor_x3 = new Mat();
		List<Mat> lmat = Arrays.asList(
				imgDescriptor_r.submat(0,imgDescriptor_r.rows(),0,16),
				imgDescriptor_g.submat(0,imgDescriptor_g.rows(),0,16),
				imgDescriptor_b.submat(0,imgDescriptor_b.rows(),0,16));
		Core.hconcat(lmat, imgDescriptor_x3);
		Log.i(TAG, "imgDescriptor_x3 size:  "+ imgDescriptor_x3.size());
		imgDescriptor = imgDescriptor_x3;
		img_r.release();
		img_g.release();
		img_b.release();
		imgDescriptor_r.release();
		imgDescriptor_g.release();
		imgDescriptor_b.release();
		
		train_img.setKeyPoints(imgKeyPoints);
		train_img.setDescriptors(imgDescriptor);
		return imgDescriptor;
    }
    
    int CURRENT_NUMBER_OF_FEATURES = 0;
    
    // Method that returns the top 'n' best key points 
    private MatOfKeyPoint topKeyPoints(MatOfKeyPoint imgKeyPoints, int n)
    {
//		Log.i(TAG, "imgKeyPoints size:  "+ imgKeyPoints.size());
		// Sort and select n best key points
		List<KeyPoint> listOfKeypoints = imgKeyPoints.toList();
		if(listOfKeypoints.size()<n){
//			CURRENT_NUMBER_OF_FEATURES = listOfKeypoints.size();
			Log.i(ERROR, "There are not enough "+n+" key points, only "+listOfKeypoints.size());
			return imgKeyPoints;
		}else{	
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
//			CURRENT_NUMBER_OF_FEATURES = bestImgKeyPoints.size();
			
			MatOfKeyPoint result = new MatOfKeyPoint();
			result.fromList(bestImgKeyPoints); 
			return result;
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
    		//	(!) WARNING:	hard code, arbitrary constants 3 & 0.02
    		if( aMatch.distance <= Math.max(3*min_dist, 0.02)){
    			goodMatches.add(aMatch);
    		}
    	}
    	return goodMatches;
    }
    
    HashMap<TrainingImage, Integer> CURRENT_MATCH_FREQUENCY;
    
    private TrainingImage findBestMatch_noFilter(List<DMatch> good_matches, Location query_location)
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
    		
//    		if(CURRENT_MATCH_DISTANCES.get(trainImg)==null){
//    			CURRENT_MATCH_DISTANCES.put(trainImg,aMatch.distance+" ");
//    		}else{
//    			String updated_distances = 
//    					CURRENT_MATCH_DISTANCES.get(trainImg)+aMatch.distance+" ";
//    			CURRENT_MATCH_DISTANCES.put(trainImg, updated_distances);
//    		} 
    	}
    	
    	// location filter
    	HashMap<TrainingImage,Integer> filtered_hm = locationFilter(hm,query_location);
    	hm = filtered_hm;
    	
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
    
//    HashMap<TrainingImage, String> CURRENT_MATCH_DISTANCES 
//    	= new HashMap<TrainingImage, String>();
    
    // Method that finds the best match from a list of matches
    private TrainingImage findBestMatch(List<DMatch> good_matches, TrainingImage query_image)
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
    	
    	// location filter
    	HashMap<TrainingImage,Integer> filtered_hm = locationFilter(hm,query_image.location());
    	hm = filtered_hm;
    	
    	CURRENT_MATCH_FREQUENCY = hm;
    	// search for the image that matches the largest number of descriptors.
    	TrainingImage bestMatch= null;
		TrainingImage secondBestMatch= null;
//    	Log.i(TAG, "hashmap of matches size:  "+ hm.size());
				for(TrainingImage trainImg: hm.keySet()){
					if(bestMatch == null){
						bestMatch= trainImg;				
					}else{
						if(hm.get(trainImg)> hm.get(bestMatch)){
							secondBestMatch = bestMatch;
							bestMatch= trainImg;
						}else{
							if (secondBestMatch == null){
								secondBestMatch = trainImg;
							}else{
								if(trainImg.tourID() != bestMatch.tourID() 
										&& hm.get(trainImg)> hm.get(secondBestMatch)){
									secondBestMatch = trainImg;
								}
							}
						}
					}
				}

    	// print result
//    	for(TrainingImage trainImg: hm.keySet()){
//    		Log.i(TAG, "Matched img result:  "+ trainImg.pathID() +
//    				", numOfMatches: "+hm.get(trainImg));
//    	}

		if (secondBestMatch == null){
			return bestMatch;
		}
		else{ 
			int diff = hm.get(bestMatch)-hm.get(secondBestMatch) ;
			if ( diff * diff > filter_ratio * hm.get(bestMatch)){
				return bestMatch;
			}
			else{
				Log.i(TAG, "Found no best match for the query image!");
				return null;
			}
		}
	}

    public HashMap<TrainingImage,Integer> locationFilter(HashMap<TrainingImage,Integer> hm, Location query_location)
    {
    	if(query_location == null){
    		Log.i(TAG, "Image's location is not available");
    		return hm;
    	}else{
        	HashMap<TrainingImage,Integer> new_hm = new HashMap<TrainingImage,Integer>();
	    	for(TrainingImage trainImg: hm.keySet()){
	    		double distance = query_location.distanceTo(trainImg.location());
	    		if(distance < 50){
	    			int count = hm.get(trainImg);
	    			new_hm.put(trainImg,count);
	    		}
	    	}
	    	return new_hm;
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
